"""Evaluation metrics for EV-yielding runs.

Effectiveness: EV speed ratio, corridor clearance distance / time-to-clear.
Realism:       reaction-onset distances (empirically ~50-150 m behind, [18, 20]).
Safety:        minimum TTC, collision count.
Comfort:       p95 longitudinal deceleration / lateral acceleration.
Disruption:    background-traffic speed drop and lateral displacement.
"""
import numpy as np

from .state import YIELDING
from .simulate import History


def _clearance_trace(h: History, cap: float = 250.0):
    """Distance from the EV nose to the nearest vehicle blocking the corridor."""
    ev = h.ev
    w_need = 0.5 * (h.W[ev] + h.W) + h.params.margin_c
    out = np.full(h.n_frames, cap)
    for k in range(h.n_frames):
        dx = h.x[k] - h.x[k, ev] - 0.5 * (h.L[ev] + h.L)
        blocking = (np.abs(h.y[k] - h.y_corr) < w_need) & (dx > 0.0)
        blocking[ev] = False
        if blocking.any():
            out[k] = min(dx[blocking].min(), cap)
    return out


def _pairwise_safety(h: History, stride: int = 2):
    """Min TTC over vehicle-leader pairs with lateral body overlap; collision
    frames are pairs whose bodies overlap in both axes."""
    min_ttc, collisions = np.inf, 0
    for k in range(0, h.n_frames, stride):
        x, y, vx = h.x[k], h.y[k], h.vx[k]
        dx = x[None, :] - x[:, None]
        dy = np.abs(y[None, :] - y[:, None])
        half_L = 0.5 * (h.L[None, :] + h.L[:, None])
        half_W = 0.5 * (h.W[None, :] + h.W[:, None])
        gap = dx - half_L
        lat_overlap = dy < half_W - 0.05
        ahead = (dx > 0.0) & lat_overlap
        if (ahead & (gap < 0.0)).any():
            collisions += int((ahead & (gap < 0.0)).sum())
        dv = vx[:, None] - vx[None, :]
        closing = ahead & (dv > 0.3) & (gap > 0.0)
        if closing.any():
            min_ttc = min(min_ttc, float((gap[closing] / dv[closing]).min()))
    return min_ttc, collisions


def _reaction_distances(h: History):
    """EV-relative distance s at which each car visibly starts to yield."""
    ev = h.ev
    moved = (np.abs(h.y - h.y[0][None, :]) > 0.3) & (h.aware == YIELDING)
    moved[:, ev] = False
    out = []
    for i in range(h.x.shape[1]):
        k = np.argmax(moved[:, i])
        if moved[k, i]:
            out.append(h.x[k, i] - h.x[k, ev])
    return np.array(out)


def _oscillation(h: History):
    """Mean lateral direction reversals per yielding vehicle (jitter guard)."""
    vy = np.where(np.abs(h.vy) > 0.15, np.sign(h.vy), 0.0)
    yielded = (h.aware == YIELDING).any(axis=0)
    yielded[h.ev] = False
    if not yielded.any():
        return 0.0
    flips = 0
    for i in np.flatnonzero(yielded):
        s = vy[:, i][vy[:, i] != 0.0]
        if s.size > 1:
            flips += int((np.diff(s) != 0).sum())
    return flips / float(yielded.sum())


def evaluate(h: History, clear_target: float = 50.0) -> dict:
    ev = h.ev
    others = np.ones(h.x.shape[1], dtype=bool)
    others[ev] = False

    ev_speed = h.vx[:, ev]
    speed_ratio = float(ev_speed.mean() / h.v0[ev])

    clr = _clearance_trace(h)
    ok = clr > clear_target
    t_clear = float("nan")
    need = max(1, int(round(3.0 / max(h.t[1] - h.t[0], 1e-6))))
    run = 0
    for k in range(h.n_frames):
        run = run + 1 if ok[k] else 0
        if run >= need:
            t_clear = float(h.t[k - need + 1])
            break

    min_ttc, collisions = _pairwise_safety(h)
    react = _reaction_distances(h)

    rec_dt = float(h.t[1] - h.t[0]) if h.n_frames > 1 else 0.1
    decel = np.maximum(-h.ax[:, others], 0.0)
    ay = np.abs(np.diff(h.vy[:, others], axis=0)) / rec_dt

    mean_bg = h.vx[:, others].mean(axis=1)
    disruption = float((mean_bg[0] - mean_bg.min()) / max(mean_bg[0], 0.5))

    dy_max = np.abs(h.y - h.y[0][None, :]).max(axis=0)[others]
    yielded_any = (h.aware == YIELDING).any(axis=0)[others]

    return dict(
        name=h.name,
        ev_mean_speed=float(ev_speed.mean()),
        ev_speed_ratio=speed_ratio,
        ev_distance=float(h.x[-1, ev] - h.x[0, ev]),
        duration=float(h.t[-1] - h.t[0]),
        t_clear=t_clear,
        clearance_mean=float(clr.mean()),
        min_ttc=float(min_ttc),
        collisions=int(collisions),
        react_dist_mean=float(react.mean()) if react.size else float("nan"),
        react_dist_p10=float(np.percentile(react, 10)) if react.size else float("nan"),
        react_dist_p90=float(np.percentile(react, 90)) if react.size else float("nan"),
        p95_decel=float(np.percentile(decel, 95)),
        p95_alat=float(np.percentile(ay, 95)),
        oscillation=float(_oscillation(h)),
        disruption=disruption,
        n_yielded=int(yielded_any.sum()),
        lat_disp_mean=float(dy_max[yielded_any].mean()) if yielded_any.any() else 0.0,
        clearance_trace=clr,
        ev_speed_trace=ev_speed,
        mean_bg_speed_trace=mean_bg,
        react_dists=react,
    )


# ---------------------------------------------------------------------------
# Behavioural observables, in the form the empirical literature reports them
# (lane-change duration/rate, acceleration percentiles, median speeds) so that
# model output can be placed against naturalistic-driving statistics.
# ---------------------------------------------------------------------------
def lane_change_from_track(t, y, vy, W, lane_width=None, n_lanes=None,
                           aware=None, persist: float = 0.5,
                           settle: float = 0.6, veh: int = 0, marks=None):
    """Two-lane-occupancy lane changes of ONE trajectory, one dict per event.

    Extracted verbatim from `lane_change_events` so that the model side and the
    real-trajectory side (`emv/highd.py`) time manoeuvres with *identical code*
    rather than with two implementations of the same definition.

    A change is registered when the nearest-lane index moves and stays moved for
    `persist` seconds (hysteresis against boundary flutter). Duration is the
    two-lane-occupancy time of Thiemann et al. (2008): the body straddles the
    crossed marking while it is within half a vehicle width of it.

    **Two measurement windows are reported, and the difference matters.** The
    window start `a` was originally walked back with `abs(y[a] - y_from) <
    settle`, but at the crossing frame the vehicle is already ~half a lane width
    from the departure centre, so that condition is false immediately and `a`
    never moves: the window covers only the *arrival* half of the manoeuvre.
    `duration`/`dur_c2c`/`dy`/`peak_vy` keep that original definition so every
    number in PROJECT_LOG and papers 1-4 stays reproducible; `*_full` repeats
    the measurement with the intended walk-back (`>= settle`, i.e. walk back
    until the vehicle is inside the departure lane's settle band) and is the
    variant to compare against full-manoeuvre literature or dataset values.
    Measured difference on the model: duration 1.20 -> 3.12 s and dy 1.29 ->
    2.42 m on a 3.5 m lane, i.e. the original is a half-manoeuvre quantity.

    Geometry comes either from a uniform road (`lane_width`, `n_lanes` - the
    model) or from a measured marking array (`marks`, ascending, length
    n_lanes + 1 - highD, whose lanes are 3.1-4.4 m wide). That choice only
    changes how a lateral position maps to a lane index and where the straddled
    marking sits; the timing rule is the same. `aware=None` reports
    `yielding=False`, as real data carries no awareness state.
    """
    n = int(np.size(y))
    dt = float(t[1] - t[0]) if n > 1 else 0.1
    n_hold = max(1, int(round(persist / max(dt, 1e-6))))
    if marks is not None:
        marks = np.asarray(marks, dtype=float)
        li = np.clip(np.searchsorted(marks, y, side="right") - 1,
                     0, marks.size - 2).astype(int)
        centres = 0.5 * (marks[:-1] + marks[1:])
    else:
        li = np.clip(np.round(y / lane_width - 0.5), 0, n_lanes - 1).astype(int)
        centres = (np.arange(n_lanes) + 0.5) * lane_width
    out = []
    k = 1
    while k < n:
        if li[k] != li[k - 1]:
            nxt = li[k:k + n_hold]
            if nxt.size == n_hold and np.all(nxt == li[k]):        # committed
                y_from = centres[li[k - 1]]
                y_to = centres[li[k]]
                a = k - 1
                while a > 0 and abs(y[a] - y_from) < settle:
                    a -= 1
                a_full = k - 1                     # intended walk-back
                while a_full > 0 and abs(y[a_full] - y_from) >= settle:
                    a_full -= 1
                b = k
                while b < n - 1 and abs(y[b] - y_to) >= settle:
                    b += 1
                seg = slice(min(a, b), max(a, b) + 1)
                seg_full = slice(min(a_full, b), max(a_full, b) + 1)
                # Thiemann et al. (2008) time the manoeuvre by two-lane
                # occupancy: the body straddles the marking while it is
                # within half a vehicle width of it. Measured the same way
                # here so the two numbers mean the same thing.
                y_b = (marks[max(li[k - 1], li[k])] if marks is not None
                       else 0.5 * (y_from + y_to))
                straddle = np.abs(y[seg] - y_b) < 0.5 * W
                straddle_full = np.abs(y[seg_full] - y_b) < 0.5 * W
                out.append(dict(
                    veh=int(veh), t_start=float(t[a]), t_end=float(t[b]),
                    duration=float(straddle.sum()) * dt,
                    dur_c2c=float(t[b] - t[a]),
                    dy=float(abs(y[b] - y[a])),
                    peak_vy=float(np.abs(vy[seg]).max()),
                    from_lane=int(li[k - 1]), to_lane=int(li[k]),
                    yielding=bool(aware is not None
                                  and (aware[seg] == YIELDING).any()),
                    # full-manoeuvre window (see the docstring)
                    t_start_full=float(t[a_full]),
                    duration_full=float(straddle_full.sum()) * dt,
                    dur_c2c_full=float(t[b] - t[a_full]),
                    dy_full=float(abs(y[b] - y[a_full])),
                    peak_vy_full=float(np.abs(vy[seg_full]).max()),
                    truncated=bool(a_full == 0 or b == n - 1)))
                k = b + 1
                continue
        k += 1
    return out


def lane_change_events(h: History, persist: float = 0.5, settle: float = 0.6):
    """Lane changes of the background traffic, one dict per event.

    Thin loop over `lane_change_from_track` (which holds the definition); the EV
    itself is skipped.
    """
    road = h.road
    out = []
    for i in range(h.x.shape[1]):
        if i == h.ev:
            continue
        out.extend(lane_change_from_track(
            h.t, h.y[:, i], h.vy[:, i], float(h.W[i]),
            lane_width=road.lane_width, n_lanes=road.n_lanes,
            aware=h.aware[:, i], persist=persist, settle=settle, veh=i))
    return out


#: Gates applied to the headway/TTC pools, copied from `emv/highd.py:track_pools`
#: so the model and the dataset discard the same samples.
THW_MAX, TTC_MAX = 20.0, 60.0


def headway_pools(h: History, v_min: float = 0.1):
    """Pooled time headway and time-to-collision, in highD's own definitions.

    The definitions were pinned *by measurement* against recording 01 rather than
    read off the dataset description, because the description is ambiguous about
    which bumper `dhw` refers to:

      * `dhw` = |x_lead - x_ego| - L_preceding, i.e. the **bumper-to-bumper gap**
        from the ego's front to the leader's rear (median residual 0.01 m, which
        is the column's own rounding). This is exactly `gap = dx - half_L` as
        `forces.car_car` already computes it.
      * `thw = dhw / v_ego`                       (max residual 0.012 s)
      * `ttc = dhw / (v_ego - v_lead)` when closing, and 0 otherwise - highD
        leaves TTC unset for 63 % of frames for that reason (median residual
        0.115 s; the alternative `(dhw - L_lead)/dv` is off by 4.8 s and is not
        what the dataset stores).

    The leader is the nearest vehicle ahead in the **same lane**, which mirrors
    highD's use of its `precedingId` column (a lane-local relation), not the
    smooth lateral-overlap gate `car_car` uses for the IDM bound. Lane index
    comes from the same expression as `behaviour_pools`, so a vehicle midway
    through a manoeuvre switches leader at the marking - as it does in highD.

    Vectorised over frames: sorting by `lane * OFFSET + x` puts each lane's
    vehicles in consecutive, x-ordered slots, so the leader of every vehicle is
    its successor in that order whenever the lane label matches. No (N, N)
    pairwise array is built.
    """
    ev = h.ev
    x, y, vx, L = h.x, h.y, h.vx, h.L
    lane = np.clip(np.round(y / h.road.lane_width - 0.5), 0,
                   h.road.n_lanes - 1)
    span = float(np.ptp(x)) + 1.0
    order = np.argsort(lane * span + x, axis=1)
    take = lambda a: np.take_along_axis(np.broadcast_to(a, x.shape), order, axis=1)
    xs, lanes = np.take_along_axis(x, order, axis=1), np.take_along_axis(lane, order, axis=1)
    vs, Ls = np.take_along_axis(vx, order, axis=1), take(L)
    is_ev_s = take(h.ev == np.arange(x.shape[1]))

    same = lanes[:, 1:] == lanes[:, :-1]
    gap = (xs[:, 1:] - xs[:, :-1]) - 0.5 * (Ls[:, 1:] + Ls[:, :-1])
    v_ego, v_lead = vs[:, :-1], vs[:, 1:]
    # the EV may be somebody's leader, but is never an ego (highD has no EV)
    ok = same & ~is_ev_s[:, :-1] & (gap > 0.0) & (v_ego > v_min)

    thw = np.where(ok, gap / np.maximum(v_ego, v_min), np.nan)
    dv = v_ego - v_lead
    ttc = np.where(ok & (dv > 0.0), gap / np.maximum(dv, 1e-9), np.nan)
    thw = thw[np.isfinite(thw)]
    ttc = ttc[np.isfinite(ttc)]
    return (thw[(thw > 0) & (thw < THW_MAX)].astype(np.float32),
            ttc[(ttc > 0) & (ttc < TTC_MAX)].astype(np.float32))


def _chunk_peaks(a, n_chunk: int):
    """Per-column maxima over consecutive `n_chunk`-frame blocks.

    Per-vehicle peak statistics grow with how long a vehicle is observed, so a
    60 s simulated run is not comparable with highD's ~12 s median track unless
    the model side is cut into blocks of the same length (see
    `behaviour_stats(track_window=...)`).
    """
    if n_chunk < 1 or a.shape[0] <= n_chunk:
        return a.max(axis=0)
    n_full = a.shape[0] // n_chunk
    body = a[:n_full * n_chunk].reshape(n_full, n_chunk, a.shape[1]).max(axis=1)
    tail = a[n_full * n_chunk:]
    if tail.shape[0] > 1:
        body = np.vstack([body, tail.max(axis=0)[None, :]])
    return body.reshape(-1)


def behaviour_stats(h: History, track_window: float | None = None) -> dict:
    """Distributional observables of one run (background traffic + EV).

    Percentiles are pooled over vehicles and frames, which is how naturalistic
    datasets report them. Accelerations are frame-invariant, so they compare
    directly with dashcam ground truth; speeds are absolute here and relative in
    the ego-frame GT, so only the EV/background speeds are literature-comparable.

    `track_window` (seconds) cuts every vehicle's trace into blocks of that
    length before taking per-vehicle peaks, so that `peak_*` statistics can be
    compared with a trajectory dataset whose tracks are much shorter than a
    simulated run (highD's median track is ~12 s). `None` keeps the whole-run
    peaks, which is the behaviour of record for every result in PROJECT_LOG.
    """
    ev = h.ev
    others = np.ones(h.x.shape[1], bool)
    others[ev] = False
    dt = float(h.t[1] - h.t[0]) if h.n_frames > 1 else 0.1

    v_bg = h.vx[:, others]
    a_bg = h.ax[:, others]
    a_lat = np.abs(np.diff(h.vy[:, others], axis=0)) / dt
    veh_km = float(np.abs(np.diff(h.x[:, others], axis=0)).sum()) / 1000.0

    lcs = lane_change_events(h)
    dur = np.array([e["duration"] for e in lcs])
    pvy = np.array([e["peak_vy"] for e in lcs])
    # full-manoeuvre window, untruncated events only (a clipped manoeuvre has no
    # meaningful duration, and highD clips ~25 % of them at its 420 m boundary)
    full = [e for e in lcs if not e["truncated"]]
    durf = np.array([e["duration_full"] for e in full])
    c2cf = np.array([e["dur_c2c_full"] for e in full])
    dyf = np.array([e["dy_full"] for e in full])
    pvyf = np.array([e["peak_vy_full"] for e in full])

    def pct(a, q):
        return float(np.percentile(a, q)) if np.size(a) else float("nan")

    # Per-vehicle peaks. Pooled percentiles are dominated by cruising (95 % of
    # all (vehicle, frame) samples are near-zero acceleration), whereas comfort
    # thresholds and naturalistic braking-event statistics describe the peak of
    # a manoeuvre - so both are reported, and the yielding subset separately.
    peak_dec_v = -a_bg.min(axis=0)                       # one value per vehicle
    peak_lat_v = a_lat.max(axis=0) if a_lat.size else np.zeros(int(others.sum()))
    n_chunk = (0 if track_window is None
               else max(1, int(round(track_window / max(dt, 1e-6)))))
    if n_chunk:
        peak_dec = _chunk_peaks(-a_bg, n_chunk)
        peak_lat = (_chunk_peaks(a_lat, n_chunk) if a_lat.size else peak_lat_v)
    else:
        peak_dec, peak_lat = peak_dec_v, peak_lat_v
    yielded = (h.aware == YIELDING).any(axis=0)[others]

    # Lane keeping: distance to the current lane centre while the body is NOT
    # straddling a marking, i.e. excluding the lane changes themselves. Same
    # rule as emv/highd.py:track_pools so the two numbers are comparable.
    y_bg = h.y[:, others]
    W_bg = h.W[others]
    marks = np.arange(h.road.n_lanes + 1) * h.road.lane_width
    lane_bg = np.clip(np.round(y_bg / h.road.lane_width - 0.5), 0,
                      h.road.n_lanes - 1)
    d_mark = np.abs(y_bg[:, :, None] - marks[None, None, :]).min(axis=2)
    keeping = d_mark >= 0.5 * W_bg[None, :]
    off = np.abs(y_bg - (lane_bg + 0.5) * h.road.lane_width)
    off_per_veh = np.array([off[keeping[:, j], j].mean()
                            for j in range(off.shape[1])
                            if keeping[:, j].any()])

    return dict(
        ev_med_speed=float(np.median(h.vx[:, ev])),
        # v0[ev] is 0 only for the EV-inert control (scenarios.make_highd_like)
        ev_speed_ratio=float(np.median(h.vx[:, ev]) / h.v0[ev])
        if h.v0[ev] > 0 else float("nan"),
        bg_med_speed=float(np.median(v_bg)),
        bg_p15_speed=pct(v_bg, 15),
        accel_p99=pct(a_bg, 99),
        decel_p01=-pct(a_bg, 1),                 # strongest 1 % decel, positive
        peak_decel_med=pct(peak_dec, 50), peak_decel_p90=pct(peak_dec, 90),
        peak_alat_med=pct(peak_lat, 50), peak_alat_p90=pct(peak_lat, 90),
        peak_decel_p99=pct(peak_dec, 99), peak_alat_p99=pct(peak_lat, 99),
        # yield subsets index vehicles, so they always use the unchunked peaks
        yield_peak_decel_med=pct(peak_dec_v[yielded], 50),
        yield_peak_alat_med=pct(peak_lat_v[yielded], 50),
        alat_p99=pct(a_lat, 99),
        lat_speed_med=pct(np.abs(h.vy[:, others]), 50),
        lat_speed_p90=pct(np.abs(h.vy[:, others]), 90),
        n_lane_changes=len(lcs),
        lc_per_veh_km=len(lcs) / veh_km if veh_km > 0 else float("nan"),
        lc_dur_med=pct(dur, 50), lc_dur_p10=pct(dur, 10), lc_dur_p90=pct(dur, 90),
        lc_peak_vy_med=pct(pvy, 50),
        lc_yield_share=float(np.mean([e["yielding"] for e in lcs])) if lcs else 0.0,
        n_yielded=int(yielded.sum()),
        veh_km=veh_km,
        # --- added 2026-07-28 for the highD comparison (emv/highd.py emits the
        # same keys under the same definitions); all keys above are unchanged.
        lc_dur_mean=float(dur.mean()) if dur.size else float("nan"),
        lc_dur_sd=float(dur.std()) if dur.size else float("nan"),
        lc_peak_vy_p90=pct(pvy, 90),
        # full-manoeuvre window: the variant comparable with dataset/literature
        # lane-change durations (see lane_change_from_track's docstring)
        lc_dur_full_med=pct(durf, 50), lc_dur_full_p10=pct(durf, 10),
        lc_dur_full_p90=pct(durf, 90),
        lc_dur_full_mean=float(durf.mean()) if durf.size else float("nan"),
        lc_dur_full_sd=float(durf.std()) if durf.size else float("nan"),
        lc_c2c_med=pct(c2cf, 50), lc_c2c_p90=pct(c2cf, 90),
        lc_dy_full_med=pct(dyf, 50),
        lc_peak_vy_full_med=pct(pvyf, 50),
        lc_peak_vy_full_p90=pct(pvyf, 90),
        lc_truncated_share=(float(np.mean([e["truncated"] for e in lcs]))
                            if lcs else 0.0),
        lane_offset_mean=(float(off_per_veh.mean()) if off_per_veh.size
                          else float("nan")),
        lane_offset_p90=pct(off_per_veh, 90),
        obs_dur=float(h.t[-1] - h.t[0]) if h.n_frames > 1 else 0.0,
        n_tracks=int(others.sum()),
        # --- added 2026-08-06: longitudinal spacing, in highD's own definitions
        # (see headway_pools). These targets were measured in
        # out/highd_targets.json from the start but had no model counterpart, so
        # the car-following block could not be scored against real data.
        **_headway_stats(*headway_pools(h)),
    )


def _headway_stats(thw, ttc) -> dict:
    def pct(a, q):
        return float(np.percentile(a, q)) if np.size(a) else float("nan")
    return dict(thw_med=pct(thw, 50), thw_p15=pct(thw, 15), thw_p85=pct(thw, 85),
                ttc_p01=pct(ttc, 1), ttc_p05=pct(ttc, 5),
                n_thw=int(np.size(thw)), n_ttc=int(np.size(ttc)))


def behaviour_pools(h: History, track_window: float | None = None) -> dict:
    """Raw value pools behind `behaviour_stats`, for pooling across runs.

    A single run yields only a handful of lane changes, so a per-run median is
    dominated by sampling noise; pooling the events of several seeds and taking
    one percentile is the same thing the highD side does across carriageways
    (`emv/highd.py:track_pools`). Keys mirror that function so the two can be
    compared directly.
    """
    ev = h.ev
    others = np.ones(h.x.shape[1], bool)
    others[ev] = False
    dt = float(h.t[1] - h.t[0]) if h.n_frames > 1 else 0.1
    a_bg = h.ax[:, others]
    a_lat = np.abs(np.diff(h.vy[:, others], axis=0)) / dt
    n_chunk = (0 if track_window is None
               else max(1, int(round(track_window / max(dt, 1e-6)))))
    if n_chunk:
        peak_dec = _chunk_peaks(-a_bg, n_chunk)
        peak_lat = (_chunk_peaks(a_lat, n_chunk) if a_lat.size
                    else np.zeros(int(others.sum())))
    else:
        peak_dec = -a_bg.min(axis=0)
        peak_lat = a_lat.max(axis=0) if a_lat.size else np.zeros(int(others.sum()))

    y_bg = h.y[:, others]
    W_bg = h.W[others]
    marks = np.arange(h.road.n_lanes + 1) * h.road.lane_width
    lane_bg = np.clip(np.round(y_bg / h.road.lane_width - 0.5), 0,
                      h.road.n_lanes - 1)
    d_mark = np.abs(y_bg[:, :, None] - marks[None, None, :]).min(axis=2)
    keeping = d_mark >= 0.5 * W_bg[None, :]
    off = np.abs(y_bg - (lane_bg + 0.5) * h.road.lane_width)
    off_per_veh = np.array([off[keeping[:, j], j].mean()
                            for j in range(off.shape[1]) if keeping[:, j].any()])

    lcs = lane_change_events(h)
    full = [e for e in lcs if not e["truncated"]]
    thw, ttc = headway_pools(h)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return dict(
        thw=thw, ttc=ttc,
        trk_peak_decel=f32(peak_dec), trk_peak_alat=f32(peak_lat),
        trk_mean_vx=f32(h.vx[:, others].mean(axis=0)),
        trk_offset_abs=f32(off_per_veh),
        vy=f32(np.abs(h.vy[:, others]).ravel()),
        alat=f32(a_lat.ravel()),
        accel=f32(a_bg.ravel()),
        lc_dur=f32([e["duration"] for e in lcs]),
        lc_peak_vy=f32([e["peak_vy"] for e in lcs]),
        lc_dur_full=f32([e["duration_full"] for e in full]),
        lc_c2c=f32([e["dur_c2c_full"] for e in full]),
        lc_peak_vy_full=f32([e["peak_vy_full"] for e in full]),
        n_lane_changes=float(len(lcs)),
        veh_km=float(np.abs(np.diff(h.x[:, others], axis=0)).sum()) / 1000.0)


def stats_from_pools(p: dict) -> dict:
    """`behaviour_stats`-compatible observables from pooled `behaviour_pools`."""
    def pct(a, q):
        a = np.asarray(a, float)
        return float(np.percentile(a, q)) if a.size else float("nan")
    return dict(
        bg_med_speed=pct(p["trk_mean_vx"], 50),
        bg_p15_speed=pct(p["trk_mean_vx"], 15),
        accel_p99=pct(p["accel"], 99), decel_p01=-pct(p["accel"], 1),
        peak_decel_med=pct(p["trk_peak_decel"], 50),
        peak_decel_p90=pct(p["trk_peak_decel"], 90),
        peak_decel_p99=pct(p["trk_peak_decel"], 99),
        peak_alat_med=pct(p["trk_peak_alat"], 50),
        peak_alat_p90=pct(p["trk_peak_alat"], 90),
        peak_alat_p99=pct(p["trk_peak_alat"], 99),
        alat_p99=pct(p["alat"], 99),
        lat_speed_med=pct(p["vy"], 50), lat_speed_p90=pct(p["vy"], 90),
        lane_offset_mean=(float(np.mean(p["trk_offset_abs"]))
                          if np.size(p["trk_offset_abs"]) else float("nan")),
        lane_offset_p90=pct(p["trk_offset_abs"], 90),
        lc_dur_med=pct(p["lc_dur"], 50), lc_peak_vy_med=pct(p["lc_peak_vy"], 50),
        lc_dur_full_med=pct(p["lc_dur_full"], 50),
        lc_dur_full_p10=pct(p["lc_dur_full"], 10),
        lc_dur_full_p90=pct(p["lc_dur_full"], 90),
        lc_c2c_med=pct(p["lc_c2c"], 50),
        lc_peak_vy_full_med=pct(p["lc_peak_vy_full"], 50),
        lc_peak_vy_full_p90=pct(p["lc_peak_vy_full"], 90),
        n_lane_changes=int(p["n_lane_changes"]),
        n_lane_changes_full=int(np.size(p["lc_dur_full"])),
        lc_per_veh_km=(p["n_lane_changes"] / p["veh_km"]
                       if p["veh_km"] > 0 else float("nan")),
        veh_km=float(p["veh_km"]),
        # highD reports the *complete*-manoeuvre rate separately (0.2758 vs 0.3241
        # for all recorded events); the model's own untruncated count is the
        # like-for-like counterpart, so both are emitted under the dataset's names
        lc_per_veh_km_complete=(np.size(p["lc_dur_full"]) / p["veh_km"]
                                if p["veh_km"] > 0 else float("nan")),
        **_headway_stats(p.get("thw", np.zeros(0)), p.get("ttc", np.zeros(0))))


def summary_line(m: dict) -> str:
    return (f"{m['name']:<28s} vEV {m['ev_mean_speed']:5.1f} m/s ({m['ev_speed_ratio']*100:4.0f}%)  "
            f"clear@50m {m['t_clear']:6.1f}s  TTCmin {m['min_ttc']:5.2f}s  "
            f"coll {m['collisions']:2d}  p95dec {m['p95_decel']:4.2f}  "
            f"p95alat {m['p95_alat']:4.2f}  osc {m['oscillation']:4.2f}  "
            f"react {m['react_dist_mean']:6.1f}m  disrupt {m['disruption']*100:4.1f}%")
