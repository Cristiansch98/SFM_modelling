"""highD observable targets and the calibration objective.

Mirrors the two patterns already in the repo: a `(key, scale, weight)` spec with
a weighted normalised distance (as `emv/groundtruth.py:_DIST_SPEC` does for the
dashcam fingerprint) and a `loss(theta, seeds) -> (total, parts)` signature (as
`emv/calibrate.py:loss` does), so `emv.calibrate`'s scipy-free
`latin_hypercube` / `nelder_mead` drive this objective unchanged.

**Scope.** highD has no emergency vehicle, so this objective can only speak about
host-traffic behaviour: how a lateral manoeuvre is executed and how precisely
drivers hold a lane. The EV-response block (`A_ev, B_ev, A_c, B_c, T_react,
gamma_c, R_front, p_noncomply, delay_*, urgency_pin_relief`) is not identifiable
here and keeps its dashcam-GT / literature calibration - which is why the
calibration is two-stage and why `emv/params.py:TUNED` is left intact.

**One target is deliberately not optimised.** `lc_per_veh_km` is measured at
0.31 on highD and is almost entirely *discretionary* lane changing; this model
has no discretionary lane-change mechanism (the EV-inert control produces
exactly 0.00), so the rate is structurally unmatchable. It is carried at weight
0 - reported, never fitted around. Every claim about manoeuvre kinematics is
therefore explicitly *conditional on a manoeuvre occurring*.

**Which duration is used.** `lc_dur_full_*` and `lc_c2c_*`, i.e. the
full-manoeuvre window - see `emv/metrics.py:lane_change_from_track`, which
documents why the originally shipped `duration` is a half-manoeuvre quantity and
keeps it for reproducibility.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import metrics
from .params import Params
from .scenarios import highd_regime, make_highd_like

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGETS_PATH = os.path.join(ROOT, "out", "highd_targets.json")

#: (key, scale, weight). `scale` normalises |model - highD| to O(1) and is set
#: to roughly the between-carriageway spread of that observable, so one unit of
#: deviation means "one recording's worth of natural variation".
OBS_SPEC: tuple = (
    # --- manoeuvre kinematics (conditional on a lane change happening) -------
    ("lc_dur_full_med",     0.35, 2.0),
    ("lc_dur_full_p90",     0.70, 0.5),
    ("lc_c2c_med",          0.45, 1.0),
    ("lc_peak_vy_full_med", 0.12, 2.0),
    ("lc_peak_vy_full_p90", 0.18, 0.5),
    # --- lateral comfort envelope -------------------------------------------
    ("peak_alat_p90",       0.10, 1.5),
    ("peak_alat_p99",       0.20, 1.0),
    # --- lane keeping --------------------------------------------------------
    ("lane_offset_mean",    0.05, 1.5),
    ("lane_offset_p90",     0.10, 1.0),
    # lateral-speed floor: highD's value is partly tracking noise (yVelocity is
    # quantised at 0.01 m/s), so it is scored at low weight
    ("lat_speed_p90",       0.08, 0.25),
    # --- reported at weight 0: real gaps, but not controlled by the parameters
    # this stage fits. Deceleration is driven by the EV-response block
    # (gamma_c, kappa_merge - stage 2) and the background speed by the measured
    # scenario itself, so scoring them here would only distort the lateral fit.
    ("peak_decel_p90",      0.25, 0.0),
    ("peak_decel_p99",      0.60, 0.0),
    ("bg_med_speed",        1.50, 0.0),
    # structurally unmatchable: no discretionary lane-change mechanism
    ("lc_per_veh_km",       0.10, 0.0),
)

# ===========================================================================
# The EV-FREE ("normal traffic") objective, added 2026-08-06.
#
# OBS_SPEC above is the objective of record for the 2026-07-29 fit and is left
# alone. It has two properties this one deliberately does not share:
#
#   1. It is scored with a *live emergency vehicle* in the scenario. Since the
#      model had no discretionary lane-change mechanism, every lane change it
#      measured was EV-induced - and those were compared against highD's, which
#      are almost all discretionary. Measured: with the EV made inert the model
#      produced 0 lane changes over 206 veh-km. So the "host-traffic" fit was
#      only identifiable because an EV was present, which is circular.
#   2. It scores nothing longitudinal. highD's headway and TTC distributions were
#      extracted from the start and carried at weight 0 or not at all, because
#      the model side had no counterpart (now `metrics.headway_pools`).
#
# The normal objective fixes both: `ev_inert=True`, and the car-following block
# is scored against real spacing. Its lane-change rate is a *fitted* target
# rather than a structurally unmatchable one.
# ===========================================================================
OBS_SPEC_NORMAL: tuple = (
    # --- manoeuvre kinematics (conditional on a lane change happening) -------
    ("lc_dur_full_med",     0.35, 2.0),
    ("lc_dur_full_p90",     0.70, 0.5),
    ("lc_c2c_med",          0.45, 1.0),
    ("lc_peak_vy_full_med", 0.12, 2.0),
    ("lc_peak_vy_full_p90", 0.18, 0.5),
    # --- how OFTEN a manoeuvre happens. Weight 1.5 rather than 0: with the
    # incentive term this is now a property of the model, and it is the only
    # observable that constrains A_pass / A_keep_right at all.
    ("lc_per_veh_km_complete", 0.106, 1.5),
    # --- lateral comfort envelope -------------------------------------------
    ("peak_alat_p90",       0.10, 1.5),
    ("peak_alat_p99",       0.20, 1.0),
    # --- lane keeping --------------------------------------------------------
    ("lane_offset_mean",    0.05, 1.5),
    ("lane_offset_p90",     0.10, 1.0),
    ("lat_speed_p90",       0.08, 0.25),
    # --- longitudinal spacing: the car-following block, scored for the first
    # time. New scales are set at **2.5x the measured between-carriageway s.d.**
    # (out/highd_targets.json), because that is the convention OBS_SPEC's scales
    # already follow - their scale/sd ratios run 1.7-5.4 with a median of 2.44,
    # not the 1.0 its docstring suggests. Using the raw s.d. here would have made
    # every longitudinal residual count ~2.5x a lateral one of the same
    # statistical size, silently reweighting the objective. The two keys that
    # already existed keep their published scales exactly.
    ("thw_med",             0.238, 1.5),
    ("thw_p15",             0.107, 1.0),
    ("thw_p85",             0.453, 1.0),
    ("ttc_p01",             0.990, 0.75),
    ("ttc_p05",             1.306, 0.75),
    ("peak_decel_p90",      0.250, 1.0),     # scale as in OBS_SPEC
    ("peak_decel_p99",      0.600, 0.5),     # scale as in OBS_SPEC
    ("accel_p99",           0.173, 0.5),
    ("decel_p01",           0.378, 0.5),
    # --- reported at weight 0 -------------------------------------------------
    # background speed is set by the measured lane-speed profile, not fitted
    ("bg_med_speed",        1.50, 0.0),
    ("bg_p15_speed",        1.605, 0.0),
    # the all-events rate, kept for comparison with the complete-manoeuvre one
    ("lc_per_veh_km",       0.10, 0.0),
)

#: Longitudinal (car-following) block. `tau` and the IDM headway/gap set the
#: spacing distribution; `A_v`/`Bx_v` set how early a driver starts reacting to
#: the vehicle in front, which shows up in the TTC tail rather than the median.
SPEC_LON: list = [
    ("tau",      0.30, 1.60),
    ("T_hw_lo",  0.40, 1.40),
    ("T_hw_hi",  1.40, 3.20),
    ("s0",       0.80, 5.00),
    ("A_v",      0.50, 6.00),
    ("Bx_v",     1.50, 12.0),
    ("a_max",    1.00, 4.00),
    ("b_comf",   1.50, 5.00),
]

#: Lateral block: the six of the 2026-07-29 fit plus the density exponent.
SPEC_LAT: list = [
    ("a_pin",      0.15, 2.50),
    ("zeta_lat",   0.40, 3.00),
    ("v_lat_max",  0.50, 2.20),
    ("a_lat_max",  0.20, 3.00),
    ("sigma_off",  0.00, 0.60),
    ("het_lat",    0.00, 0.60),
    ("k_rho",      0.00, 2.00),
]

#: Discretionary lane changing. Bounds start above a_pin's upper bound because a
#: manoeuvre only happens when the incentive exceeds the pinning threshold, so
#: values below it are all equivalent to "off".
SPEC_LC: list = [
    ("A_pass",        0.50, 12.0),
    ("A_keep_right",  0.00, 12.0),
    ("T_frust",       0.80,  6.0),
    ("s_veto",        5.00, 60.0),
]

#: The full normal-traffic parameter vector.
SPEC_NORMAL: list = SPEC_LON + SPEC_LAT + SPEC_LC

#: `latlc` fits the lateral block and the discretionary block **together**,
#: and it is the block to use rather than `lat` then `lc`. They are coupled
#: through the depinning threshold - a lane change happens iff
#: `A_pass * frust > a_pin_eff` - so fitting them separately lets one undo the
#: other. Measured: fitting `lat` alone with the incentive held fixed raised
#: `a_pin` to 1.53 and thereby suppressed the manoeuvre rate to 0.008 against a
#: measured 0.276, while scoring *better* on the kinematics because the few
#: surviving manoeuvres were the well-shaped ones (that loophole is now closed by
#: LC_MIN_EVENTS, but the coupling remains and is best handled directly).
SPEC_LATLC: list = SPEC_LAT + SPEC_LC
NORMAL_BLOCKS: dict = dict(lon=SPEC_LON, lat=SPEC_LAT, lc=SPEC_LC,
                           latlc=SPEC_LATLC, joint=SPEC_NORMAL)

#: Regimes the normal objective is fitted on. Two are needed because `k_rho` is
#: a *density* exponent and cannot be identified at a single density; congested
#: (2 carriageways) is deliberately excluded and held out, being both the weakest
#: evidence in the dataset and the block the previous free-flow-only fit damaged.
FIT_REGIMES: tuple = ("freeflow", "dense")

#: Parameters the highD objective can identify, with search bounds. `a_pin` is
#: the real lateral knob: the washboard amplitude *is* a_pin
#: (`forces.lane_keep`), so the model's peak lateral acceleration tracks it
#: almost 1:1, whereas `a_lat_max` at its default 3.0 never binds at all.
HIGHD_SPEC: list = [
    ("a_pin",      0.15, 2.5),
    ("zeta_lat",   0.40, 3.0),
    ("v_lat_max",  0.50, 2.2),
    ("a_lat_max",  0.20, 3.0),
    ("sigma_off",  0.00, 0.60),
    # Driver-to-driver dispersion of the lateral block. Added after the
    # moments-only fit was found to match every median while producing
    # distributions up to 4x too narrow (see docs/HIGHD_VALIDATION.md sec. 7).
    ("het_lat",    0.00, 0.60),
]
HIGHD_NAMES = [s[0] for s in HIGHD_SPEC]

#: Weight of the distribution-shape term relative to the moment terms. Matching
#: percentiles does not imply matching a distribution, so the objective scores
#: both; 1.0 puts them on an equal footing since both are O(1) normalised.
W_SHAPE = 1.0


def load_targets(path: str | None = None) -> dict:
    with open(path or TARGETS_PATH) as fh:
        return json.load(fh)


def target_block(name: str = "train:dense", path: str | None = None) -> dict:
    """One pooled target block, e.g. 'train:dense' or 'holdout:freeflow'."""
    js = load_targets(path)
    t = js["targets"]
    if name not in t:
        raise KeyError(f"no target block {name!r}; have {sorted(t)}")
    return t[name]


def targets_of(block: dict) -> dict:
    """(key -> measured value) for the keys this objective scores."""
    return {k: block[k]["value"] for k, _, _ in OBS_SPEC if k in block}


def spread_of(block: dict) -> dict:
    """(key -> between-carriageway s.d.), the natural-variation yardstick."""
    return {k: block[k]["sd"] for k, _, _ in OBS_SPEC if k in block}


def make_params(theta, names=None, base: Params | None = None) -> Params:
    names = names or HIGHD_NAMES
    lo = np.array([s[1] for s in HIGHD_SPEC])
    hi = np.array([s[2] for s in HIGHD_SPEC])
    th = np.clip(np.asarray(theta, float), lo, hi) if len(names) == len(lo) \
        else np.asarray(theta, float)
    p = base if base is not None else Params()
    return p.copy(**dict(zip(names, th.tolist())))


def model_observables(p: Params, seeds=(11,), regime: str = "dense",
                      T: float = 110.0, road_len: float = 3000.0,
                      track_window: float | None = None,
                      spec: dict | None = None,
                      return_pools: bool = False,
                      ev_inert: bool = False):
    """Observables of the highD-matched scenario, averaged over seeds.

    `track_window` equalises the observation length: per-vehicle peak statistics
    grow with how long a vehicle is watched, and a highD track lasts ~13 s
    against a full simulated run.

    `ev_inert=True` removes the emergency vehicle from the scenario entirely
    (parked 5 km beyond the road end), which is what the normal-traffic
    calibration requires: highD contains no emergency vehicle, so scoring a
    scenario that does have one against it confounds EV-induced behaviour with
    ordinary behaviour. It also disables the early stop, since there is no EV
    position to stop on. Default False = the objective of record.

    The road is 3 km rather than the 1.5 km of `make_overtake`: at highD's real
    free-flow density (~10 veh/km/lane, less than half what `make_overtake`
    assumes) a shorter road holds too few vehicles for the EV to induce enough
    lane changes to estimate a median duration - 3 seeds x 1.5 km gave 9 events,
    3 km gives ~60.
    """
    sp = spec or highd_regime(regime)
    pools, coll, ev_ratio = [], 0, []
    for s in seeds:
        sim = make_highd_like(seed=s, p=p, regime=regime, spec=sp,
                              road_len=road_len, ev_inert=ev_inert)
        h = sim.run(T, rec_dt=0.12,
                    stop_when_ev_x=None if ev_inert else road_len - 50.0)
        pools.append(metrics.behaviour_pools(h, track_window=track_window))
        with np.errstate(invalid="ignore", divide="ignore"):
            ev = metrics.evaluate(h)
        coll += int(ev["collisions"])
        # the inert EV has v0 = 0, so its speed ratio is undefined by construction
        if np.isfinite(ev["ev_speed_ratio"]):
            ev_ratio.append(float(ev["ev_speed_ratio"]))
    merged = {}
    for k in pools[0]:
        vals = [q[k] for q in pools]
        merged[k] = (float(np.sum(vals)) if np.isscalar(vals[0])
                     else np.concatenate(vals))
    out = metrics.stats_from_pools(merged)
    out["collisions"] = coll
    out["ev_speed_ratio"] = (float(np.mean(ev_ratio)) if ev_ratio
                             else float("nan"))
    return (out, merged) if return_pools else out


_DATA_POOLS = {}


def data_pools(path: str | None = None):
    """Pooled highD samples (out/highd_pools.npz), cached."""
    p = path or os.path.join(ROOT, "out", "highd_pools.npz")
    if p not in _DATA_POOLS:
        _DATA_POOLS[p] = np.load(p)
    return _DATA_POOLS[p]


#: Observables whose *shape* is scored, mapping the pooled-sample key in
#: out/highd_pools.npz to the matching key in metrics.behaviour_pools.
SHAPE_KEYS = ("lc_dur_full", "lc_peak_vy_full", "trk_offset_abs",
              "trk_peak_alat")

#: Shape keys of the normal objective: the four above plus the two longitudinal
#: distributions, whose *shape* is the whole point - a car-following model can
#: match a median headway with a distribution of quite the wrong width, and
#: `out/highd_pools.npz` already stores 200k real samples of each.
SHAPE_KEYS_NORMAL = SHAPE_KEYS + ("thw", "ttc", "accel")


def wasserstein1(a, b, n_grid: int = 200) -> float:
    """1-Wasserstein distance between two samples, scipy-free.

    Both empirical quantile functions are evaluated on a common grid and the
    mean absolute difference taken, so the result is in the observable's own
    units and is monotone in the physical mismatch. Preferred over a
    Kolmogorov-Smirnov statistic, which is a supremum over probabilities and
    saturates once two distributions barely overlap.

    Percentile targets can be met by distributions of quite different shape, so
    this is the metric that sees a model matching the median while being too
    narrow - which is exactly what the calibrated model does.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size < 2 or b.size < 2:
        return float("nan")
    q = (np.arange(n_grid) + 0.5) / n_grid
    qa = a[np.clip((q * a.size).astype(int), 0, a.size - 1)]
    qb = b[np.clip((q * b.size).astype(int), 0, b.size - 1)]
    return float(np.mean(np.abs(qa - qb)))


def dispersion_ratio(a, b) -> float:
    """Model interquartile range divided by the data's. 1.0 = matched spread;
    below 1 = the model is too narrow (too little driver-to-driver variety)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 4 or b.size < 4:
        return float("nan")
    ia = np.percentile(a, 75) - np.percentile(a, 25)
    ib = np.percentile(b, 75) - np.percentile(b, 25)
    return float(ia / ib) if ib > 0 else float("nan")


#: Price of a distribution the model does not produce at all, for the EV-free
#: objective. Same failure mode as `D_CAP` and just as easy to miss: with no
#: discretionary lane changes the `lc_dur_full` and `lc_peak_vy_full` pools are
#: EMPTY, `wasserstein1` returns NaN, and the un-priced version silently averaged
#: over the *remaining* keys - so a model that produced no manoeuvres scored a
#: better shape distance than one that produced imperfect manoeuvres. Measured:
#: it was enough to put the no-lane-change baseline (4.23) ahead of the one with
#: the term switched on (4.72), i.e. the objective preferred the defect.
#: 2.0 matches the value `loss()` already uses when the whole shape term is
#: unavailable. None = the published behaviour (skip), left as the default so the
#: 2026-07-29 record is untouched.
SHAPE_MISS = 2.0


def shape_distance(model_pools: dict, data_pools, keys=SHAPE_KEYS,
                   miss: float | None = None) -> dict:
    """1-Wasserstein and dispersion ratio per observable, plus a scale-free
    total (W1 divided by the data's own interquartile range).

    `miss` prices a distribution the model cannot produce; see SHAPE_MISS.
    """
    out, rel = {}, []
    for k in keys:
        d = data_pools[k] if k in getattr(data_pools, "files", data_pools) else None
        if d is None:
            continue
        w = wasserstein1(model_pools.get(k, np.zeros(0)), d)
        r = dispersion_ratio(model_pools.get(k, np.zeros(0)), d)
        d = np.asarray(d, float)
        iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
        out[k] = dict(w1=round(w, 4), iqr_ratio=round(r, 3),
                      w1_over_iqr=round(w / iqr, 3) if iqr > 0 else None)
        if iqr > 0 and np.isfinite(w):
            rel.append(w / iqr)
        elif miss is not None:
            out[k]["missing"] = True
            rel.append(float(miss))
    out["total"] = round(float(np.mean(rel)), 4) if rel else float("nan")
    return out


#: Per-key deviation ceiling for the normal objective, and the price of an
#: observable the model cannot produce at all.
#:
#: The published objective uses `nan=2.0, cap=None`, which is fine when every
#: observable exists. It is NOT fine for the EV-free objective: with no
#: discretionary lane changes the five `lc_*` kinematic keys are undefined and
#: cost 2 units each, whereas a real but badly-shaped manoeuvre costs up to 50
#: (a 37 s manoeuvre against a 2.28 s target on a 0.35 s scale is 99 units).
#: The optimiser is then **rewarded for switching the new term off**, which would
#: have quietly reproduced the very gap this extension exists to close.
#:
#: Setting the ceiling and the NaN price to the same value makes "the model
#: cannot produce this observable" exactly as bad as the worst measurable value
#: and never better, and bounds each key's contribution so one catastrophic
#: observable cannot dominate the other nineteen.
D_CAP = 8.0


def distance(obs: dict, tgt: dict, spec=OBS_SPEC, nan: float = 2.0,
             cap: float | None = None) -> dict:
    """Weighted normalised deviation, same construction as
    `groundtruth.fingerprint_distance`: a NaN in the model is penalised at `nan`
    scale units, a missing target skips the key. Zero-weight keys are reported
    but excluded from the total. `cap` bounds each key's deviation."""
    out, num, den = {}, 0.0, 0.0
    for key, scale, weight in spec:
        if key not in tgt or not np.isfinite(tgt[key]):
            continue
        v = obs.get(key, float("nan"))
        d = nan if not np.isfinite(v) else abs(v - tgt[key]) / scale
        if cap is not None:
            d = min(d, cap)
        out[key] = round(float(d), 4)
        if weight > 0:
            num += weight * d
            den += weight
    out["total"] = round(num / den, 4) if den else float("nan")
    return out


def loss(theta, seeds=(11, 12, 13), names=None, tgt: dict | None = None,
         regime: str = "dense", base: Params | None = None,
         track_window: float | None = None,
         spec_regime: dict | None = None,
         w_shape: float = W_SHAPE) -> tuple[float, dict]:
    """Total loss and its parts.

    Two terms. The *moment* term is the weighted normalised deviation of the
    percentile observables (`OBS_SPEC`). The *shape* term is the mean
    1-Wasserstein distance between the model's and the data's pooled samples,
    normalised by the data's interquartile range, because matching a set of
    percentiles does not imply matching a distribution: the moments-only fit
    reproduced every median while leaving the peak-lateral-speed distribution
    four times too narrow. Collisions are penalised exactly as `calibrate.loss`
    does (5 each, capped at 50) so the objectives stay comparable.
    """
    tgt = tgt if tgt is not None else targets_of(target_block(f"train:{regime}"))
    p = make_params(theta, names, base)
    obs, mp = model_observables(p, seeds=seeds, regime=regime,
                                track_window=track_window, spec=spec_regime,
                                return_pools=True)
    d = distance(obs, tgt)
    fit = d["total"]
    sh = shape_distance(mp, data_pools())
    shp = sh["total"] if np.isfinite(sh["total"]) else 2.0
    pen = min(5.0 * obs.get("collisions", 0), 50.0)
    parts = {k: v for k, v in d.items() if k != "total"}
    parts["_moment"] = fit
    parts["_shape"] = round(float(shp), 4)
    parts["_collisions"] = pen
    return float(fit + w_shape * shp + pen), parts


# ---------------------------------------------------------------------------
# the EV-free, multi-regime objective (stage 1)
# ---------------------------------------------------------------------------
def normal_context(regimes=FIT_REGIMES, targets_path: str | None = None) -> dict:
    """Everything the normal objective needs, loaded once.

    Kept out of `loss_normal` so a fit does not re-read and re-parse the target
    file on every one of its ~200 evaluations.
    """
    ctx = {}
    for r in regimes:
        block = target_block(f"train:{r}", targets_path)
        ctx[r] = dict(tgt=targets_of_spec(block, OBS_SPEC_NORMAL),
                      sd=spread_of_spec(block, OBS_SPEC_NORMAL),
                      spec=highd_regime(r),
                      window=block.get("obs_dur_med", {}).get("value"),
                      n_cw=block.get("_n_carriageways"))
    return ctx


def targets_of_spec(block: dict, spec=OBS_SPEC_NORMAL) -> dict:
    return {k: block[k]["value"] for k, _, _ in spec if k in block}


def spread_of_spec(block: dict, spec=OBS_SPEC_NORMAL) -> dict:
    return {k: block[k]["sd"] for k, _, _ in spec if k in block}


#: Minimum number of complete manoeuvres before the lane-change *kinematics* may
#: be scored at all. The third instance of one failure mode, and the one that
#: would have invalidated the headline: the `lat` fit of 2026-08-06 produced
#: **5 complete manoeuvres over 612 veh-km** (a rate of 0.008 against a measured
#: 0.276) and was nevertheless credited with a lane-change duration only 1.25 s.d.
#: from the target, because `stats_from_pools` takes a median of whatever exists.
#: Missing the rate cost 2.5 scale units; matching a median estimated from 5
#: events earned far more than that, so the optimiser suppressed manoeuvres
#: almost entirely and kept the few well-shaped ones.
#:
#: A median from n events has standard error ~1.25 sigma/sqrt(n); with the
#: measured sigma of 0.78 s, n = 20 still gives ~0.22 s, i.e. ~2.3x the target's
#: own between-carriageway spread. So 20 is a floor for "not meaningless", not a
#: claim of precision. It is trivially met by any model with a roughly correct
#: rate (highD's rate over the ~600 veh-km of a 3-seed run implies ~170 events)
#: and unreachable by a degenerate one, which is exactly the discrimination
#: wanted.
LC_MIN_EVENTS = 20

#: The keys that require that evidence. The *rate* is deliberately not among them:
#: it must stay measurable so that producing no manoeuvres is scored as the large
#: deviation it is, rather than blanked.
LC_KINEMATIC_KEYS = ("lc_dur_full_med", "lc_dur_full_p90", "lc_c2c_med",
                     "lc_peak_vy_full_med", "lc_peak_vy_full_p90")


def normal_observables(p: Params, ctx: dict, regime: str, seeds,
                       road_len: float = 3000.0, T: float = 110.0,
                       return_pools: bool = False,
                       min_events: int = LC_MIN_EVENTS):
    """Model observables for one regime with **no emergency vehicle present**.

    Blanks the manoeuvre-kinematics keys (and their sample pools, so the shape
    term prices them too) when fewer than `min_events` complete manoeuvres were
    observed - see LC_MIN_EVENTS for why this is not optional.
    """
    c = ctx[regime]
    out = model_observables(p, seeds=seeds, regime=regime, T=T,
                            road_len=road_len, track_window=c["window"],
                            spec=c["spec"], return_pools=return_pools,
                            ev_inert=True)
    obs, pools = out if return_pools else (out, None)
    n_full = int(obs.get("n_lane_changes_full", 0))
    obs["n_lane_changes_full"] = n_full
    if n_full < min_events:
        for k in LC_KINEMATIC_KEYS:
            obs[k] = float("nan")
        obs["lc_kinematics_unreliable"] = True
        if pools is not None:
            for k in ("lc_dur_full", "lc_c2c", "lc_peak_vy_full"):
                pools[k] = np.zeros(0, np.float32)
    return (obs, pools) if return_pools else obs


def loss_normal(theta, ctx: dict, seeds=(11, 12, 13), names=None,
                base: Params | None = None, spec_theta=None,
                w_shape: float = W_SHAPE, regimes=None,
                return_detail: bool = False, clip: bool = True):
    """Stage-1 objective: ordinary traffic against highD, EV absent.

    The total is the **mean over regimes** of (moment + w_shape * shape). Summing
    rather than fitting free-flow alone is not a refinement, it is a requirement:
    `k_rho` is a density exponent, so a single density cannot identify it - and
    the free-flow-only fit of 2026-07-29 is precisely what drove `a_pin` down 84 %
    and left the congested block worse than no calibration at all.

    Regimes are weighted equally rather than by carriageway count (44 free-flow
    against 27 dense). Weighting by evidence would hand free-flow 62 % of the
    objective and reproduce the failure this is designed to avoid; the price is
    that the dense block is over-weighted relative to how much data supports it,
    which is stated rather than hidden.

    Deviations are capped at `D_CAP` and an unproducible observable is priced at
    `D_CAP` too - see its docstring, because getting that wrong makes "make no
    lane changes at all" the cheapest option available to the optimiser.
    """
    regimes = tuple(regimes or ctx.keys())
    p = make_params_spec(theta, names, base, spec_theta, clip=clip)
    per, tot = {}, []
    for r in regimes:
        obs, mp = normal_observables(p, ctx, r, seeds, return_pools=True)
        d = distance(obs, ctx[r]["tgt"], OBS_SPEC_NORMAL,
                     nan=D_CAP, cap=D_CAP)
        sh = shape_distance(mp, data_pools(), SHAPE_KEYS_NORMAL,
                            miss=SHAPE_MISS)
        shp = sh["total"] if np.isfinite(sh["total"]) else 2.0
        pen = min(5.0 * obs.get("collisions", 0), 50.0)
        val = d["total"] + w_shape * shp + pen
        per[r] = dict(moment=d["total"], shape=round(float(shp), 4),
                      collisions=pen, total=round(float(val), 4),
                      distance=d, shape_parts=sh,
                      observables=obs if return_detail else None)
        tot.append(val)
    total_ = float(np.mean(tot))
    parts = {f"{r}_{k}": per[r][k] for r in regimes
             for k in ("moment", "shape", "collisions")}
    parts["_total"] = round(total_, 4)
    return (total_, parts, per) if return_detail else (total_, parts)


def make_params_spec(theta, names=None, base: Params | None = None,
                     spec=None, clip: bool = True) -> Params:
    """`make_params` for an arbitrary parameter spec.

    `clip=False` is required by the ablation study: switching a term off means
    setting its amplitude to exactly 0, and several amplitudes have search
    bounds that start above 0 (`a_pin` at 0.15, `A_pass` at 0.5, because values
    below the pinning threshold are all equivalent to "off"). Clipping would
    silently turn "no lane discipline" into "a little lane discipline".
    """
    spec = spec or SPEC_NORMAL
    names = names or [s[0] for s in spec]
    lo = np.array([s[1] for s in spec], float)
    hi = np.array([s[2] for s in spec], float)
    th = np.asarray(theta, float)
    if clip and th.size == lo.size:
        th = np.clip(th, lo, hi)
    p = base if base is not None else Params()
    return p.copy(**dict(zip(names, th.tolist())))
