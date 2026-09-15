"""Scenario builders.

* make_overtake: free-flowing 3-lane motorway, EV approaches from behind at
  high speed; corridor = the EV's own lane (US-style 'vacate the lane').
* make_jam: dense stop-and-go traffic; corridor on the boundary between the
  leftmost and middle lane (German Rettungsgasse rule: leftmost lane pulls
  left, everyone else pulls right) [17, 21].

Baselines use yielding=False: drivers never become aware of the EV, which
then has to make progress on car-following alone.
"""
import numpy as np

from .params import Params
from .road import Road
from .state import VehState, blank_state
from .simulate import Sim


def _spawn_traffic(rng, road: Road, x_lo, x_hi, spacing, lane_speeds, jitter_x, p):
    xs, ys, lanes, v0s = [], [], [], []
    for k in range(road.n_lanes):
        x = x_lo + rng.uniform(0.0, spacing)
        while x < x_hi:
            xs.append(x + rng.uniform(-jitter_x, jitter_x))
            ys.append(road.lane_center(k) + rng.uniform(-0.25, 0.25))
            lanes.append(k)
            v0s.append(max(rng.normal(lane_speeds[k], 1.2), 2.0))
            x += spacing
    return (np.array(xs), np.array(ys), np.array(lanes, dtype=int), np.array(v0s))


def _assemble(rng, road, p, xs, ys, v0s, ev_kw, seed, yielding, name):
    n = xs.size + 1
    d = blank_state(n)
    # EV is index 0
    d["x"][0], d["y"][0] = ev_kw["x"], ev_kw["y"]
    d["vx"][0] = ev_kw["v_init"]
    d["v0"][0] = ev_kw["v0"]
    d["L"][0], d["W"][0] = 6.2, 2.2          # ambulance footprint [16]
    d["tau"][0] = p.tau_ev
    d["T_hw"][0], d["s0"][0] = p.ev_T, p.ev_s0
    d["amax"][0], d["bcomf"][0] = p.ev_amax, p.ev_b
    d["is_ev"][0] = True

    order = np.argsort(xs)
    d["x"][1:], d["y"][1:] = xs[order], ys[order]
    d["v0"][1:] = v0s[order]
    d["vx"][1:] = d["v0"][1:] * 0.95
    d["tau"][1:] = p.tau
    d["T_hw"][1:] = rng.uniform(1.1, 1.8, n - 1)
    d["s0"][1:] = p.s0
    d["amax"][1:] = p.a_max
    d["bcomf"][1:] = p.b_comf

    st = VehState(**d)
    return Sim(st, road, p, y_corr=ev_kw["y_corr"], seed=seed + 7919,
               yielding=yielding, name=name)


def make_overtake(seed: int = 1, p: Params | None = None, density: float = 22.0,
                  n_lanes: int = 3, road_len: float = 2400.0,
                  yielding: bool = True) -> Sim:
    """EV overtaking free-flowing traffic. density in veh/km/lane."""
    p = p or Params()
    rng = np.random.default_rng(seed)
    road = Road(n_lanes=n_lanes, lane_width=3.5, shoulder_right=2.0,
                shoulder_left=0.8, length=road_len)
    spacing = 1000.0 / density
    lane_speeds = [24.5, 27.5, 30.0][:n_lanes]
    xs, ys, _, v0s = _spawn_traffic(rng, road, 200.0, road_len - 120.0,
                                    spacing, lane_speeds, jitter_x=0.18 * spacing, p=p)
    ev_lane = min(1, n_lanes - 1)
    ev_kw = dict(x=40.0, y=road.lane_center(ev_lane), v_init=30.0, v0=36.0,
                 y_corr=road.lane_center(ev_lane))
    return _assemble(rng, road, p, xs, ys, v0s, ev_kw, seed,
                     yielding, f"overtake_d{density:g}_s{seed}" + ("" if yielding else "_base"))


def make_jam(seed: int = 3, p: Params | None = None, spacing: float = 13.0,
             n_lanes: int = 3, road_len: float = 760.0,
             yielding: bool = True) -> Sim:
    """Rescue-lane formation in a crawling jam. Corridor between the leftmost
    and middle lane; EV threads the opened gap at moderate speed."""
    p = p or Params()
    rng = np.random.default_rng(seed)
    road = Road(n_lanes=n_lanes, lane_width=3.5, shoulder_right=2.5,
                shoulder_left=1.0, length=road_len)
    lane_speeds = [2.5] * n_lanes
    xs, ys, _, v0s = _spawn_traffic(rng, road, 70.0, road_len - 40.0,
                                    spacing, lane_speeds, jitter_x=1.8, p=p)
    y_corr = (n_lanes - 1) * road.lane_width          # boundary left of middle lane
    ev_kw = dict(x=20.0, y=y_corr, v_init=6.0, v0=9.0, y_corr=y_corr)
    return _assemble(rng, road, p, xs, ys, v0s, ev_kw, seed,
                     yielding, f"jam_sp{spacing:g}_s{seed}" + ("" if yielding else "_base"))


# ---------------------------------------------------------------------------
# SUMO-mirror scenario (the one behind out/anim_surrogate_sumo.gif).
# Moved verbatim from experiments/run_surrogate_sumo.py so the standalone game
# bridge (emv/ue) and the surrogate experiment share one source of truth.
# The RNG call order MUST stay byte-identical or the surrogate results change.
# ---------------------------------------------------------------------------
def make_sumo_like(seed: int, p: Params) -> Sim:
    """Standalone scenario that mirrors the SUMO network and demand
    (3 x 3.5 m motorway, 3 km, 1700 veh/h/lane, speedFactor normc(0.92,0.08),
    EV desired speed min(41, 1.40 x 27.78) = 38.9 m/s). The corridor sits on
    the boundary between the two leftmost lanes = where SUMO's bluelight rescue
    gap forms, so the surrogate matches the mechanism, not just the numbers."""
    N_LANES, LANE_W, EDGE_LEN, V_MAX = 3, 3.5, 3000.0, 27.78
    FLOW = 1700.0                      # veh/h/lane
    EV_V0 = min(41.0, 1.40 * V_MAX)    # SUMO ev type: maxSpeed 41, speedFactor 1.4

    rng = np.random.default_rng(seed)
    road = Road(n_lanes=N_LANES, lane_width=LANE_W, shoulder_right=0.0,
                shoulder_left=0.0, length=EDGE_LEN)
    v_mean = 0.92 * V_MAX
    spacing = 1000.0 / (FLOW / (v_mean * 3.6))        # veh/km at mean speed
    xs, ys, v0s = [], [], []
    for k in range(N_LANES):
        x = 130.0 + rng.uniform(0.0, spacing)
        while x < EDGE_LEN - 80.0:
            xs.append(x + rng.uniform(-0.2, 0.2) * spacing)
            ys.append(road.lane_center(k) + rng.uniform(-0.25, 0.25))
            # speedFactor normc(0.92, 0.08, 0.6, 1.2) on the edge limit
            f = float(np.clip(rng.normal(0.92, 0.08), 0.6, 1.2))
            v0s.append(f * V_MAX)
            x += spacing
    xs, ys, v0s = map(np.array, (xs, ys, v0s))

    n = xs.size + 1
    d = blank_state(n)
    y_corr = (N_LANES - 1) * LANE_W                   # rescue-gap boundary
    d["x"][0], d["y"][0] = 30.0, y_corr
    d["vx"][0], d["v0"][0] = V_MAX, EV_V0
    d["L"][0], d["W"][0] = 6.2, 2.2
    d["tau"][0] = p.tau_ev
    d["T_hw"][0], d["s0"][0] = p.ev_T, p.ev_s0
    d["amax"][0], d["bcomf"][0] = p.ev_amax, p.ev_b
    d["is_ev"][0] = True
    order = np.argsort(xs)
    d["x"][1:], d["y"][1:] = xs[order], ys[order]
    d["v0"][1:] = v0s[order]
    d["vx"][1:] = d["v0"][1:] * 0.97
    d["tau"][1:] = p.tau
    d["T_hw"][1:] = rng.uniform(1.1, 1.8, n - 1)
    d["s0"][1:] = p.s0
    d["amax"][1:] = p.a_max
    d["bcomf"][1:] = p.b_comf
    return Sim(VehState(**d), road, p, y_corr=y_corr, seed=seed + 7919,
               yielding=True, name=f"sumo_like_s{seed}")


def surrogate_params(json_path: str | None = None) -> Params:
    """Params that reproduce out/anim_surrogate_sumo.gif: the fitted theta from
    out/surrogate_sumo.json applied on top of the GT-mirror regime, identical
    to experiments/run_surrogate_sumo.py::make_params(theta)."""
    import json
    import os
    if json_path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        json_path = os.path.join(root, "out", "surrogate_sumo.json")
    with open(json_path) as fh:
        theta = json.load(fh)["theta"]
    return Params().copy(**theta, dt=0.06, p_noncomply=0.0, v_lat_max=1.5)


#: Overrides for the drivable UE demo, on top of surrogate_params(). NOT a
#: calibration result - the calibrated theta is fitted to a *model-driven* EV
#: that closes on traffic fast, whereas a human drives ~120 km/h into ~90 km/h
#: traffic, i.e. ~8 m/s closing speed. Under the pure time-to-arrival channel
#: that means drivers only feel urgency ~50 m out, which reads as "nobody gets
#: out of my way". These values restore the *intended* behaviour (yield ~150 m
#: ahead, pull aside, brake, wait) at a human closing speed.
GAME: dict = dict(
    # --- act on siren distance, not just time-to-arrival ---
    prox_gate_floor=1.0,      # proximity channel at any speed (see params.py)
    R_urgent=140.0,           # m, start clearing this far ahead
    sigma_s=25.0,             # m, smooth onset rather than a switch
    T_react=14.0,             # s, also stretch the time channel
    # --- notice sooner, react sooner ---
    R_front=220.0,            # m, siren/lights detection ahead of the EV
    delay_med=0.7, delay_sig=0.35,
    # --- move aside harder and faster ---
    A_c=3.2,                  # m/s^2, corridor push (surrogate: 1.51)
    B_c=1.6,                  # m
    margin_c=0.8,             # m, aim for a wider gap
    a_pin=1.0,                # m/s^2, easier depinning out of the lane
    urgency_pin_relief=0.6,   # -, urgency relaxes lane discipline further
    v_lat_max=2.6,            # m/s, faster sidestep (surrogate: 1.5)
    a_lat_max=3.5,            # m/s^2
    # --- and slow down while doing it, then wait ---
    kappa_merge=0.55,         # -, target speed fraction while merging out
    gamma_c=1.30,             # 1/s, longitudinal yield-brake gain
    T_hold_lo=3.0, T_hold_hi=6.5,   # s, hold the gap after the EV passes
)


def game_params(**over) -> Params:
    """surrogate_params() retuned for a human-driven EV (see GAME)."""
    d = dict(GAME)
    d.update(over)
    return surrogate_params().copy(**d)


# ---------------------------------------------------------------------------
# highD-matched scenario (real German motorway geometry, fleet and demand).
# Added 2026-07-28. Separate from make_overtake/make_jam on purpose: those two
# are the scenarios of record for papers 1-4 and their RNG call order must not
# move. Everything here is SET from measurement (emv/highd.py), not fitted.
# ---------------------------------------------------------------------------
#: Regime specifications measured on highD. `lane_speeds`/`lane_speed_sd` are
#: per-lane free-driving means, lane 0 = rightmost. Fallbacks are used when
#: out/highd_recordings.json is absent; the loader below prefers the measured
#: file. Sources: 337 lane widths (median 3.89 m), per-lane speed and truck
#: profiles from the recordings named in each entry.
HIGHD_REGIMES: dict = dict(
    freeflow=dict(
        recordings=(7, 8, 9, 10), n_lanes=3, lane_width=3.74,
        lane_speeds=(25.1, 34.4, 40.7), lane_speed_sd=(3.7, 4.4, 4.7),
        truck_share=(0.64, 0.09, 0.01), density=8.9, ev_v0=41.0,
        note="location 4, unrestricted, ~890 veh/h/lane"),
    dense=dict(
        recordings=(11, 12), n_lanes=3, lane_width=3.96,
        lane_speeds=(23.3, 25.4, 27.1), lane_speed_sd=(3.0, 3.5, 4.0),
        truck_share=(0.45, 0.10, 0.01), density=17.0, ev_v0=41.0,
        note="location 1, ~1750 veh/h/lane - the density twin of make_sumo_like"),
    congested=dict(
        recordings=(25, 26), n_lanes=3, lane_width=3.96,
        lane_speeds=(7.3, 9.8, 10.7), lane_speed_sd=(3.7, 3.5, 3.9),
        truck_share=(0.48, 0.12, 0.01), density=37.8, ev_v0=20.0,
        note="location 1 dir 1, genuine congestion; highD never fully stops, so "
             "make_jam (2.5 m/s) remains an extrapolation beyond this dataset"),
)

#: Fleet dimensions by class (highD tracksMeta; the model's homogeneous default
#: is L=4.5 W=1.8). Overridden by measured values when available.
HIGHD_FLEET: dict = dict(car=dict(L_mean=4.79, L_sd=0.73, W_mean=1.95, W_sd=0.15),
                         truck=dict(L_mean=14.3, L_sd=4.2, W_mean=2.50, W_sd=0.02))


def highd_regime(name: str = "freeflow", path: str | None = None) -> dict:
    """Regime spec, preferring the measured out/highd_recordings.json.

    Falls back to HIGHD_REGIMES so the scenario is usable on a machine that has
    never seen the dataset (the same contract as emv/empirical.py).
    """
    import json
    import os
    spec = dict(HIGHD_REGIMES[name])
    if path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, "out", "highd_recordings.json")
    if not os.path.exists(path):
        return spec
    with open(path) as fh:
        js = json.load(fh)
    recs = js.get("recordings", {})
    wins = [w for w in js.get("windows", []) if w.get("regime") == name]
    if not wins:
        return spec
    keys = sorted({f"{w['rid']:02d}" for w in wins})
    lane_med, lane_sd, truck, widths = [], [], [], []
    for k in keys:
        prof = recs.get(k, {}).get("lane_profile", {})
        for d, pr in prof.items():
            if len(pr.get("lane_med", [])) != spec["n_lanes"]:
                continue
            lane_med.append(pr["lane_med"])
            lane_sd.append(pr["lane_sd"])
            truck.append(pr["truck_share"])
        widths.append(recs[k]["meta"]["lane_width_med"])
    if lane_med:
        nan_mean = lambda A: tuple(round(float(v), 3) for v in
                                   np.nanmean(np.asarray(A, float), axis=0))
        spec.update(lane_speeds=nan_mean(lane_med), lane_speed_sd=nan_mean(lane_sd),
                    truck_share=nan_mean(truck),
                    lane_width=round(float(np.mean(widths)), 3),
                    recordings=tuple(int(k) for k in keys))
    dens = [w["flow_veh_h_lane"] / max(w["med_speed"] * 3.6, 1e-6) * 1000.0 / 1000.0
            for w in wins if w["med_speed"] > 0.5]
    if dens:
        spec["density"] = round(float(np.median(dens)), 2)
    spec["measured"] = True
    return spec


def make_highd_like(seed: int = 1, p: Params | None = None,
                    regime: str = "freeflow", spec: dict | None = None,
                    road_len: float = 1500.0, yielding: bool = True,
                    ev_inert: bool = False, fleet: dict | None = None) -> Sim:
    """Scenario mirroring real highD motorway geometry, fleet mix and demand.

    Everything that describes the *road and the fleet* is measured, not fitted:
    lane count and width, per-lane desired-speed means and spreads, the truck
    share per lane with class-conditional dimensions, and the density. That
    matters because the model's own free-flow scenario is not free-flowing by
    highD's standard (`make_overtake` uses 22 veh/km/lane against a measured
    ~9-12) and its lane-speed profile is far too flat (24.5/27.5/30.0 against a
    measured 25/34/41 - real left-lane traffic outruns the model's EV).

    `ev_inert=True` parks the EV 5 km beyond the road end with `v0=0`, well
    outside `pair_cutoff`, giving a clean EV-free control: the model must
    reproduce highD's host traffic with no emergency vehicle present at all.
    """
    p = p or Params()
    sp = spec or highd_regime(regime)
    fl = fleet or HIGHD_FLEET
    rng = np.random.default_rng(seed)
    n_lanes = int(sp["n_lanes"])
    road = Road(n_lanes=n_lanes, lane_width=float(sp["lane_width"]),
                shoulder_right=2.0, shoulder_left=0.8, length=road_len)
    spacing = 1000.0 / float(sp["density"])

    xs, ys, v0s, Ls, Ws, is_truck = [], [], [], [], [], []
    for k in range(n_lanes):
        x = 200.0 + rng.uniform(0.0, spacing)
        while x < road_len - 120.0:
            truck = rng.random() < float(sp["truck_share"][k])
            cls = fl["truck"] if truck else fl["car"]
            xs.append(x + rng.uniform(-0.18 * spacing, 0.18 * spacing))
            ys.append(road.lane_center(k) + rng.uniform(-0.25, 0.25))
            v0s.append(max(rng.normal(float(sp["lane_speeds"][k]),
                                      float(sp["lane_speed_sd"][k])), 2.0))
            Ls.append(max(rng.normal(cls["L_mean"], cls["L_sd"]), 3.2))
            Ws.append(max(rng.normal(cls["W_mean"], cls["W_sd"]), 1.5))
            is_truck.append(truck)
            x += spacing
    xs = np.array(xs)
    order = np.argsort(xs)
    n = xs.size + 1

    d = blank_state(n)
    ev_lane = min(1, n_lanes - 1)
    y_corr = road.lane_center(ev_lane)
    if ev_inert:
        d["x"][0], d["y"][0] = road_len + 5000.0, y_corr
        d["vx"][0], d["v0"][0] = 0.0, 0.0
    else:
        d["x"][0], d["y"][0] = 40.0, y_corr
        d["vx"][0] = 0.85 * float(sp["ev_v0"])
        d["v0"][0] = float(sp["ev_v0"])
    d["L"][0], d["W"][0] = 6.2, 2.2
    d["tau"][0] = p.tau_ev
    d["T_hw"][0], d["s0"][0] = p.ev_T, p.ev_s0
    d["amax"][0], d["bcomf"][0] = p.ev_amax, p.ev_b
    d["is_ev"][0] = True

    d["x"][1:], d["y"][1:] = xs[order], np.array(ys)[order]
    d["v0"][1:] = np.array(v0s)[order]
    # Initial speeds must be consistent with the vehicle ahead in the same lane,
    # or a fast draw spawned behind a slow truck has to brake at b_emerg in the
    # first second and that start-up transient dominates the deceleration tail
    # (it pinned peak_decel_p99 at exactly 8.0 m/s^2 before this).
    lane_of = np.clip(np.round(np.array(ys)[order] / road.lane_width - 0.5),
                      0, n_lanes - 1).astype(int)
    v_init = d["v0"][1:] * 0.95
    for k in range(n_lanes):
        idx = np.flatnonzero(lane_of == k)          # ascending x
        for a, b in zip(idx[:-1], idx[1:]):         # b is ahead of a
            v_init[a] = min(v_init[a], v_init[b])
    d["vx"][1:] = v_init
    d["L"][1:], d["W"][1:] = np.array(Ls)[order], np.array(Ws)[order]
    d["tau"][1:] = p.tau
    # Parameterised (2026-08-06) so the car-following block can be fitted against
    # highD's measured headway distribution. Defaults are the literal 1.1/1.8 the
    # builders used before and the RNG call is identical, so this scenario's
    # existing records are unchanged; make_overtake/make_jam/make_sumo_like keep
    # the literals because their RNG call order is the record for papers 1-4.
    d["T_hw"][1:] = rng.uniform(p.T_hw_lo, p.T_hw_hi, n - 1)
    d["s0"][1:] = p.s0
    d["amax"][1:] = p.a_max
    d["bcomf"][1:] = p.b_comf
    if p.sigma_off > 0:
        d["y_pref"][1:] = rng.normal(0.0, p.sigma_off, n - 1)
    if p.het_lat > 0:
        # Lognormal so the draws stay positive and the median is the fitted
        # value; the EV keeps the global block (index 0 is restored below).
        #
        # Two traits, not four independent draws. A driver's lateral speed and
        # lateral acceleration limits are the same trait - how briskly they are
        # willing to move sideways - so they share one factor; drawing them
        # independently would produce incoherent drivers (a high acceleration
        # ceiling with a low speed ceiling). Lane-keeping discipline (a_pin,
        # zeta_lat) is a separate trait with its own factor.
        #
        # a_lat_max is included here because leaving it global made it a single
        # hard ceiling: 15 % of tracks sat exactly on it and the p90, p95 and p99
        # of lateral acceleration collapsed onto the same number, truncating the
        # distribution instead of reproducing its tail
        # (docs/HIGHD_VALIDATION.md sec. 7c).
        s = np.sqrt(np.log1p(p.het_lat ** 2))
        draw = lambda: rng.lognormal(-0.5 * s * s, s, n)
        envelope, discipline = draw(), draw()
        d["v_lat_max"] = p.v_lat_max * envelope
        d["a_lat_max"] = p.a_lat_max * envelope
        d["a_pin"] = p.a_pin * discipline
        d["zeta_lat"] = p.zeta_lat * draw()
        d["v_lat_max"][0], d["a_lat_max"][0] = p.v_lat_max, p.a_lat_max
        d["a_pin"][0], d["zeta_lat"][0] = p.a_pin, p.zeta_lat

    name = (f"highd_{regime}_s{seed}" + ("_inert" if ev_inert else "")
            + ("" if yielding else "_base"))
    return Sim(VehState(**d), road, p, y_corr=y_corr, seed=seed + 7919,
               yielding=yielding, name=name)
