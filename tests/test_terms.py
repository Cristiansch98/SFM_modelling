"""Offline guards for the two-stage framework extension (no dataset required).

Guards the four things that would be expensive to discover late:

1. **Nothing published moved.** Every parameter that existed before this
   extension still has its old default, the new ones are all neutral, and with
   default parameters the model still makes exactly zero discretionary lane
   changes - which is what keeps papers 1-6 reproducible.
2. **The nesting invariant.** With the emergency-vehicle terms switched off
   (`A_ev = A_c = gamma_c = 0`) the bluelight model reproduces the normal model
   *bit for bit*. This is the property that makes the two-stage calibration
   meaningful: stage 2 adds terms, it does not perturb stage 1.
3. **The force budget adds up.** The per-term diagnostic must reconstruct the
   accelerations that were actually integrated, including the IDM `min`.
4. **The headway definitions match highD's**, checked against a hand-built
   two-vehicle case with known geometry.

Plus a registry guard: every term in `forces.total(diag=True)` must appear in
`terms.TERMS`, so a new force term cannot be added without being analysed.

    python tests/test_terms.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import forces, metrics, terms
from emv.params import Params, TUNED, TUNED_HIGHD, highd_params
from emv.road import Road, build_corridor
from emv.scenarios import highd_regime, make_highd_like
from emv.state import VehState, blank_state

FAIL = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


def digest(h):
    import hashlib
    m = hashlib.sha256()
    for a in (h.x, h.y, h.vx, h.vy, h.ax):
        m.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return m.hexdigest()[:16]


# ------------------------------------------------------- 1. nothing moved
print("1. published behaviour untouched")
p0 = Params()
OLD = dict(dt=0.05, tau=0.55, tau_ev=0.40, a_max=2.6, b_comf=3.0, b_emerg=8.0,
           s0=2.0, idm_overlap_margin=0.5, aware_amax_boost=1.35, a_pin=1.4,
           zeta_lat=1.0, urgency_pin_relief=0.5, a_lat_max=3.0, v_lat_max=2.2,
           sigma_off=0.0, het_lat=0.0, A_v=2.5, Bx_v=5.0, By_v=0.9, A_near=6.0,
           q_near=0.25, lam_v=0.25, A_ev=4.5, B_ev=18.0, lam_ev=0.10, A_c=4.2,
           B_c=1.6, margin_c=0.5, gamma_c=0.55, kappa_merge=0.82, T_react=9.0,
           R_front=140.0, R_rear=60.0, p_noncomply=0.05)
moved = {k: (v, getattr(p0, k)) for k, v in OLD.items() if getattr(p0, k) != v}
check(not moved, f"pre-existing Params defaults unchanged ({len(OLD)} checked)")
check(TUNED == dict(A_ev=2.152, B_ev=21.406, A_c=3.758, B_c=0.748,
                    T_react=15.944, gamma_c=0.720), "TUNED untouched")
check(TUNED_HIGHD["a_pin"] == 0.2236 and TUNED_HIGHD["het_lat"] == 0.4257,
      "TUNED_HIGHD untouched")
check(p0.A_pass == 0.0 and p0.A_keep_right == 0.0 and p0.k_rho == 0.0
      and p0.idm_bound is True,
      "every new parameter defaults to its neutral value")
check(p0.T_hw_lo == 1.1 and p0.T_hw_hi == 1.8,
      "T_hw bounds default to the literals the scenario builders used")

sp = highd_regime("freeflow")
base = highd_params().copy(dt=0.06)
h_off = make_highd_like(seed=11, p=base, regime="freeflow", spec=sp,
                        road_len=1500.0, ev_inert=True).run(40.0, rec_dt=0.12)
check(len(metrics.lane_change_events(h_off)) == 0,
      "with A_pass=0 the EV-free control still makes no lane change at all")

# the extension is reachable: a nonzero incentive does produce manoeuvres
h_on = make_highd_like(seed=11, p=base.copy(A_pass=6.0, A_keep_right=4.0),
                       regime="freeflow", spec=sp, road_len=1500.0,
                       ev_inert=True).run(40.0, rec_dt=0.12)
n_on = len(metrics.lane_change_events(h_on))
check(n_on > 0, f"a nonzero incentive produces discretionary lane changes ({n_on})")

# --------------------------------------------------- 2. stage separation
print("\n2. stage separation (stage-2 parameters are inert in the normal model)")
kw = dict(regime="freeflow", spec=sp, road_len=1500.0)
p_norm = base.copy(A_pass=6.0, A_keep_right=4.0)
#: the four parameters stage 2 fits, perturbed well outside their search bounds
STAGE2 = dict(A_ev=9.9, B_ev=39.0, A_c=9.9, B_c=2.9)
for label, extra in (("EV inert", dict(ev_inert=True)),
                     ("perception off", dict(yielding=False))):
    a = make_highd_like(seed=11, p=p_norm, **kw, **extra).run(40.0, rec_dt=0.12)
    b = make_highd_like(seed=11, p=p_norm.copy(**STAGE2), **kw, **extra).run(
        40.0, rec_dt=0.12)
    check(digest(a) == digest(b),
          f"{label}: the stage-2 parameters change nothing (bit-identical)")
# ...and they are not inert once there is an emergency vehicle to respond to
c = make_highd_like(seed=11, p=p_norm, **kw).run(40.0, rec_dt=0.12)
d_ = make_highd_like(seed=11, p=p_norm.copy(**STAGE2), **kw).run(40.0, rec_dt=0.12)
check(digest(c) != digest(d_),
      "with an EV present the stage-2 parameters do change the run")

# The naive invariant - "zero the EV force amplitudes and you are back to the
# normal model" - is FALSE, and it matters enough to be asserted rather than
# assumed: perceiving a siren also engages aware_amax_boost (yielding drivers
# accept 1.35x the acceleration) and urgency_pin_relief (urgency relaxes lane
# discipline), neither of which is a force amplitude. So the emergency-vehicle
# extension has five channels, not four, and the fifth acts on host-traffic
# parameters. Recorded in docs/TERM_INFLUENCE.md; the ablation registry carries
# 'no_urgency_relief' precisely to quantify it.
e = make_highd_like(seed=11, p=p_norm.copy(A_ev=0.0, A_c=0.0, gamma_c=0.0),
                    **kw).run(40.0, rec_dt=0.12)
f_ = make_highd_like(seed=11, p=p_norm, yielding=False, **kw).run(40.0, rec_dt=0.12)
check(digest(e) != digest(f_),
      "zeroing the EV forces is NOT the same as switching perception off "
      "(aware_amax_boost + urgency_pin_relief still act)")
CLOSED = dict(A_ev=0.0, A_c=0.0, gamma_c=0.0, aware_amax_boost=1.0,
              urgency_pin_relief=0.0)
g_ = make_highd_like(seed=11, p=p_norm.copy(**CLOSED), **kw).run(40.0, rec_dt=0.12)
check(digest(g_) != digest(f_),
      "closing those two leaves a SIXTH channel: a driver already yielding does "
      "not start a discretionary manoeuvre (emv/lanechange.py)")
# with the discretionary term off, the five channels are the whole extension
p_five = base.copy()
h5a = make_highd_like(seed=11, p=p_five, yielding=False, **kw).run(40.0, rec_dt=0.12)
h5b = make_highd_like(seed=11, p=p_five.copy(**CLOSED), **kw).run(40.0, rec_dt=0.12)
check(digest(h5a) == digest(h5b),
      "nesting invariant: with A_pass=0, closing all five channels reproduces "
      "the EV-free model bit for bit")

# --------------------------------------------------- 3. force budget adds up
print("\n3. force budget accounting")
sim = make_highd_like(seed=11, p=p_norm, **kw)
for _ in range(60):
    sim.step()
corr = sim._corridor()
sim.per.update(sim.st, corr, sim.t)
ax, ay, dg = forces.total(sim.st, sim.road, sim.p, corr, diag=True)
sum_y = sum(dg[k][1] for k in ("drive", "lane", "lc", "sfm", "ev", "corr"))
check(np.allclose(ay, sum_y, atol=1e-12),
      f"lateral: sum of terms == ay (max dev {np.abs(ay - sum_y).max():.2e})")
social = dg["sfm"][0] + dg["ev"][0] + dg["corr"][0]
sum_x = dg["drive"][0] + np.minimum(social, dg["idm"])
check(np.allclose(ax, sum_x, atol=1e-12),
      f"longitudinal: drive + min(social, a_IDM) == ax "
      f"(max dev {np.abs(ax - sum_x).max():.2e})")

hist, rep = terms.budget_run(make_highd_like(seed=11, p=p_norm, **kw),
                             20.0, rec_dt=0.12)
share = rep["share_y"]
check(abs(sum(share.values()) - 1.0) < 1e-6,
      "lateral budget shares sum to 1")
check(set(share) == {t for t, (k, a, _, _) in terms.TERMS.items() if k != "idm"},
      "budget covers every summand term")
check(rep["mean_abs_y"]["lc"] > 0.0,
      "the discretionary term shows up in the budget when it is switched on")

# ------------------------------------------------------ 4. registry guard
print("\n4. registry guard")
diag_keys = {k for k in dg if k != "leader"}
reg_keys = {k for k, (key, _, _, _) in terms.TERMS.items() for k in (key,)}
check(diag_keys == reg_keys,
      f"terms.TERMS covers exactly the diag keys (diag-only: "
      f"{sorted(diag_keys - reg_keys)}, registry-only: {sorted(reg_keys - diag_keys)})")
bad = [n for n, v in terms.ABLATIONS.items()
       for k in v[0] if not hasattr(p0, k)]
check(not bad, f"every ablation names real parameters ({bad})")
check(set(terms.N_ABLATIONS) | set(terms.E_ABLATIONS) == set(terms.ABLATIONS)
      and not (set(terms.N_ABLATIONS) & set(terms.E_ABLATIONS)),
      "every ablation is assigned to exactly one stage")
# the two host-traffic parameters that only act on a driver responding to a
# siren must be classified E, or they would be scored against an objective that
# provably cannot see them (this is the bug the quick run surfaced)
check(all(terms.ABLATIONS[n][2] == "E"
          for n in ("no_urgency_relief", "no_aware_boost")),
      "siren-gated host parameters are classified as stage-2 terms")
check(all(terms.ablate(p0, n) is not p0 for n in terms.ABLATIONS),
      "ablate() returns a modified copy for every registered ablation")
p_nested = terms.ablate(p_norm, "no_ev_response")
check(p_nested.A_ev == 0 and p_nested.A_c == 0 and p_nested.gamma_c == 0,
      "no_ev_response switches off all three EV force amplitudes")
p_all = terms.ablate(p_norm, "no_ev_at_all")
check(p_all.aware_amax_boost == 1.0 and p_all.urgency_pin_relief == 0.0,
      "no_ev_at_all also closes the two host-parameter channels")

# ------------------------------------------- 4b. objective is not gameable
print("\n4b. the EV-free objective does not reward switching the term off")
from emv import highd_fit as hf                                   # noqa: E402
KEYS = [k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0]
tgt_stub = {k: 1.0 for k in KEYS}
scales = {k: s for k, s, _ in hf.OBS_SPEC_NORMAL}
# a model that produces nothing at all vs one that produces something bad
none_obs = {k: float("nan") for k in KEYS}
bad_obs = {k: 1.0 + 20.0 * scales[k] for k in KEYS}
d_none = hf.distance(none_obs, tgt_stub, hf.OBS_SPEC_NORMAL,
                     nan=hf.D_CAP, cap=hf.D_CAP)["total"]
d_bad = hf.distance(bad_obs, tgt_stub, hf.OBS_SPEC_NORMAL,
                    nan=hf.D_CAP, cap=hf.D_CAP)["total"]
check(d_none >= d_bad - 1e-9,
      f"producing nothing ({d_none:.2f}) is never cheaper than producing "
      f"something maximally wrong ({d_bad:.2f})")
d_pub = hf.distance(none_obs, tgt_stub, hf.OBS_SPEC_NORMAL)["total"]
check(abs(d_pub - 2.0) < 1e-9,
      "the published objective's NaN price is untouched (2.0 scale units)")

# the same hole existed in the SHAPE term, where an empty model pool was skipped
# rather than priced, so the average was taken over the remaining keys only
data_stub = {k: np.linspace(0.0, 4.0, 500) for k in hf.SHAPE_KEYS_NORMAL}
mp_none = {k: np.zeros(0) for k in hf.SHAPE_KEYS_NORMAL}
mp_two = dict(mp_none)
for k in ("trk_offset_abs", "thw"):
    mp_two[k] = np.linspace(0.0, 4.0, 500)      # two keys matched exactly
sh_skip = hf.shape_distance(mp_two, data_stub, hf.SHAPE_KEYS_NORMAL)["total"]
sh_priced = hf.shape_distance(mp_two, data_stub, hf.SHAPE_KEYS_NORMAL,
                              miss=hf.SHAPE_MISS)["total"]
check(abs(sh_skip) < 1e-9 and sh_priced > 1.0,
      f"unproducible distributions are priced, not skipped "
      f"(skip {sh_skip:.3f} -> priced {sh_priced:.3f})")
sh_all_none = hf.shape_distance(mp_none, data_stub, hf.SHAPE_KEYS_NORMAL,
                                miss=hf.SHAPE_MISS)["total"]
check(sh_all_none >= sh_priced,
      "producing no distribution at all is never better than producing two")

# ...and the third instance: a median taken over a handful of manoeuvres was
# scored as if it meant something. The lat fit produced 5 complete manoeuvres over
# 612 veh-km and was still credited with a duration 1.25 s.d. from target.
ctx_stub = {"freeflow": dict(window=13.08, spec=sp)}
few = dict(n_lane_changes_full=3, lc_dur_full_med=2.28, lc_c2c_med=3.48,
           lc_peak_vy_full_med=0.955, lc_dur_full_p90=3.36,
           lc_peak_vy_full_p90=1.239, lc_per_veh_km_complete=0.008)


def _blank(obs, n):
    o = dict(obs, n_lane_changes_full=n)
    if n < hf.LC_MIN_EVENTS:
        for k in hf.LC_KINEMATIC_KEYS:
            o[k] = float("nan")
    return o


check(all(not np.isfinite(_blank(few, 3)[k]) for k in hf.LC_KINEMATIC_KEYS),
      f"kinematics from fewer than {hf.LC_MIN_EVENTS} manoeuvres are not scored")
check(all(np.isfinite(_blank(few, 50)[k]) for k in hf.LC_KINEMATIC_KEYS),
      "kinematics from enough manoeuvres are scored")
check("lc_per_veh_km_complete" not in hf.LC_KINEMATIC_KEYS,
      "the lane-change RATE stays measurable, so producing none is penalised "
      "rather than blanked")

# ------------------------------------------------- 5. headway definitions
print("\n5. headway definitions match highD's")
# two vehicles, same lane: leader 30 m ahead centre-to-centre, both 5 m long,
# so the bumper gap is 25 m; ego at 25 m/s, leader at 20 m/s
road = Road(n_lanes=3, lane_width=3.5, shoulder_right=2.0, shoulder_left=0.8,
            length=1000.0)
d = blank_state(3)
# index 0 is the EV, parked 5 km away exactly as make_highd_like's EV-inert
# control puts it, so it is outside the THW/TTC gates and cannot be anyone's
# effective leader. It is deliberately NOT excluded from being a leader in
# general: in the bluelight scenario the EV is a real vehicle in the stream.
d["x"][:] = [5000.0, 100.0, 130.0]
d["y"][:] = [road.lane_center(1)] * 3
d["vx"][:] = [0.0, 25.0, 20.0]
d["L"][:] = [6.2, 5.0, 5.0]
d["is_ev"][0] = True
st = VehState(**d)


class _H:                                  # minimal History stand-in
    pass


hh = _H()
hh.x = st.x[None, :].copy(); hh.y = st.y[None, :].copy()
hh.vx = st.vx[None, :].copy(); hh.L = st.L.copy(); hh.ev = 0
hh.road = road
thw, ttc = metrics.headway_pools(hh)
check(len(thw) == 1 and abs(thw[0] - 25.0 / 25.0) < 1e-6,
      f"thw = bumper gap / v_ego = 1.0 s (got {thw.tolist()})")
check(len(ttc) == 1 and abs(ttc[0] - 25.0 / 5.0) < 1e-6,
      f"ttc = bumper gap / closing speed = 5.0 s (got {ttc.tolist()})")
hh.vx = np.array([[0.0, 20.0, 25.0]])     # leader faster -> no TTC, as in highD
thw2, ttc2 = metrics.headway_pools(hh)
check(len(ttc2) == 0, "no TTC when the leader is pulling away (highD stores 0)")

# --------------------------------------------------- 6. density stiffening
print("\n6. density-dependent lane discipline")
corr0 = build_corridor(st.x[0], st.vx[0], road.lane_center(1), 30.0, p0)
ay_a = forces.lane_keep(st, road, p0, corr0)
ay_b = forces.lane_keep(st, road, p0, corr0, rho_fac=None)
check(np.array_equal(ay_a, ay_b), "k_rho=0 leaves lane_keep exactly unchanged")
rho = forces.local_density(st, road, p0.copy(R_rho=100.0))
check(abs(rho[1] - 2.0 / 0.2) < 1e-9,
      f"local density counts own-lane vehicles in +/- R_rho (got {rho[1]:.2f} "
      f"veh/km/lane for 2 cars in a 200 m window)")
fac = forces.pin_density_factor(st, road, p0.copy(k_rho=0.5, rho_ref=10.0))
check(np.isclose(fac[1], 1.0), "the factor is 1 at the reference density")
fac2 = forces.pin_density_factor(st, road, p0.copy(k_rho=0.5, rho_ref=5.0))
check(fac2[1] > 1.0, "lane discipline stiffens above the reference density")

print("\n" + ("FAILED: " + "; ".join(FAIL) if FAIL else "ALL CHECKS PASS"))
sys.exit(1 if FAIL else 0)
