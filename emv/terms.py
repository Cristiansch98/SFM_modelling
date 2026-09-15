"""Per-term influence analysis: one registry, three instruments.

The model is a sum of force terms (`emv/forces.py:total`). This module is the
single place that knows what those terms are, so that the three ways of asking
"how much does this term matter?" cannot drift apart, and so a term added to the
model but forgotten here is a test failure rather than a silent omission
(`tests/test_terms.py` compares `TERMS` against the keys of
`forces.total(diag=True)`).

The three instruments
--------------------
1. **Force budget** - `budget_from_history`. What each term actually contributes
   to the accelerations that were integrated, split by driver awareness state.
   Descriptive: it measures effort, not importance. A term can dominate the
   budget and change no outcome (the lane potential mostly cancels itself), or be
   a small fraction of it and decide everything (the corridor term).
2. **Ablation** - `ABLATIONS`. Set one term's amplitude to zero, re-score the
   whole objective, and report the change. Prescriptive: it measures what the
   term buys against real data. This is the instrument that answers the
   question, and the other two exist to explain its answers.
3. **Sensitivity** - the parameter bounds in `emv/highd_fit.py` plus the
   LHS/Spearman/SRC machinery of `experiments/run_param_stats.py`. Continuous
   rather than on/off, and the only one that sees interactions.

Terms are grouped by calibration stage: `block="N"` terms describe ordinary
traffic and are fitted against highD with no emergency vehicle present;
`block="E"` terms are the emergency-vehicle extension, fitted afterwards with
every N parameter frozen.
"""
from __future__ import annotations

import numpy as np

#: name -> (diag key, axis, block, one-line description).
#: `axis` is the component the term acts on: "x", "y" or "both".
TERMS: dict = {
    "drive": ("drive", "both", "N",
              "relaxation to the desired speed, (v0 - v)/tau"),
    "sfm": ("sfm", "both", "N",
            "elliptic car-car repulsion incl. the near-field body term"),
    "idm": ("idm", "x", "N",
            "IDM car-following demand, applied as an upper bound not a summand"),
    "lane": ("lane", "y", "N",
             "washboard lane potential, road-edge walls and lateral damping"),
    "lc": ("lc", "y", "N",
           "discretionary lane-change incentive (overtake + keep-right)"),
    "ev": ("ev", "both", "E",
           "anisotropic point repulsion from the emergency vehicle"),
    "corr": ("corr", "both", "E",
             "repulsion from the emergency vehicle's predicted corridor"),
}

#: Ablation name -> (parameter overrides, term(s) removed, stage, what it tests).
#: Every entry must switch a term off *through a parameter*, never by editing the
#: force composition - so an ablation is a point in the same parameter space the
#: calibration searches, and "the term is off" means exactly "its amplitude is 0".
#:
#: `stage` says which objective can see the term, and is stated per ablation
#: rather than inferred from the term, because two entries would be classified
#: wrongly by inference: `urgency_pin_relief` and `aware_amax_boost` are
#: *host-traffic* parameters that only ever act on a driver who has perceived a
#: siren, so they are invisible to the EV-free objective despite belonging to the
#: lane-keeping and car-following blocks.
ABLATIONS: dict = {
    "no_sfm_far": (dict(A_v=0.0), "sfm", "N",
                   "long-range car-car repulsion; the near-field body term and "
                   "the IDM bound still prevent overlap"),
    "no_sfm_near": (dict(A_near=0.0), "sfm", "N",
                    "near-field body term: does packing stay a force balance?"),
    "no_idm_bound": (dict(idm_bound=False), "idm", "N",
                     "the longitudinal safety layer. Expected to produce "
                     "collisions - it is the term that makes them impossible"),
    "no_pin": (dict(a_pin=0.0), "lane", "N",
               "lane discipline: without it there are no lanes, only a road"),
    "no_offset": (dict(sigma_off=0.0), "lane", "N",
                  "per-driver lateral offset (lane-keeping precision)"),
    "no_het": (dict(het_lat=0.0), "lane", "N",
               "driver-to-driver variety in the lateral block (distribution "
               "widths, not medians)"),
    "no_density_pin": (dict(k_rho=0.0), "lane", "N",
                       "density stiffening of lane discipline: the term added "
                       "to stop the free-flow fit degrading the congested one"),
    "no_overtake": (dict(A_pass=0.0), "lc", "N",
                    "the overtaking incentive"),
    "no_keep_right": (dict(A_keep_right=0.0), "lc", "N",
                      "keep-right pressure from a faster follower"),
    "no_lc": (dict(A_pass=0.0, A_keep_right=0.0), "lc", "N",
              "all discretionary lane changing: the state the model was in "
              "before this extension, where every lane change was EV-induced"),
    "no_ev_field": (dict(A_ev=0.0), "ev", "E",
                    "point repulsion from the emergency vehicle itself"),
    "no_corridor_lat": (dict(A_c=0.0), "corr", "E",
                        "the lateral corridor push - the term that forms the "
                        "rescue lane"),
    "no_corridor_lon": (dict(gamma_c=0.0), "corr", "E",
                        "the 'slow down while pulling over' component"),
    "no_urgency_relief": (dict(urgency_pin_relief=0.0), "lane", "E",
                          "urgency relaxing lane discipline, i.e. raising the "
                          "depinning threshold a yielding driver must cross"),
    "no_aware_boost": (dict(aware_amax_boost=1.0), "idm", "E",
                       "yielding drivers accepting stronger acceleration"),
    "no_ev_response": (dict(A_ev=0.0, A_c=0.0, gamma_c=0.0), "ev+corr", "E",
                       "every emergency-vehicle force at once. Note this does "
                       "NOT recover the normal model: perceiving a siren also "
                       "engages aware_amax_boost and urgency_pin_relief, so the "
                       "extension has five channels and two of them retune "
                       "host-traffic parameters (asserted in tests/test_terms.py)"),
    "no_ev_at_all": (dict(A_ev=0.0, A_c=0.0, gamma_c=0.0, aware_amax_boost=1.0,
                          urgency_pin_relief=0.0), "ev+corr", "E",
                     "all five channels: this one does reproduce the normal "
                     "model bit for bit, which is the nesting invariant"),
}

#: Ablations each stage's objective can actually see. An "N" ablation scored
#: against the emergency-vehicle objective is meaningful; an "E" ablation scored
#: against the EV-free one is not - it provably changes nothing, because there is
#: no emergency vehicle in that scenario.
N_ABLATIONS = tuple(k for k, v in ABLATIONS.items() if v[2] == "N")
E_ABLATIONS = tuple(k for k, v in ABLATIONS.items() if v[2] == "E")


def ablate(p, name: str):
    """Params with one term switched off. Raises on an unknown ablation."""
    if name in ("none", "", None):
        return p
    return p.copy(**ABLATIONS[name][0])


# ---------------------------------------------------------------------------
# instrument 1: force budget
# ---------------------------------------------------------------------------
def accumulate(acc: dict | None, dg: dict, st, dt: float,
               b_emerg: float = 8.0) -> dict:
    """Add one step's per-term contribution to a budget accumulator.

    Accumulates the mean |a| per term and the *impulse* (signed, integrated over
    time) separately, because they answer different questions: |a| says how hard
    a term is working, the signed impulse says whether that work went anywhere.
    The lane potential is the reason both are needed - it is one of the largest
    |a| contributors and its impulse is near zero, because it spends its effort
    holding a car where it already is.

    Split by awareness so that a term's effect on *yielding* drivers can be read
    off separately from its effect on the traffic at large; with no emergency
    vehicle every sample lands in the "unaware" bucket.
    """
    from .state import YIELDING, HOLD
    if acc is None:
        acc = dict(n=0, dt=float(dt),
                   abs_x={}, abs_y={}, imp_x={}, imp_y={},
                   abs_x_yield={}, abs_y_yield={}, n_yield=0,
                   idm_bind_n=0, abs_social_x=0.0, abs_eff_x=0.0,
                   idm_added=0.0)
    other = ~st.is_ev
    yld = other & ((st.aware == YIELDING) | (st.aware == HOLD))
    acc["n"] += int(other.sum())
    acc["n_yield"] += int(yld.sum())

    # The IDM demand is a *bound*, not a summand, so it has no share of the
    # budget - and the plain longitudinal shares are ~99 % `drive` at motorway
    # speeds, which says nothing. What answers "how much does the safety layer
    # matter" is how often it is the binding constraint and how much braking it
    # *imposes* when it binds.
    #
    # Note the direction, which is easy to get backwards: the bound does not
    # shrink the social force, it REPLACES it with a more negative acceleration.
    # So the quantity of interest is `social - applied >= 0`, the extra braking
    # the safety layer adds, measured after the same `-b_emerg` clip the
    # integrator applies - the raw IDM demand is unbounded (it reached ~5 m/s2
    # here) and reporting it would overstate what was actually integrated.
    social_x = dg["sfm"][0] + dg["ev"][0] + dg["corr"][0]
    applied = np.maximum(np.minimum(social_x, dg["idm"]), -b_emerg)
    binding = dg["idm"][other] < social_x[other]
    acc["idm_bind_n"] += int(binding.sum())
    acc["abs_social_x"] += float(np.abs(social_x[other]).sum())
    acc["abs_eff_x"] += float(np.abs(applied[other]).sum())
    acc["idm_added"] += float((social_x[other] - applied[other]).sum())
    for name, (key, axis, _, _) in TERMS.items():
        if key == "idm":                       # a bound, not a summand
            continue
        comps = dict(zip(("x", "y"), dg[key]))
        for tag in ("x", "y") if axis == "both" else (axis,):
            a = np.asarray(comps[tag], float)
            acc[f"abs_{tag}"][name] = (acc[f"abs_{tag}"].get(name, 0.0)
                                       + float(np.abs(a[other]).sum()))
            acc[f"imp_{tag}"][name] = (acc[f"imp_{tag}"].get(name, 0.0)
                                       + float(a[other].sum()) * dt)
            acc[f"abs_{tag}_yield"][name] = (acc[f"abs_{tag}_yield"].get(name, 0.0)
                                             + float(np.abs(a[yld]).sum()))
    return acc


def budget_report(acc: dict) -> dict:
    """Mean |a| per term and its share of the total, per axis."""
    if not acc or not acc["n"]:
        return {}
    out = {}
    for tag in ("x", "y"):
        for scope, n in (("", acc["n"]), ("_yield", acc["n_yield"])):
            d = acc[f"abs_{tag}{scope}"]
            tot = sum(d.values()) or 1.0
            out[f"mean_abs_{tag}{scope}"] = {k: round(v / max(n, 1), 5)
                                             for k, v in d.items()}
            out[f"share_{tag}{scope}"] = {k: round(v / tot, 4) for k, v in d.items()}
        out[f"impulse_{tag}"] = {k: round(v / max(acc["n"], 1), 5)
                                 for k, v in acc[f"imp_{tag}"].items()}
    # longitudinal *interaction* shares (drive excluded): at motorway speeds the
    # drive term is ~99 % of the raw longitudinal budget, so the composition of
    # what is left is the only informative part
    inter = {k: v for k, v in acc["abs_x"].items() if k != "drive"}
    tot_i = sum(inter.values()) or 1.0
    out["share_x_interaction"] = {k: round(v / tot_i, 4) for k, v in inter.items()}
    n = max(acc["n"], 1)
    out["idm"] = dict(
        binding_share=round(acc["idm_bind_n"] / n, 4),
        mean_abs_social=round(acc["abs_social_x"] / n, 5),
        mean_abs_applied=round(acc["abs_eff_x"] / n, 5),
        added_braking=round(acc["idm_added"] / n, 5),
        added_braking_when_binding=round(
            acc["idm_added"] / max(acc["idm_bind_n"], 1), 5))
    out["n_samples"], out["n_yield_samples"] = acc["n"], acc["n_yield"]
    return out


def budget_run(sim, T: float, rec_dt: float = 0.12,
               stop_when_ev_x: float | None = None) -> tuple:
    """Run `sim` accumulating the per-term budget. Returns (History, report).

    Uses `Sim.run`'s `on_diag` hook, which is opt-in precisely so the diagnostic
    (an extra per-term bookkeeping pass every step) never touches the hot path
    that the calibration objectives run thousands of times.
    """
    acc = {}

    def hook(dg, st, dt):
        acc["a"] = accumulate(acc.get("a"), dg, st, dt,
                              b_emerg=sim.p.b_emerg)

    hist = sim.run(T, rec_dt=rec_dt, stop_when_ev_x=stop_when_ev_x,
                   on_diag=hook)
    return hist, budget_report(acc.get("a"))


def attribution(base_obs: dict, abl_obs: dict, keys, scales: dict) -> dict:
    """Per-observable movement caused by an ablation, in scale units.

    Reported in the objective's own normalisation (`highd_fit.OBS_SPEC`'s scale),
    so "1.0" means the ablation moved that observable by one unit of the natural
    between-carriageway variation of real traffic.
    """
    out = {}
    for k in keys:
        a, b = base_obs.get(k), abl_obs.get(k)
        s = scales.get(k) or 1.0
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            out[k] = None
        else:
            out[k] = round(float((b - a) / s), 4)
    return out
