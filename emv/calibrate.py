"""Parameter calibration: Latin-hypercube exploration + Nelder-Mead refinement.

No public trajectory dataset exists for fitting (A, B, lambda)-type EV
repulsion parameters (gap 3 of the literature review), so we calibrate
against *behavioural targets* distilled from the EV literature:

  * EV progress: speed ratio -> 1 (pre-clearing ideal, [19, 20])
  * timing realism: drivers start yielding ~110 m ahead of the EV [18, 20]
  * safety: min TTC >= 1.2 s, zero collisions
  * comfort: p95 decel <= 3.2 m/s^2, p95 lateral accel <= 2.5 m/s^2 [10]
  * stability: no lateral oscillation (anti-jitter, [7, 10])
  * low disruption of background traffic [22]

The same loss doubles as the reward-shaping objective suggested by [22].
scipy-free: both samplers are implemented here.
"""
import json
import numpy as np

from .params import Params
from .scenarios import make_overtake, make_jam
from .metrics import evaluate

THETA_SPEC = [
    # name      lo    hi
    ("A_ev",    1.5,  8.0),
    ("B_ev",    8.0, 40.0),
    ("A_c",     1.5,  8.0),
    ("B_c",     0.6,  3.0),
    ("T_react", 4.0, 16.0),
    ("gamma_c", 0.05, 1.2),
]
NAMES = [n for n, _, _ in THETA_SPEC]
LO = np.array([lo for _, lo, _ in THETA_SPEC])
HI = np.array([hi for _, _, hi in THETA_SPEC])


def make_params(theta) -> Params:
    return Params().copy(**dict(zip(NAMES, np.clip(theta, LO, HI))), dt=0.06)


def _run_pair(theta, seed):
    p = make_params(theta)
    sim_o = make_overtake(seed=seed, p=p, density=22.0, road_len=1500.0)
    h_o = sim_o.run(60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
    sim_j = make_jam(seed=seed + 100, p=p, road_len=520.0)
    h_j = sim_j.run(55.0, rec_dt=0.12, stop_when_ev_x=480.0)
    return evaluate(h_o), evaluate(h_j)


def loss(theta, seeds=(11,)) -> tuple[float, dict]:
    parts = dict(progress=0.0, timing=0.0, safety=0.0, comfort=0.0,
                 stability=0.0, disruption=0.0)
    for seed in seeds:
        mo, mj = _run_pair(theta, seed)
        parts["progress"] += 12.0 * (1.0 - mo["ev_speed_ratio"]) ** 2
        parts["progress"] += 8.0 * (1.0 - mj["ev_speed_ratio"]) ** 2
        if np.isfinite(mo["react_dist_mean"]):
            parts["timing"] += 0.6 * ((mo["react_dist_mean"] - 110.0) / 60.0) ** 2
        else:
            parts["timing"] += 2.0
        for m in (mo, mj):
            ttc = m["min_ttc"] if np.isfinite(m["min_ttc"]) else 10.0
            parts["safety"] += 4.0 * max(0.0, 1.2 - ttc) ** 2
            parts["safety"] += min(5.0 * m["collisions"], 50.0)
            parts["comfort"] += 1.5 * max(0.0, m["p95_decel"] - 3.2) ** 2
            parts["comfort"] += 1.0 * max(0.0, m["p95_alat"] - 2.5) ** 2
            parts["stability"] += 0.4 * m["oscillation"]
        parts["disruption"] += 0.5 * mo["disruption"] ** 2
    k = float(len(seeds))
    parts = {n: v / k for n, v in parts.items()}
    return float(sum(parts.values())), parts


# ----------------------------------------------------------------------
def latin_hypercube(n, rng):
    dims = LO.size
    u = (rng.permuted(np.tile(np.arange(n), (dims, 1)), axis=1).T
         + rng.random((n, dims))) / n
    return LO + u * (HI - LO)


def nelder_mead(f, x0, maxiter=60, scale=0.18):
    """Bounded Nelder-Mead (reflection/expansion/contraction/shrink)."""
    d = x0.size
    simplex = [np.clip(x0, LO, HI)]
    for i in range(d):
        x = x0.copy()
        x[i] = np.clip(x[i] + scale * (HI[i] - LO[i]), LO[i], HI[i])
        simplex.append(x)
    simplex = np.array(simplex)
    fv = np.array([f(x) for x in simplex])
    trace = [float(fv.min())]
    for _ in range(maxiter):
        o = np.argsort(fv)
        simplex, fv = simplex[o], fv[o]
        c = simplex[:-1].mean(axis=0)
        xr = np.clip(c + (c - simplex[-1]), LO, HI)
        fr = f(xr)
        if fr < fv[0]:
            xe = np.clip(c + 2.0 * (c - simplex[-1]), LO, HI)
            fe = f(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        else:
            xc = np.clip(c + 0.5 * (simplex[-1] - c), LO, HI)
            fc = f(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fv[1:] = [f(x) for x in simplex[1:]]
        trace.append(float(fv.min()))
    o = np.argsort(fv)
    return simplex[o][0], fv[o][0], trace


def run_calibration(n_lhs=40, n_nm=60, seeds=(11,), out_path="out/calibration.json",
                    verbose=True):
    rng = np.random.default_rng(2026)
    X = latin_hypercube(n_lhs, rng)
    X = np.vstack([X, [Params().__getattribute__(n) for n in NAMES]])  # incl. defaults
    evals = []

    def f(theta):
        val, _ = loss(theta, seeds)
        evals.append((list(map(float, theta)), val))
        return val

    fv = []
    for i, x in enumerate(X):
        fv.append(f(x))
        if verbose and (i + 1) % 10 == 0:
            print(f"  LHS {i+1}/{len(X)}  best so far {min(fv):.3f}", flush=True)
    best0 = X[int(np.argmin(fv))]

    theta, fbest, trace = nelder_mead(f, best0, maxiter=n_nm)
    _, parts = loss(theta, seeds)
    result = dict(
        theta=dict(zip(NAMES, map(float, theta))),
        loss=float(fbest), parts=parts,
        loss_default=float(fv[-1]),          # defaults appended last in X
        n_evals=len(evals), nm_trace=trace,
        evals=[dict(zip(NAMES + ["loss"], t + [v])) for t, v in evals],
    )
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    if verbose:
        print("calibrated:", {k: round(v, 3) for k, v in result["theta"].items()})
        print(f"loss {fbest:.3f} (default {result['loss_default']:.3f})")
    return result
