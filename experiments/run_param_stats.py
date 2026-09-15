"""Parameter-sensitivity and behavioural-realism study.

Four designs, all on the same evaluation kernel (one overtake + one jam run per
(parameter vector, seed), the pair used by emv.calibrate so results are
comparable with the calibration record):

  A  GSA      Latin-hypercube sample over the 8 decisive parameters, analysed
               with standardized regression coefficients (SRC, linear effect
               with sign) and Spearman rank correlation (monotone effect,
               robust to saturation). Common random numbers across samples:
               the same seeds everywhere, so differences between samples are
               parameter effects, not seed noise. R2 of the linear surrogate is
               reported per metric - SRC is only interpretable where it is high.
  B  OAT      One-at-a-time response curves around the calibrated point, with
               percentile bootstrap CIs over seeds. Shows the *shape* (threshold,
               saturating, monotone) that a global index cannot.
  C  surface  A_c x a_pin grid: the depinning interaction the theory predicts
               (escape iff A_c*u > a_pin*(1 - relief*u); cleared half-width
               d* = w_need + B_c ln(A_c/a_pin_eff), docs/FRAMEWORK.md sec. 4).
  D  refer.   Calibrated / default / game presets plus the no-yielding control,
               many seeds, for the distributional comparison against dashcam
               ground truth and naturalistic-driving literature.

    python experiments/run_param_stats.py            # full (~35 min)
    python experiments/run_param_stats.py --quick    # ~1 min smoke design
    python experiments/run_param_stats.py --only gsa oat

Outputs out/param_stats.json (design + analysis) and out/param_stats.npz (raw
matrices, so figures can be re-made without re-running).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.metrics import behaviour_stats, evaluate
from emv.params import Params, tuned_params
from emv.runlog import RunLogger
from emv.scenarios import game_params, make_jam, make_overtake

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

# (attribute, lo, hi, short label) - the eight parameters the live lab exposes,
# over the ranges the calibrator searches (emv/calibrate.py THETA_SPEC) widened
# to the physically sensible span for a_pin / R_front / p_noncomply.
PARAM_SPEC = [
    ("A_c",         0.5,   8.0, "$A_c$"),
    ("B_c",         0.3,   3.0, "$B_c$"),
    ("a_pin",       0.2,   3.5, "$a_{pin}$"),
    ("A_ev",        0.5,   8.0, "$A_{ev}$"),
    ("B_ev",        8.0,  40.0, "$B_{ev}$"),
    ("T_react",     2.0,  18.0, "$T_{react}$"),
    ("R_front",    40.0, 300.0, "$R_{front}$"),
    ("p_noncomply", 0.0,  0.40, "$p_{nc}$"),
]
PNAMES = [s[0] for s in PARAM_SPEC]
PLO = np.array([s[1] for s in PARAM_SPEC])
PHI = np.array([s[2] for s in PARAM_SPEC])

#: metrics kept per run, in report order. 'o_' = overtake, 'j_' = jam.
METRIC_SPEC = [
    ("ev_speed_ratio", "EV speed / desired", "higher"),
    ("ev_med_speed",   "EV median speed",    "higher"),
    ("clearance_mean", "corridor clearance", "higher"),
    ("min_ttc",        "min TTC",            "higher"),
    ("react_dist_mean", "yield onset dist.", "-"),
    ("lc_per_veh_km",  "lane changes / veh-km", "-"),
    ("lc_dur_med",     "lane-change duration", "-"),
    ("lc_peak_vy_med", "peak lateral speed", "-"),
    ("yield_peak_decel_med", "peak decel (yielding)", "lower"),
    ("yield_peak_alat_med",  "peak lat. accel (yielding)", "lower"),
    ("bg_med_speed",   "traffic median speed", "higher"),
    ("disruption",     "traffic disruption", "lower"),
]
MNAMES = [s[0] for s in METRIC_SPEC]


# ----------------------------------------------------------------- kernel
def run_one(p: Params, seed: int, scenario: str) -> dict:
    """One scenario run -> flat scalar metrics (evaluate + behaviour_stats)."""
    if scenario == "overtake":
        sim = make_overtake(seed=seed, p=p, density=22.0, road_len=1500.0)
        h = sim.run(60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
    else:
        sim = make_jam(seed=seed + 100, p=p, road_len=520.0)
        h = sim.run(55.0, rec_dt=0.12, stop_when_ev_x=480.0)
    m = {k: v for k, v in evaluate(h).items() if isinstance(v, (int, float))}
    m.update(behaviour_stats(h))
    return m


def eval_theta(theta, seeds, scenarios=("overtake", "jam"), base=None) -> dict:
    """Seed-averaged metrics for one parameter vector, per scenario."""
    p0 = (base or tuned_params()).copy(dt=0.06)
    p = p0.copy(**{k: float(v) for k, v in zip(PNAMES, theta)})
    out = {}
    for sc in scenarios:
        runs = [run_one(p, s, sc) for s in seeds]
        pre = "o_" if sc == "overtake" else "j_"
        for k in MNAMES:
            out[pre + k] = nanmean([r.get(k, np.nan) for r in runs])
        out[pre + "collisions"] = float(np.mean([r["collisions"] for r in runs]))
    return out


def theta_of(p: Params) -> np.ndarray:
    return np.array([getattr(p, k) for k in PNAMES], float)


# ----------------------------------------------------------------- designs
def latin_hypercube(n, d, rng):
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    return np.stack([rng.permutation(cut[:, j]) for j in range(d)], axis=1)


def study_gsa(n, seeds, log):
    rng = np.random.default_rng(20260723)
    U = latin_hypercube(n, len(PNAMES), rng)
    X = PLO + U * (PHI - PLO)
    Y = []
    for i, theta in enumerate(X):
        Y.append(eval_theta(theta, seeds))
        log(f"  gsa {i + 1}/{n}")
    keys = sorted(Y[0])
    return dict(X=X, Y=np.array([[y[k] for k in keys] for y in Y]), keys=keys)


def study_oat(levels, seeds, log):
    base = theta_of(tuned_params())
    curves = {}
    for j, (name, lo, hi, _lab) in enumerate(PARAM_SPEC):
        grid = np.linspace(lo, hi, levels)
        per_seed = []
        for v in grid:
            theta = base.copy()
            theta[j] = v
            rows = []
            for s in seeds:                      # keep seeds separate for the CI
                rows.append(eval_theta(theta, [s]))
            per_seed.append(rows)
            log(f"  oat {name}={v:.3g}")
        keys = sorted(per_seed[0][0])
        curves[name] = dict(
            grid=grid,
            Y=np.array([[[r[k] for k in keys] for r in lvl] for lvl in per_seed]),
            keys=keys, base=float(base[j]))
    return curves


def study_surface(n_grid, seeds, log):
    base = tuned_params()
    ac = np.linspace(0.4, 6.0, n_grid)
    ap = np.linspace(0.2, 3.2, n_grid)
    out = {}
    for sc in ("overtake", "jam"):
        Z = np.full((n_grid, n_grid, 3), np.nan)
        for ia, a in enumerate(ac):
            for ip, q in enumerate(ap):
                p = base.copy(A_c=float(a), a_pin=float(q), dt=0.06)
                rows = [run_one(p, s, sc) for s in seeds]
                Z[ia, ip] = [np.mean([r["ev_speed_ratio"] for r in rows]),
                             np.mean([r["clearance_mean"] for r in rows]),
                             np.mean([r["lc_per_veh_km"] for r in rows])]
            log(f"  surface {sc} A_c={a:.2f}")
        out[sc] = Z
    return dict(A_c=ac, a_pin=ap, **out)


def study_reference(seeds, log):
    """Distributional reference: presets + the no-yielding control."""
    cfg = {"tuned": tuned_params(), "default": Params(), "game": game_params()}
    out = {}
    for name, p in cfg.items():
        rows = {sc: [] for sc in ("overtake", "jam")}
        for sc in rows:
            for s in seeds:
                rows[sc].append(run_one(p.copy(dt=0.06), s, sc))
            log(f"  reference {name}/{sc}")
        out[name] = rows
    # control: identical scenario, drivers never become aware of the EV
    rows = {sc: [] for sc in ("overtake", "jam")}
    p = tuned_params().copy(dt=0.06)
    for s in seeds:
        sim = make_overtake(seed=s, p=p, density=22.0, road_len=1500.0,
                            yielding=False)
        h = sim.run(60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
        rows["overtake"].append({**{k: v for k, v in evaluate(h).items()
                                    if isinstance(v, (int, float))},
                                 **behaviour_stats(h)})
        sim = make_jam(seed=s + 100, p=p, road_len=520.0, yielding=False)
        h = sim.run(55.0, rec_dt=0.12, stop_when_ev_x=480.0)
        rows["jam"].append({**{k: v for k, v in evaluate(h).items()
                               if isinstance(v, (int, float))},
                            **behaviour_stats(h)})
        log(f"  reference no-yield seed {s}")
    out["no_yield"] = rows
    return out


# ---------------------------------------------------------------- analysis
def _rank(a):
    """Average ranks (ties shared), so Spearman is exact on duplicates."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    rx, ry = _rank(x[ok]), _rank(y[ok])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def src(X, y):
    """Standardized regression coefficients + R2 of the linear surrogate."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < X.shape[1] + 3:
        return np.full(X.shape[1], np.nan), np.nan
    Xs = (X[ok] - X[ok].mean(0)) / (X[ok].std(0) + 1e-12)
    ys = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-12)
    beta, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(ok.sum())], ys, rcond=None)
    r2 = 1.0 - np.sum((ys - np.c_[Xs, np.ones(ok.sum())] @ beta) ** 2) / max(
        np.sum(ys ** 2), 1e-12)
    return beta[:-1], float(r2)


def nanmean(a):
    """np.nanmean without the all-NaN warning (the no-yield control legitimately
    has no lane changes, so its lane-change metrics are empty)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def boot_ci(a, n=2000, q=(2.5, 97.5), rng=None):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(7)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, q))


def analyse(gsa):
    X, Y, keys = gsa["X"], gsa["Y"], gsa["keys"]
    rho = np.full((len(PNAMES), len(keys)), np.nan)
    beta = np.full((len(PNAMES), len(keys)), np.nan)
    r2 = np.full(len(keys), np.nan)
    for c in range(len(keys)):
        for j in range(len(PNAMES)):
            rho[j, c] = spearman(X[:, j], Y[:, c])
        beta[:, c], r2[c] = src(X, Y[:, c])
    return dict(rho=rho, beta=beta, r2=r2, keys=keys)


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny design (~1 min)")
    ap.add_argument("--only", nargs="+", default=["gsa", "oat", "surface", "reference"],
                    choices=["gsa", "oat", "surface", "reference"])
    ap.add_argument("--n-lhs", type=int, default=128)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--grid", type=int, default=9)
    args = ap.parse_args()

    n_lhs, levels, grid = args.n_lhs, args.levels, args.grid
    seeds_gsa, seeds_oat, seeds_surf, seeds_ref = (11, 12), (11, 12, 13, 14), (11, 12), tuple(range(11, 19))
    if args.quick:
        n_lhs, levels, grid = 8, 3, 3
        seeds_gsa, seeds_oat, seeds_surf, seeds_ref = (11,), (11, 12), (11,), (11, 12)

    t0 = time.time()
    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    raw, summary = {}, dict(
        design=dict(n_lhs=n_lhs, levels=levels, grid=grid, quick=args.quick,
                    seeds_gsa=list(seeds_gsa), seeds_oat=list(seeds_oat),
                    seeds_surface=list(seeds_surf), seeds_reference=list(seeds_ref),
                    params={n: [lo, hi] for n, lo, hi, _ in PARAM_SPEC},
                    metrics=MNAMES))

    with RunLogger("param_stats", note=f"n_lhs={n_lhs} levels={levels} grid={grid}"
                                       f"{' quick' if args.quick else ''}") as rl:
        if "gsa" in args.only:
            log(f"study A: GSA, {n_lhs} samples x {len(seeds_gsa)} seeds")
            gsa = study_gsa(n_lhs, seeds_gsa, log)
            an = analyse(gsa)
            raw.update(gsa_X=gsa["X"], gsa_Y=gsa["Y"], gsa_rho=an["rho"],
                       gsa_beta=an["beta"], gsa_r2=an["r2"])
            summary["gsa"] = dict(
                keys=an["keys"], params=PNAMES,
                r2={k: float(v) for k, v in zip(an["keys"], an["r2"])},
                rho={k: {p: float(an["rho"][j, c]) for j, p in enumerate(PNAMES)}
                     for c, k in enumerate(an["keys"])})

        if "oat" in args.only:
            log(f"study B: OAT, {levels} levels x {len(seeds_oat)} seeds")
            oat = study_oat(levels, seeds_oat, log)
            for name, d in oat.items():
                raw[f"oat_{name}_grid"] = d["grid"]
                raw[f"oat_{name}_Y"] = d["Y"]
            summary["oat"] = dict(keys=oat[PNAMES[0]]["keys"],
                                  base={n: oat[n]["base"] for n in PNAMES})

        if "surface" in args.only:
            log(f"study C: A_c x a_pin surface, {grid}x{grid} x {len(seeds_surf)} seeds")
            sf = study_surface(grid, seeds_surf, log)
            raw.update(surf_A_c=sf["A_c"], surf_a_pin=sf["a_pin"],
                       surf_overtake=sf["overtake"], surf_jam=sf["jam"])
            summary["surface"] = dict(A_c=[float(sf["A_c"][0]), float(sf["A_c"][-1])],
                                      a_pin=[float(sf["a_pin"][0]), float(sf["a_pin"][-1])],
                                      layers=["ev_speed_ratio", "clearance_mean",
                                              "lc_per_veh_km"])

        if "reference" in args.only:
            log(f"study D: reference presets x {len(seeds_ref)} seeds")
            ref = study_reference(seeds_ref, log)
            summary["reference"] = {
                cfg: {sc: {k: [nanmean([r.get(k, np.nan) for r in rows]),
                               *boot_ci([r.get(k, np.nan) for r in rows])]
                           for k in MNAMES}
                      for sc, rows in scs.items()}
                for cfg, scs in ref.items()}
            summary["reference_raw"] = {
                cfg: {sc: [{k: float(r.get(k, np.nan)) for k in MNAMES + ["collisions"]}
                           for r in rows] for sc, rows in scs.items()}
                for cfg, scs in ref.items()}

        with open(os.path.join(OUT, "param_stats.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        np.savez_compressed(os.path.join(OUT, "param_stats.npz"), **raw)
        log("wrote out/param_stats.json + out/param_stats.npz")
        rl.finish(metrics=dict(n_lhs=n_lhs, levels=levels, grid=grid,
                               studies=",".join(args.only)),
                  outputs=["out/param_stats.json", "out/param_stats.npz"])
