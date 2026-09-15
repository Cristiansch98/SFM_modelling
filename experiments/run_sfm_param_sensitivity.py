"""Performance and parameter-sensitivity study of the two-stage SFM EV-yielding
model (base = emv.params.bluelight_params(), the calibration the ITEC full paper
describes).

Four designs, one evaluation kernel (one overtake + one jam run per parameter
vector and seed -- the pair emv.calibrate uses, so numbers line up with the
calibration record):

  OAT       one-at-a-time response curves for EVERY tunable parameter, swept
            across its calibration search bound, seeds kept separate for a
            percentile-bootstrap CI. Gives the shape (threshold / saturating /
            monotone) and the swing (max-min of the seed-mean) that a global
            index cannot.
  GSA       Latin-hypercube sample over all tunable parameters at once,
            analysed with Spearman rank correlation (monotone effect, robust to
            saturation) and standardized regression coefficients (signed linear
            effect); R2 of the linear surrogate reported per metric. Common
            random numbers: the same seeds in every sample, so sample-to-sample
            differences are parameter effects.
  SURFACE   A_c x a_pin grid: the depinning interaction the theory predicts
            (escape iff A_c u > a_pin(1 - eps u); d* = w_need + B_c ln(A_c /
            a_pin_eff)).
  REF       calibrated (blue-light) vs dataclass-default vs a no-yielding
            control, many seeds -- the performance baseline.

    .venv/bin/python experiments/run_sfm_param_sensitivity.py            # ~45 min
    .venv/bin/python experiments/run_sfm_param_sensitivity.py --quick    # ~2 min
    .venv/bin/python experiments/run_sfm_param_sensitivity.py --only oat gsa

Writes out/sfm_param_sensitivity.json (design + analysis) and
out/sfm_param_sensitivity.npz (raw matrices, so figures re-make without a
re-run). Self-logs kind='param_sensitivity'.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import Params, bluelight_params
from emv.runlog import RunLogger
from emv.scenarios import make_jam, make_overtake

# reuse the kernel + analysis of the existing param study
from run_param_stats import (MNAMES, boot_ci, latin_hypercube, nanmean, run_one,
                             spearman, src)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

# ---- every tunable parameter, with its calibration search bound -------------
# stage-1 host block: emv/highd_fit.py SPEC_LON + SPEC_LAT + SPEC_LC
SPEC_STAGE1 = [
    ("tau",          0.30,  1.60, "stage1-lon"),
    ("T_hw_lo",      0.40,  1.40, "stage1-lon"),
    ("T_hw_hi",      1.40,  3.20, "stage1-lon"),
    ("s0",           0.80,  5.00, "stage1-lon"),
    ("A_v",          0.50,  6.00, "stage1-lon"),
    ("Bx_v",         1.50, 12.00, "stage1-lon"),
    ("a_max",        1.00,  4.00, "stage1-lon"),
    ("b_comf",       1.50,  5.00, "stage1-lon"),
    ("a_pin",        0.15,  2.50, "stage1-lat"),
    ("zeta_lat",     0.40,  3.00, "stage1-lat"),
    ("v_lat_max",    0.50,  2.20, "stage1-lat"),
    ("a_lat_max",    0.20,  3.00, "stage1-lat"),
    ("sigma_off",    0.00,  0.60, "stage1-lat"),
    ("het_lat",      0.00,  0.60, "stage1-lat"),
    ("k_rho",        0.00,  2.00, "stage1-lat"),
    ("A_pass",       0.50, 12.00, "stage1-lc"),
    ("A_keep_right", 0.00, 12.00, "stage1-lc"),
    ("T_frust",      0.80,  6.00, "stage1-lc"),
    ("s_veto",       5.00, 60.00, "stage1-lc"),
]
# stage-2 EV block: emv/calibrate.py THETA_SPEC (A_ev,B_ev,A_c,B_c shipped;
# gamma_c,T_react also live in the EV objective)
SPEC_STAGE2 = [
    ("A_ev",         1.50,  8.00, "stage2-ev"),
    ("B_ev",         8.00, 40.00, "stage2-ev"),
    ("A_c",          1.50,  8.00, "stage2-ev"),
    ("B_c",          0.60,  3.00, "stage2-ev"),
    ("gamma_c",      0.05,  1.20, "stage2-ev"),
    ("T_react",      4.00, 16.00, "stage2-ev"),
]
# decisive hand-set perception parameters (fixed in calibration; swept here to
# show what they would cost / buy if moved)
SPEC_PERCEPT = [
    ("p_noncomply",  0.00,  0.40, "handset-perc"),
    ("R_front",     40.00, 300.0, "handset-perc"),
    ("delay_med",    0.40,  2.50, "handset-perc"),
    ("urgency_pin_relief", 0.0, 0.9, "handset-perc"),
]
FULL_SPEC = SPEC_STAGE1 + SPEC_STAGE2 + SPEC_PERCEPT
PNAMES = [s[0] for s in FULL_SPEC]
PLO = np.array([s[1] for s in FULL_SPEC])
PHI = np.array([s[2] for s in FULL_SPEC])
PCLASS = {s[0]: s[3] for s in FULL_SPEC}

#: the metrics whose sign of "good" is unambiguous, used for the tornado / GSA
HEADLINE = ["ev_speed_ratio", "ev_med_speed", "clearance_mean", "min_ttc",
            "react_dist_mean", "disruption", "lc_per_veh_km"]


# ----------------------------------------------------------------- kernel
def base_params(dt=0.06) -> Params:
    return bluelight_params().copy(dt=dt)


def eval_params(p: Params, seeds, scenarios=("overtake", "jam")) -> dict:
    """Seed-averaged scalar metrics for one Params, per scenario ('o_'/'j_')."""
    out = {}
    for sc in scenarios:
        runs = [run_one(p, s if sc == "overtake" else s + 100, sc)
                for s in seeds]
        pre = "o_" if sc == "overtake" else "j_"
        for k in MNAMES:
            out[pre + k] = nanmean([r.get(k, np.nan) for r in runs])
        out[pre + "collisions"] = float(np.mean([r["collisions"] for r in runs]))
    return out


def base_vec(p: Params) -> np.ndarray:
    return np.array([float(getattr(p, k)) for k in PNAMES])


# ----------------------------------------------------------------- designs
def study_oat(levels, seeds, log):
    p0 = base_params()
    b = base_vec(p0)
    curves = {}
    for j, (name, lo, hi, _c) in enumerate(FULL_SPEC):
        grid = np.unique(np.sort(np.append(np.linspace(lo, hi, levels), b[j])))
        per_level = []
        for v in grid:
            rows = []
            for s in seeds:
                p = p0.copy(**{name: float(v)})
                rows.append(eval_params(p, [s]))
            per_level.append(rows)
        keys = sorted(per_level[0][0])
        curves[name] = dict(
            grid=grid, base=float(b[j]), keys=keys,
            Y=np.array([[[r[k] for k in keys] for r in lvl]
                        for lvl in per_level]))
        log(f"  oat {name} ({_c}) done, {len(grid)} levels")
    return curves


def study_gsa(n, seeds, log):
    rng = np.random.default_rng(20260907)
    U = latin_hypercube(n, len(PNAMES), rng)
    X = PLO + U * (PHI - PLO)
    p0 = base_params()
    Y = []
    for i, theta in enumerate(X):
        p = p0.copy(**{k: float(v) for k, v in zip(PNAMES, theta)})
        Y.append(eval_params(p, seeds))
        if (i + 1) % 10 == 0:
            log(f"  gsa {i + 1}/{n}")
    keys = sorted(Y[0])
    Ym = np.array([[y[k] for k in keys] for y in Y])
    rho = np.full((len(PNAMES), len(keys)), np.nan)
    beta = np.full((len(PNAMES), len(keys)), np.nan)
    r2 = np.full(len(keys), np.nan)
    for c in range(len(keys)):
        for j in range(len(PNAMES)):
            rho[j, c] = spearman(X[:, j], Ym[:, c])
        beta[:, c], r2[c] = src(X, Ym[:, c])
    return dict(X=X, Y=Ym, keys=keys, rho=rho, beta=beta, r2=r2)


def study_surface(n_grid, seeds, log):
    p0 = base_params()
    ac = np.linspace(0.4, 6.0, n_grid)
    ap = np.linspace(0.2, 3.2, n_grid)
    out = {}
    for sc in ("overtake", "jam"):
        Z = np.full((n_grid, n_grid, 4), np.nan)
        for ia, a in enumerate(ac):
            for ip, q in enumerate(ap):
                p = p0.copy(A_c=float(a), a_pin=float(q))
                rows = [run_one(p, s if sc == "overtake" else s + 100, sc)
                        for s in seeds]
                Z[ia, ip] = [np.mean([r["ev_speed_ratio"] for r in rows]),
                             np.mean([r["clearance_mean"] for r in rows]),
                             np.mean([r.get("lc_per_veh_km", np.nan)
                                      for r in rows]),
                             np.mean([r["collisions"] for r in rows])]
            log(f"  surface {sc} A_c={a:.2f}")
        out[sc] = Z
    return dict(A_c=ac, a_pin=ap, **out)


def study_reference(seeds, log):
    cfg = {"bluelight": base_params(), "default": Params().copy(dt=0.06)}
    out = {}
    for name, p in cfg.items():
        rows = {sc: [] for sc in ("overtake", "jam")}
        for sc in rows:
            for s in seeds:
                rows[sc].append(run_one(p, s if sc == "overtake" else s + 100,
                                        sc))
            log(f"  reference {name}/{sc}")
        out[name] = rows
    # no-yielding control: same traffic, drivers never register the EV
    p = base_params()
    rows = {sc: [] for sc in ("overtake", "jam")}
    for s in seeds:
        so = make_overtake(seed=s, p=p, density=22.0, road_len=1500.0,
                           yielding=False)
        ho = so.run(60.0, rec_dt=0.12, stop_when_ev_x=1450.0)
        sj = make_jam(seed=s + 100, p=p, road_len=520.0, yielding=False)
        hj = sj.run(55.0, rec_dt=0.12, stop_when_ev_x=480.0)
        from emv.metrics import behaviour_stats, evaluate
        for sc, h in (("overtake", ho), ("jam", hj)):
            rows[sc].append({**{k: v for k, v in evaluate(h).items()
                                if isinstance(v, (int, float))},
                             **behaviour_stats(h)})
        log(f"  reference no_yield seed {s}")
    out["no_yield"] = rows
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+",
                    default=["oat", "gsa", "surface", "ref"],
                    choices=["oat", "gsa", "surface", "ref"])
    ap.add_argument("--n-lhs", type=int, default=180)
    ap.add_argument("--levels", type=int, default=7)
    ap.add_argument("--grid", type=int, default=9)
    a = ap.parse_args()

    n_lhs, levels, grid = a.n_lhs, a.levels, a.grid
    seeds_oat, seeds_gsa, seeds_surf, seeds_ref = (11, 12, 13, 14), (11, 12), \
        (11, 12), tuple(range(11, 21))
    if a.quick:
        n_lhs, levels, grid = 12, 3, 3
        seeds_oat, seeds_gsa, seeds_surf, seeds_ref = (11, 12), (11,), (11,), \
            (11, 12)

    t0 = time.time()
    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    raw = {}
    summary = dict(design=dict(
        base="bluelight_params", quick=a.quick, n_lhs=n_lhs, levels=levels,
        grid=grid, seeds_oat=list(seeds_oat), seeds_gsa=list(seeds_gsa),
        seeds_surface=list(seeds_surf), seeds_reference=list(seeds_ref),
        params={n: [lo, hi, c] for n, lo, hi, c in FULL_SPEC},
        base_values={n: float(getattr(base_params(), n)) for n in PNAMES},
        metrics=MNAMES, headline=HEADLINE))

    with RunLogger("param_sensitivity",
                   note=f"SFM two-stage model; {'quick' if a.quick else 'full'}; "
                        f"n_lhs={n_lhs} levels={levels} grid={grid}; "
                        f"studies={','.join(a.only)}") as rl:
        if "oat" in a.only:
            log(f"OAT: {len(FULL_SPEC)} params x ~{levels} levels x "
                f"{len(seeds_oat)} seeds")
            oat = study_oat(levels, seeds_oat, log)
            for name, d in oat.items():
                raw[f"oat_{name}_grid"] = d["grid"]
                raw[f"oat_{name}_Y"] = d["Y"]
            # swing table: max-min of the seed-mean over the swept range
            swing = {}
            k0 = oat[PNAMES[0]]["keys"]
            for name, d in oat.items():
                ym = np.nanmean(d["Y"], axis=1)              # levels x metrics
                swing[name] = {k: float(np.nanmax(ym[:, i]) - np.nanmin(ym[:, i]))
                               for i, k in enumerate(d["keys"])}
            summary["oat"] = dict(keys=k0,
                                  base={n: oat[n]["base"] for n in PNAMES},
                                  grids={n: [float(x) for x in oat[n]["grid"]]
                                         for n in PNAMES},
                                  swing=swing)

        if "gsa" in a.only:
            log(f"GSA: {n_lhs} LHS samples x {len(seeds_gsa)} seeds over "
                f"{len(PNAMES)} params")
            g = study_gsa(n_lhs, seeds_gsa, log)
            raw.update(gsa_X=g["X"], gsa_Y=g["Y"], gsa_rho=g["rho"],
                       gsa_beta=g["beta"], gsa_r2=g["r2"])
            summary["gsa"] = dict(
                keys=g["keys"], params=PNAMES,
                r2={k: float(v) for k, v in zip(g["keys"], g["r2"])},
                rho={k: {p: float(g["rho"][j, c]) for j, p in enumerate(PNAMES)}
                     for c, k in enumerate(g["keys"])},
                beta={k: {p: float(g["beta"][j, c]) for j, p in enumerate(PNAMES)}
                      for c, k in enumerate(g["keys"])})

        if "surface" in a.only:
            log(f"SURFACE: {grid}x{grid} A_c x a_pin x {len(seeds_surf)} seeds")
            sf = study_surface(grid, seeds_surf, log)
            raw.update(surf_A_c=sf["A_c"], surf_a_pin=sf["a_pin"],
                       surf_overtake=sf["overtake"], surf_jam=sf["jam"])
            summary["surface"] = dict(
                A_c=[float(sf["A_c"][0]), float(sf["A_c"][-1])],
                a_pin=[float(sf["a_pin"][0]), float(sf["a_pin"][-1])],
                layers=["ev_speed_ratio", "clearance_mean", "lc_per_veh_km",
                        "collisions"])

        if "ref" in a.only:
            log(f"REF: presets + no-yield control x {len(seeds_ref)} seeds")
            ref = study_reference(seeds_ref, log)
            mkeys = MNAMES + ["collisions"]
            summary["reference"] = {
                cfg: {sc: {k: [nanmean([r.get(k, np.nan) for r in rows]),
                               *boot_ci([r.get(k, np.nan) for r in rows])]
                           for k in mkeys}
                      for sc, rows in scs.items()}
                for cfg, scs in ref.items()}
            summary["reference_raw"] = {
                cfg: {sc: [{k: float(r.get(k, np.nan)) for k in mkeys}
                           for r in rows] for sc, rows in scs.items()}
                for cfg, scs in ref.items()}

        with open(os.path.join(OUT, "sfm_param_sensitivity.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        np.savez_compressed(os.path.join(OUT, "sfm_param_sensitivity.npz"), **raw)
        log("wrote out/sfm_param_sensitivity.json + .npz")
        rl.finish(metrics=dict(n_lhs=n_lhs, levels=levels, grid=grid,
                               n_params=len(PNAMES),
                               studies=",".join(a.only)),
                  outputs=["out/sfm_param_sensitivity.json",
                           "out/sfm_param_sensitivity.npz"])


if __name__ == "__main__":
    main()
