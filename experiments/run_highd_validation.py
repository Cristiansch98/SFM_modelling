"""Validate the highD-calibrated model out of sample, and map parameter effects.

Four designs, selectable with `--only`:

  selfdist    highD's own leave-one-carriageway-out distance. This is the
              acceptance bar: a model that sits no further from the pooled real
              reference than one real carriageway sits from the others has
              matched the data as well as the data matches itself. Same argument
              that made SUMO's 0.213 self-distance the bar for the surrogate fit.
  holdout     The calibrated parameters scored against target blocks they were
              never fitted on - the held-out locations (3, 4, 6) and the dense
              and congested regimes - with bootstrap CIs on both sides and a
              per-observable pass/fail against 2 s.d. of natural variation.
  correlation Latin-hypercube sample over the five fitted parameters with common
              random numbers, analysed with Spearman rho and standardized
              regression coefficients against *each observable's error*. Answers
              "which parameter controls which mismatch", and which are not
              identifiable from highD at all.
  tradeoff    What the realism gain costs the EV: speed ratio, min TTC,
              collisions and clearance at TUNED versus the highD-calibrated set,
              plus the analytic depinning half-width d*, which moves with a_pin.

    python experiments/run_highd_validation.py                    # ~25 min
    python experiments/run_highd_validation.py --quick             # ~5 min
    python experiments/run_highd_validation.py --only holdout tradeoff

Writes out/highd_validation.json and out/highd_validation.npz.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import calibrate as cal
from emv import highd_fit as hf
from emv import metrics
from emv.params import Params, tuned_params
from emv.runlog import RunLogger
from emv.scenarios import highd_regime, make_highd_like

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

SCORED = [k for k, _, w in hf.OBS_SPEC if w > 0]


# ---------------------------------------------------------------- helpers
def load_calibration(path=None):
    p = path or os.path.join(OUT, "highd_calibration.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found - run experiments/run_highd_calibration.py")
    return json.load(open(p))


def carriageway_cells(block_regime=None, path=None):
    """Per-carriageway observable dicts from out/highd_recordings.json."""
    p = path or os.path.join(OUT, "highd_recordings.json")
    js = json.load(open(p))
    cells = []
    for key, blk in js["recordings"].items():
        for d, o in blk["observables"].items():
            if block_regime and o.get("regime") != block_regime:
                continue
            cells.append(dict(o, _rid=key, _dir=d,
                              _split=js["split"].get(key, "unused")))
    return cells


def pool_mean(cells, keys=SCORED):
    out = {}
    for k in keys:
        v = np.array([c[k] for c in cells if k in c and np.isfinite(c[k])], float)
        if v.size:
            out[k] = float(np.mean(v))
    return out


def boot_ci(a, n=2000, q=(2.5, 97.5), seed=7):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, q))


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, float)
    r[order] = np.arange(a.size, dtype=float)
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def src(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    m = np.isfinite(y)
    X, y = X[m], y[m]
    if y.size < X.shape[1] + 2:
        return np.full(X.shape[1], np.nan), float("nan")
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xs)), Xs], ys, rcond=None)
    pred = np.c_[np.ones(len(Xs)), Xs] @ beta
    ss = ((ys - pred) ** 2).sum()
    r2 = 1.0 - ss / max(((ys - ys.mean()) ** 2).sum(), 1e-12)
    return beta[1:], float(r2)


# ---------------------------------------------------------------- designs
def study_selfdist(regime="freeflow"):
    """highD's leave-one-carriageway-out distance to its own pooled reference."""
    cells = carriageway_cells(regime)
    ds = []
    for i, c in enumerate(cells):
        others = pool_mean([x for j, x in enumerate(cells) if j != i])
        ds.append(hf.distance(c, others)["total"])
    ds = np.array([d for d in ds if np.isfinite(d)])
    return dict(regime=regime, n_cells=len(cells),
                mean=round(float(ds.mean()), 4),
                med=round(float(np.median(ds)), 4),
                p90=round(float(np.percentile(ds, 90)), 4),
                max=round(float(ds.max()), 4),
                values=[round(float(v), 4) for v in ds])


def study_holdout(theta, names, base, seeds, blocks, log=print):
    out = {}
    for name in blocks:
        try:
            blk = hf.target_block(name)
        except KeyError:
            continue
        tgt, sd = hf.targets_of(blk), hf.spread_of(blk)
        regime = name.split(":")[-1] if ":" in name else "freeflow"
        spec = highd_regime(regime)
        win = blk.get("obs_dur_med", {}).get("value")
        row = {}
        for label, p in (("tuned", base),
                         ("highd", hf.make_params(theta, names, base))):
            obs = hf.model_observables(p, seeds=seeds, regime=regime,
                                       track_window=win, spec=spec)
            d = hf.distance(obs, tgt)
            row[label] = dict(observables={k: round(float(v), 4)
                                           for k, v in obs.items()
                                           if isinstance(v, (int, float))},
                              distance=d)
        cells = carriageway_cells(regime)
        row["pass"] = {}
        for k in SCORED:
            if k not in tgt or not sd.get(k):
                continue
            got = row["highd"]["observables"].get(k, float("nan"))
            lo, hi = boot_ci([c[k] for c in cells if k in c])
            row["pass"][k] = dict(
                highd=round(tgt[k], 4), sd=round(sd[k], 4),
                highd_ci=[round(lo, 4), round(hi, 4)],
                model=round(float(got), 4),
                dev_sd=round(abs(got - tgt[k]) / sd[k], 2),
                ok=bool(abs(got - tgt[k]) <= 2.0 * sd[k]))
        row["n_pass"] = int(sum(v["ok"] for v in row["pass"].values()))
        row["n_obs"] = len(row["pass"])
        row["targets"] = {k: round(v, 4) for k, v in tgt.items()}
        row["target_sd"] = {k: round(v, 4) for k, v in sd.items() if v}
        out[name] = row
        log(f"  {name:22s} tuned {row['tuned']['distance']['total']:.3f}  "
            f"highD-cal {row['highd']['distance']['total']:.3f}   "
            f"pass {row['n_pass']}/{row['n_obs']}")
    return out


def study_correlation(n, seeds, base, regime, log=print):
    blk = hf.target_block(f"train:{regime}")
    tgt = hf.targets_of(blk)
    win = blk.get("obs_dur_med", {}).get("value")
    spec = highd_regime(regime)
    names = hf.HIGHD_NAMES
    lo = np.array([s[1] for s in hf.HIGHD_SPEC])
    hi = np.array([s[2] for s in hf.HIGHD_SPEC])
    rng = np.random.default_rng(2027)
    u = (rng.permuted(np.tile(np.arange(n), (len(names), 1)), axis=1).T
         + rng.random((n, len(names)))) / n
    X = lo + u * (hi - lo)
    rows, errs = [], []
    t0 = time.time()
    for i, th in enumerate(X):
        p = hf.make_params(th, names, base)
        obs = hf.model_observables(p, seeds=seeds, regime=regime,
                                   track_window=win, spec=spec)
        d = hf.distance(obs, tgt)
        rows.append(obs)
        errs.append([obs.get(k, np.nan) - tgt[k] if k in tgt else np.nan
                     for k in SCORED] + [d["total"]])
        if (i + 1) % 16 == 0:
            log(f"  LHS {i+1:3d}/{n}  ({time.time()-t0:.0f} s)")
    E = np.array(errs, float)
    rho = np.array([[spearman(X[:, i], E[:, j]) for j in range(E.shape[1])]
                    for i in range(X.shape[1])])
    beta, r2 = [], []
    for j in range(E.shape[1]):
        b, r = src(X, E[:, j])
        beta.append(b)
        r2.append(r)
    return dict(names=names, responses=SCORED + ["total"],
                X=X, E=E, rho=rho, beta=np.array(beta).T, r2=np.array(r2),
                n=n, seeds=list(seeds), regime=regime,
                significant=round(float(2.0 / np.sqrt(n)), 3))


def study_shape(theta, names, base, seeds, regime="freeflow", n_bar=24):
    """Distribution-shape agreement, with highD's own between-recording shape
    variability as the acceptance bar.

    Percentiles can be matched by distributions of quite different shape, so this
    is scored separately from the moment distances (see
    docs/HIGHD_VALIDATION.md sec. 7b).
    """
    import glob
    spec = highd_regime(regime)
    blk = hf.target_block(f"train:{regime}")
    win = blk.get("obs_dur_med", {}).get("value")
    pools = hf.data_pools()

    # bar: one real recording's distribution against the pooled rest
    files = sorted(glob.glob(os.path.join(OUT, "highd", "rec_*.npz")))[:n_bar]
    bar = {k: [] for k in hf.SHAPE_KEYS}
    for f in files:
        z = np.load(f)
        for k in hf.SHAPE_KEYS:
            if k in z.files and z[k].size > 30:
                other = np.concatenate([np.load(g)[k] for g in files
                                        if g != f and k in np.load(g).files])
                iqr = np.percentile(other, 75) - np.percentile(other, 25)
                if iqr > 0:
                    bar[k].append(hf.wasserstein1(z[k], other) / iqr)
    bar_s = {k: dict(mean=round(float(np.mean(v)), 4),
                     p90=round(float(np.percentile(v, 90)), 4), n=len(v))
             for k, v in bar.items() if v}
    bar_total = round(float(np.mean([b["mean"] for b in bar_s.values()])), 4)

    out = dict(bar=bar_s, bar_total=bar_total, models={})
    for label, p in (("tuned", base),
                     ("highd", hf.make_params(theta, names, base))):
        _, mp = hf.model_observables(p, seeds=seeds, regime=regime,
                                     track_window=win, spec=spec,
                                     return_pools=True)
        out["models"][label] = hf.shape_distance(mp, pools)
    return out


def study_tradeoff(theta, names, base, seeds, regime="freeflow"):
    spec = highd_regime(regime)
    out = {}
    for label, p in (("tuned", base), ("highd", hf.make_params(theta, names, base))):
        rows = []
        for s in seeds:
            sim = make_highd_like(seed=s, p=p, regime=regime, spec=spec,
                                  road_len=3000.0)
            h = sim.run(110.0, rec_dt=0.12, stop_when_ev_x=2950.0)
            m = metrics.evaluate(h)
            rows.append({k: float(v) for k, v in m.items()
                         if isinstance(v, (int, float))})
        agg = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows], float)
            v = v[np.isfinite(v)]
            agg[k] = round(float(v.mean()), 4) if v.size else None
        # analytic depinning half-width (docs/FRAMEWORK.md sec. 4)
        a_pin_eff = p.a_pin * (1.0 - p.urgency_pin_relief)
        w_need = 2.3
        agg["d_star"] = (round(float(w_need + p.B_c * np.log(p.A_c / a_pin_eff)), 3)
                         if a_pin_eff > 0 and p.A_c > 0 else None)
        agg["a_pin_eff"] = round(float(a_pin_eff), 4)
        out[label] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["selfdist", "holdout", "shape", "correlation",
                             "tradeoff"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[21, 22, 23, 24, 25])
    ap.add_argument("--n-lhs", type=int, default=96)
    ap.add_argument("--corr-seeds", type=int, nargs="*", default=None,
                    help="seeds for the correlation sweep (default: first 3)")
    ap.add_argument("--regime", default="freeflow")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_lhs, args.seeds = 24, [21, 22]
    todo = args.only or ["selfdist", "holdout", "shape", "correlation",
                         "tradeoff"]

    calres = load_calibration()
    names = list(calres["theta"].keys())
    theta = np.array([calres["theta"][n] for n in names], float)
    base = tuned_params().copy(dt=0.06)
    print(f"calibrated: {calres['theta']}")
    print(f"loss {calres['loss']} vs tuned {calres['loss_tuned']}\n")

    summary, npz = dict(theta=calres["theta"], seeds=list(args.seeds)), {}

    def save():
        """Persist after every design: the correlation sweep is long enough that
        losing a whole run to a killed process is a real cost (it happened)."""
        json.dump(summary, open(os.path.join(OUT, "highd_validation.json"), "w"),
                  indent=1, default=float)
        if npz:
            np.savez_compressed(os.path.join(OUT, "highd_validation.npz"), **npz)

    with RunLogger("highd_validation",
                   note=f"only={','.join(todo)} n_lhs={args.n_lhs} "
                        f"seeds={args.seeds}") as rl:
        t0 = time.time()
        if "selfdist" in todo:
            print("[selfdist] highD leave-one-carriageway-out ...")
            summary["selfdist"] = {r: study_selfdist(r)
                                   for r in ("freeflow", "dense")}
            for r, v in summary["selfdist"].items():
                print(f"  {r:10s} n={v['n_cells']:3d}  mean {v['mean']:.3f}  "
                      f"med {v['med']:.3f}  p90 {v['p90']:.3f}")
            save()
        if "holdout" in todo:
            print("\n[holdout] model vs unseen target blocks ...")
            summary["holdout"] = study_holdout(
                theta, names, base, tuple(args.seeds),
                ["train:freeflow", "holdout:freeflow", "all:dense",
                 "all:congested"])
            save()
        if "shape" in todo:
            print("\n[shape] distribution agreement vs highD's own bar ...")
            summary["shape"] = study_shape(theta, names, base,
                                           tuple(args.seeds), args.regime)
            sh = summary["shape"]
            print(f"  bar (one real recording vs the rest): {sh['bar_total']}")
            for lab, d in sh["models"].items():
                print(f"  {lab:6s} total {d['total']}   " + "  ".join(
                    f"{k.split('_')[-1]} IQRr={d[k]['iqr_ratio']}"
                    for k in hf.SHAPE_KEYS if k in d))
            save()

        if "tradeoff" in todo:
            print("\n[tradeoff] EV performance cost ...")
            summary["tradeoff"] = study_tradeoff(theta, names, base,
                                                 tuple(args.seeds), args.regime)
            a, b = summary["tradeoff"]["tuned"], summary["tradeoff"]["highd"]
            for k in ("ev_speed_ratio", "min_ttc", "collisions",
                      "clearance_mean", "react_dist_mean", "d_star"):
                print(f"  {k:18s} tuned {a.get(k)}   highD-cal {b.get(k)}")
            save()
        if "correlation" in todo:
            # Defaults to the full seed set, because the recorded identifiability
            # numbers (docs/HIGHD_VALIDATION.md sec. 7) were produced with it.
            # --corr-seeds trades resolution for wall time on a re-run.
            cs = tuple(args.corr_seeds or args.seeds)
            print(f"\n[correlation] LHS {args.n_lhs} x {len(cs)} seeds ...")
            c = study_correlation(args.n_lhs, cs, base, args.regime)
            npz.update(corr_X=c["X"], corr_E=c["E"], corr_rho=c["rho"],
                       corr_beta=c["beta"], corr_r2=c["r2"])
            summary["correlation"] = dict(
                names=c["names"], responses=c["responses"], n=c["n"],
                regime=c["regime"], significant=c["significant"],
                rho={n: {r: round(float(c["rho"][i, j]), 3)
                         for j, r in enumerate(c["responses"])}
                     for i, n in enumerate(c["names"])},
                r2={r: round(float(c["r2"][j]), 3)
                    for j, r in enumerate(c["responses"])})
            print(f"  |rho| > {c['significant']} is significant at N={c['n']}")
            for i, n in enumerate(c["names"]):
                strong = sorted(((abs(c["rho"][i, j]), c["responses"][j],
                                  c["rho"][i, j])
                                 for j in range(len(c["responses"]))),
                                reverse=True)[:3]
                print(f"  {n:12s} " + ", ".join(
                    f"{r} {v:+.2f}" for _, r, v in strong if _ > c["significant"]))

        summary["wall_s"] = round(time.time() - t0, 1)
        save()
        print(f"\nwrote out/highd_validation.json ({summary['wall_s']} s)")
        met = dict(wall_s=summary["wall_s"])
        if "holdout" in summary:
            h = summary["holdout"].get("holdout:freeflow", {})
            met = dict(holdout_dist_highd=h.get("highd", {}).get("distance", {}).get("total"),
                       holdout_dist_tuned=h.get("tuned", {}).get("distance", {}).get("total"),
                       holdout_pass=h.get("n_pass"), **met)
        rl.finish(metrics=met, outputs=["out/highd_validation.json",
                                        "out/highd_validation.npz"])


if __name__ == "__main__":
    main()
