"""Stage 1: calibrate ordinary motorway traffic against highD, EV absent.

    python experiments/run_normal_calibration.py --block lon           # ~65 min
    python experiments/run_normal_calibration.py --block latlc         # ~95 min
    python experiments/run_normal_calibration.py --block joint --warm \
        out/normal_calibration_lon.json out/normal_calibration_latlc.json
    python experiments/run_normal_calibration.py --quick               # ~4 min
    python experiments/run_normal_calibration.py --dry-run             # presets

Use `latlc`, NOT `lat` then `lc`: the lateral and discretionary blocks are coupled
through the depinning threshold `A_pass * frust > a_pin_eff`, so fitting them
separately lets one undo what the other produced. Measured - fitting `lat` alone
raised `a_pin` to 1.53 and thereby cut the manoeuvre rate to 0.008 against a
measured 0.276, while scoring better on the kinematics because the few survivors
were the well-shaped ones (see emv/highd_fit.py:SPEC_LATLC and LC_MIN_EVENTS).

What makes this different from `run_highd_calibration.py` (which stays as the
record of the 2026-07-29 fit):

* **No emergency vehicle in the scenario** (`ev_inert=True`). The previous fit
  ran with a live EV, and since the model had no discretionary lane-change
  mechanism, every lane change it scored was EV-induced - measured: 0 lane
  changes over 206 veh-km once the EV is removed. Scoring EV-induced manoeuvres
  against highD's discretionary ones made the "host-traffic" fit circular.
* **Two densities, not one.** `k_rho` is a density exponent and cannot be
  identified at a single density, and free-flow-only fitting is what left the
  congested regime worse than no calibration at all. Congested (2 carriageways)
  is held out.
* **The car-following block is scored.** highD's headway and TTC distributions
  had no model counterpart until `metrics.headway_pools`; now they are targets.
* **The lane-change rate is fitted**, not carried at weight 0, because the model
  finally has a mechanism that can set it.

Blocks are fitted before the joint polish rather than all 19 parameters at once:
a Nelder-Mead simplex in 19 dimensions needs far more evaluations than the budget
allows, and the sensitivity pass (`run_term_influence.py --only sensitivity`)
shows 18 of the 19 parameters own a distinct primary response. `--block joint`
then polishes everything together, warm-started from the two block records.

Writes out/normal_calibration[_<block>].json.
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
from emv.params import Params, highd_params
from emv.runlog import RunLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def patch_theta_spec(spec):
    """Point emv.calibrate's samplers at this parameter space (the precedent set
    by run_surrogate_sumo.py:105 and run_highd_calibration.py:43). calibrate.py
    is never edited, so its own 6-parameter record stands."""
    cal.THETA_SPEC = list(spec)
    cal.NAMES = [s[0] for s in cal.THETA_SPEC]
    cal.LO = np.array([s[1] for s in cal.THETA_SPEC], float)
    cal.HI = np.array([s[2] for s in cal.THETA_SPEC], float)
    return cal.NAMES, cal.LO, cal.HI


def load_warm(paths, base: Params) -> Params:
    """Apply previously fitted blocks on top of `base`, in order."""
    for path in paths or []:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        with open(full) as fh:
            js = json.load(fh)
        base = base.copy(**js["theta"])
        print(f"  warm start from {path}: "
              + " ".join(f"{k}={v:g}" for k, v in js["theta"].items()))
    return base


def report_table(ctx, obs_per, regimes):
    tgt0 = ctx[regimes[0]]["tgt"]
    head = f"{'observable':26s}{'w':>4s}"
    for r in regimes:
        head += f"{r[:6]+' highD':>14s}{'model':>9s}{'|d|/sd':>8s}"
    print(head)
    print("-" * len(head))
    for key, scale, weight in hf.OBS_SPEC_NORMAL:
        if key not in tgt0:
            continue
        line = f"{key:26s}{weight:4.1f}"
        for r in regimes:
            t = ctx[r]["tgt"].get(key, float("nan"))
            sd = ctx[r]["sd"].get(key) or float("nan")
            m = obs_per[r].get(key, float("nan"))
            line += f"{t:14.3f}{m:9.3f}{abs(m - t) / sd if sd else float('nan'):8.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="joint",
                    choices=list(hf.NORMAL_BLOCKS) + ["all"])
    ap.add_argument("--n-lhs", type=int, default=None)
    ap.add_argument("--n-nm", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 12, 13])
    ap.add_argument("--regimes", nargs="*", default=list(hf.FIT_REGIMES))
    ap.add_argument("--warm", nargs="*", default=None,
                    help="previously fitted block JSONs to start from")
    ap.add_argument("--w-shape", type=float, default=hf.W_SHAPE)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = hf.NORMAL_BLOCKS[args.block]
    n_lhs = args.n_lhs if args.n_lhs is not None else max(24, 6 * len(spec))
    n_nm = args.n_nm if args.n_nm is not None else 50
    if args.quick:
        n_lhs, n_nm, args.seeds = 6, 8, [11]
    out_path = args.out or f"out/normal_calibration_{args.block}.json"

    regimes = tuple(args.regimes)
    ctx = hf.normal_context(regimes)
    # The base is the 2026-07-29 lateral fit with the discretionary term ON at a
    # hand-set starting point, so the LHS does not have to discover from scratch
    # that a nonzero incentive is needed for any lc_* observable to exist at all
    # (with A_pass below the pinning threshold every one of them is NaN, which
    # the objective scores as a flat 2 scale units - a plateau, not a gradient).
    base = highd_params().copy(dt=0.06, A_pass=4.0, A_keep_right=2.0)
    base = load_warm(args.warm, base)

    print(f"block     : {args.block}  ({len(spec)} parameters)")
    print(f"regimes   : {list(regimes)}  (congested deliberately held out)")
    for r in regimes:
        print(f"  {r:9s} {ctx[r]['n_cw']} carriageways, "
              f"density {ctx[r]['spec']['density']} veh/km/lane, "
              f"lane speeds {ctx[r]['spec']['lane_speeds']}, "
              f"window {ctx[r]['window']} s")
    print(f"seeds     : {args.seeds}   LHS {n_lhs} + NM {n_nm}\n")

    names, LO, HI = patch_theta_spec(spec)
    _n, _t0 = [0], [time.time()]

    def f(theta):
        L = hf.loss_normal(theta, ctx, seeds=tuple(args.seeds), names=names,
                           base=base, spec_theta=spec, w_shape=args.w_shape,
                           regimes=regimes)[0]
        _n[0] += 1
        if _n[0] % 10 == 0:
            print(f"    eval {_n[0]:4d}  last {L:.4f}  "
                  f"({time.time()-_t0[0]:.0f} s)", flush=True)
        return L

    # ---- baselines on the same objective ----------------------------------
    baselines = {}
    for label, p in (("default", Params().copy(dt=0.06)),
                     ("highd_2026_07_29", highd_params().copy(dt=0.06)),
                     ("start", base)):
        th = np.array([getattr(p, n) for n in names], float)
        L, parts = hf.loss_normal(th, ctx, seeds=tuple(args.seeds), names=names,
                                  base=p, spec_theta=spec,
                                  w_shape=args.w_shape, regimes=regimes)
        baselines[label] = dict(loss=round(float(L), 4), parts=parts,
                                theta=dict(zip(names, np.round(th, 4).tolist())))
        print(f"baseline {label:18s} loss {L:7.4f}")
    print()

    if args.dry_run:
        print(json.dumps(baselines, indent=1))
        return

    with RunLogger("normal_calibration",
                   note=f"block={args.block} regimes={','.join(regimes)} "
                        f"lhs={n_lhs} nm={n_nm} seeds={args.seeds} EV-free",
                   params=dict(names=",".join(names),
                               regimes=",".join(regimes))) as rl:
        rng = np.random.default_rng(2026)
        X = cal.latin_hypercube(n_lhs, rng)
        X = np.vstack([X, np.array([[getattr(base, n) for n in names]])])
        fv = np.empty(X.shape[0])
        for i, th in enumerate(X):
            fv[i] = f(th)
            if (i + 1) % 8 == 0 or i == X.shape[0] - 1:
                print(f"  LHS {i+1:3d}/{X.shape[0]}  best {fv[:i+1].min():.4f}"
                      f"  ({time.time()-_t0[0]:.0f} s)", flush=True)
        k = int(np.argmin(fv))
        print(f"\nLHS best {fv[k]:.4f} at "
              f"{dict(zip(names, X[k].round(4).tolist()))}\n")

        x_best, f_best, trace = cal.nelder_mead(f, X[k], maxiter=n_nm)
        theta = dict(zip(names, [round(float(v), 4) for v in x_best]))
        print(f"\nNelder-Mead best {f_best:.4f} at {theta}\n")

        L, parts, per = hf.loss_normal(x_best, ctx, seeds=tuple(args.seeds),
                                       names=names, base=base, spec_theta=spec,
                                       w_shape=args.w_shape, regimes=regimes,
                                       return_detail=True)
        report_table(ctx, {r: per[r]["observables"] for r in regimes}, regimes)

        res = dict(
            block=args.block, theta=theta, loss=round(float(f_best), 4),
            regimes=list(regimes), seeds=list(args.seeds),
            w_shape=args.w_shape, ev_inert=True,
            baselines=baselines, n_evals=int(X.shape[0] + len(trace)),
            nm_trace=[round(float(v), 5) for v in trace],
            per_regime={r: {k: v for k, v in per[r].items()
                            if k != "observables"} for r in regimes},
            observables={r: per[r]["observables"] for r in regimes},
            targets={r: ctx[r]["tgt"] for r in regimes},
            target_sd={r: ctx[r]["sd"] for r in regimes},
            base_theta={n: getattr(base, n) for n in
                        [s[0] for s in hf.SPEC_NORMAL]},
            evals=[dict(zip(names, [round(float(v), 4) for v in x]),
                        loss=round(float(y), 5)) for x, y in zip(X, fv)])
        path = os.path.join(ROOT, out_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(f"\nwrote {path}")
        rl.finish(metrics=dict(loss=res["loss"],
                               loss_start=baselines["start"]["loss"],
                               loss_prev=baselines["highd_2026_07_29"]["loss"],
                               n_evals=res["n_evals"], **theta),
                  outputs=[out_path])


if __name__ == "__main__":
    main()
