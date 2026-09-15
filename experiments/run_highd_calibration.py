"""Calibrate the model's lateral/lane-keeping block against real highD data.

Stage 1 of a two-stage calibration. highD contains no emergency vehicle, so what
it can identify is the *host-traffic* behaviour the EV terms act on: how a
lateral manoeuvre is executed (duration, centre-to-centre time, peak lateral
speed and acceleration) and how precisely drivers hold a lane. The EV-response
block keeps its dashcam-GT / literature calibration (`emv/params.py:TUNED`,
untouched); stage 2 re-fits it on top of whatever stage 1 changes.

Five parameters (`emv/highd_fit.py:HIGHD_SPEC`): `a_pin`, `zeta_lat`,
`v_lat_max`, `a_lat_max`, `sigma_off`. The search space is widened by
monkeypatching `emv.calibrate`'s module globals in place - the precedent set by
`experiments/run_surrogate_sumo.py:105-108` - so `emv/calibrate.py` itself is
not edited and its own 6-parameter record stands.

    python experiments/run_highd_calibration.py               # ~15 min
    python experiments/run_highd_calibration.py --quick        # ~4 min
    python experiments/run_highd_calibration.py --dry-run      # loss at the
                                                              # current presets

Writes out/highd_calibration.json (theta, loss, per-observable distances, the
full evaluation trace and the baseline losses of Params()/tuned_params()).
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
from emv.params import Params, tuned_params
from emv.runlog import RunLogger
from emv.scenarios import highd_regime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def patch_theta_spec(spec):
    """Point emv.calibrate's samplers at this parameter space (precedent:
    run_surrogate_sumo.py:105-108). calibrate.py is not modified."""
    cal.THETA_SPEC = list(spec)
    cal.NAMES = [s[0] for s in cal.THETA_SPEC]
    cal.LO = np.array([s[1] for s in cal.THETA_SPEC], float)
    cal.HI = np.array([s[2] for s in cal.THETA_SPEC], float)
    return cal.NAMES, cal.LO, cal.HI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lhs", type=int, default=64)
    ap.add_argument("--n-nm", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 12, 13, 14, 15])
    ap.add_argument("--regime", default="freeflow",
                    help="highD regime to fit on (freeflow has 58 of the 88 "
                         "carriageways, so its targets are the best determined)")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--w-shape", type=float, default=hf.W_SHAPE,
                    help="weight of the distribution-shape term; 0 = moments "
                         "only, which is what the first calibration used")
    ap.add_argument("--out", default="out/highd_calibration.json",
                    help="output path (use a distinct one when changing the "
                         "objective, so records are not overwritten)")
    args = ap.parse_args()

    if args.quick:
        args.n_lhs, args.n_nm, args.seeds = 12, 16, [11, 12]

    block_name = f"train:{args.regime}"
    try:
        block = hf.target_block(block_name, args.targets)
    except (FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"{exc}\nrun experiments/run_highd_extract.py first")
    tgt = hf.targets_of(block)
    spread = hf.spread_of(block)
    obs_win = block.get("obs_dur_med", {}).get("value")
    spec_regime = highd_regime(args.regime)
    print(f"targets   : {block_name}  "
          f"({block['_n_recordings']} recordings, "
          f"{block['_n_carriageways']} carriageways)")
    print(f"regime    : lanes={spec_regime['n_lanes']} "
          f"w={spec_regime['lane_width']} density={spec_regime['density']} "
          f"v={spec_regime['lane_speeds']}")
    print(f"obs window: {obs_win} s (highD median track)")
    print(f"seeds     : {args.seeds}\n")

    names, LO, HI = patch_theta_spec(hf.HIGHD_SPEC)
    base = tuned_params().copy(dt=0.06)

    # Progress counter lives here rather than in emv/calibrate.py: nelder_mead is
    # shared with the calibration of record for papers 1-4 and must not be
    # touched, but a silent 20-40 minute refinement phase is impossible to
    # monitor (and it is slower than the LHS phase, because low per-driver
    # lateral ceilings can block the EV so those runs never hit the early stop).
    _n = [0]
    _t0 = [time.time()]

    def f(theta):
        L = hf.loss(theta, seeds=tuple(args.seeds), names=names, tgt=tgt,
                    regime=args.regime, base=base, track_window=obs_win,
                    spec_regime=spec_regime, w_shape=args.w_shape)[0]
        _n[0] += 1
        if _n[0] % 10 == 0:
            print(f"    eval {_n[0]:4d}  last {L:.4f}  "
                  f"({time.time()-_t0[0]:.0f} s)", flush=True)
        return L

    # --- baselines on the same objective -------------------------------------
    t0 = time.time()
    baselines = {}
    for label, p in (("default", Params().copy(dt=0.06)),
                     ("tuned", base)):
        th = np.array([getattr(p, n) for n in names], float)
        L, parts = hf.loss(th, seeds=tuple(args.seeds), names=names, tgt=tgt,
                           regime=args.regime, base=base, track_window=obs_win,
                           spec_regime=spec_regime, w_shape=args.w_shape)
        baselines[label] = dict(theta=dict(zip(names, th.round(4).tolist())),
                                loss=round(L, 4),
                                parts={k: v for k, v in parts.items()})
        print(f"baseline {label:8s} loss {L:7.4f}   "
              f"({', '.join(f'{n}={v:g}' for n, v in zip(names, th))})")
    print(f"  ({time.time()-t0:.0f} s for 2 evaluations)\n")

    if args.dry_run:
        print(json.dumps(baselines, indent=1))
        return

    with RunLogger("highd_calibration",
                   note=f"regime={args.regime} lhs={args.n_lhs} nm={args.n_nm} "
                        f"seeds={args.seeds}",
                   params=dict(names=",".join(names),
                               targets=block_name)) as rl:
        rng = np.random.default_rng(2026)
        X = cal.latin_hypercube(args.n_lhs, rng)
        X = np.vstack([X, np.array([[getattr(base, n) for n in names]])])
        fv = np.empty(X.shape[0])
        for i, th in enumerate(X):
            fv[i] = f(th)
            if (i + 1) % 8 == 0 or i == X.shape[0] - 1:
                print(f"  LHS {i+1:3d}/{X.shape[0]}  best {fv[:i+1].min():.4f}"
                      f"  ({time.time()-t0:.0f} s)", flush=True)
        k = int(np.argmin(fv))
        print(f"\nLHS best {fv[k]:.4f} at "
              f"{dict(zip(names, X[k].round(4).tolist()))}\n")

        x_best, f_best, trace = cal.nelder_mead(f, X[k], maxiter=args.n_nm)
        print(f"\nNelder-Mead best {f_best:.4f} at "
              f"{dict(zip(names, x_best.round(4).tolist()))}")

        theta = dict(zip(names, [round(float(v), 4) for v in x_best]))
        p_fit = hf.make_params(x_best, names, base)
        obs = hf.model_observables(p_fit, seeds=tuple(args.seeds),
                                  regime=args.regime, track_window=obs_win,
                                  spec=spec_regime)
        d = hf.distance(obs, tgt)

        print(f"\n{'observable':24s}{'highD':>9s}{'sd':>7s}{'model':>9s}"
              f"{'|dev|/sd':>9s}")
        print("-" * 58)
        for key, scale, weight in hf.OBS_SPEC:
            if key not in tgt:
                continue
            sd = spread.get(key) or float("nan")
            m = obs.get(key, float("nan"))
            print(f"{key:24s}{tgt[key]:9.3f}{sd:7.3f}{m:9.3f}"
                  f"{abs(m-tgt[key])/sd if sd else float('nan'):9.2f}"
                  f"{'   (not fitted)' if weight == 0 else ''}")

        res = dict(theta=theta, loss=round(float(f_best), 4),
                   loss_default=baselines["default"]["loss"],
                   loss_tuned=baselines["tuned"]["loss"],
                   baselines=baselines, regime=args.regime,
                   objective=("moments+shape" if args.w_shape else "moments"),
                   w_shape=args.w_shape,
                   targets_block=block_name, seeds=list(args.seeds),
                   obs_window=obs_win, n_evals=int(X.shape[0] + len(trace)),
                   nm_trace=[round(float(v), 5) for v in trace],
                   observables=obs, distance=d,
                   targets=tgt, target_sd=spread,
                   regime_spec={k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in spec_regime.items()},
                   evals=[dict(zip(names, [round(float(v), 4) for v in x]),
                               loss=round(float(y), 5))
                          for x, y in zip(X, fv)])
        path = os.path.join(ROOT, args.out)
        json.dump(res, open(path, "w"), indent=1)
        print(f"\nwrote {path}")
        rl.finish(metrics=dict(loss=res["loss"], loss_tuned=res["loss_tuned"],
                               fit=d["total"], n_evals=res["n_evals"], **theta),
                  outputs=[args.out])


if __name__ == "__main__":
    main()
