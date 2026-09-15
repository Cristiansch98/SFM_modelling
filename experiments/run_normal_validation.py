"""Out-of-sample validation of the stage-1 (EV-free) calibration.

    python experiments/run_normal_validation.py            # ~25 min
    python experiments/run_normal_validation.py --quick

Four questions, in the order that matters:

  1. **What would "good" even mean?** highD's own leave-one-carriageway-out
     distance: how far one real carriageway sits from the pooled reference of the
     others, under *this* objective. A model no further away than that is inside
     the data's own variability. Computed per regime, and it is the only number
     the fitted losses should be compared with.
  2. **Does it hold on seeds never used in the fit?** Fresh seeds 21-25.
  3. **Does it hold on locations never used in the fit?** The holdout split.
  4. **What did it do to the regime that was NOT fitted?** Congested (2
     carriageways) is the block the 2026-07-29 free-flow-only fit made *worse
     than uncalibrated*, so it is reported first-class rather than as a footnote.

Reuses `run_highd_validation`'s `carriageway_cells` / `pool_mean` so the real-data
side is assembled exactly as it was for the published validation, and scores it
with the normal objective's spec (`OBS_SPEC_NORMAL` with the D_CAP convention).

Writes out/normal_validation.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import highd_fit as hf
from emv.params import Params, highd_params
from emv.runlog import RunLogger
from emv.scenarios import highd_regime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_highd_validation import carriageway_cells, pool_mean   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

SCORED = [k for k, _, w in hf.OBS_SPEC_NORMAL if w > 0]
#: the same D_CAP convention the objective uses, so bar and loss are commensurable
DKW = dict(spec=hf.OBS_SPEC_NORMAL, nan=hf.D_CAP, cap=hf.D_CAP)


def merged_theta(paths):
    """Fitted parameters from one or more block records, later files winning."""
    theta = {}
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(full):
            print(f"  (skipping missing {p})")
            continue
        theta.update(json.load(open(full))["theta"])
        print(f"  from {os.path.basename(p)}: "
              + " ".join(f"{k}={v:g}" for k, v in
                         json.load(open(full))["theta"].items()))
    if not theta:
        raise SystemExit("no block records found - run run_normal_calibration.py")
    return theta


def self_distance(regime):
    """highD's leave-one-carriageway-out distance under the normal objective."""
    cells = [c for c in carriageway_cells(regime)]
    ds = []
    for i, c in enumerate(cells):
        others = pool_mean([x for j, x in enumerate(cells) if j != i], SCORED)
        d = hf.distance(c, others, **DKW)["total"]
        if np.isfinite(d):
            ds.append(float(d))
    if not ds:
        return None
    a = np.array(ds)
    return dict(regime=regime, n_cells=len(cells), mean=round(float(a.mean()), 4),
                med=round(float(np.median(a)), 4),
                p90=round(float(np.percentile(a, 90)), 4))


def score_block(p, block, regime, seeds, ctx_spec, window):
    """Model observables vs one pooled target block, EV absent."""
    obs, mp = hf.model_observables(p, seeds=seeds, regime=regime,
                                   track_window=window, spec=ctx_spec,
                                   return_pools=True, ev_inert=True)
    d = hf.distance(obs, block["tgt"], **DKW)
    sh = hf.shape_distance(mp, hf.data_pools(), hf.SHAPE_KEYS_NORMAL,
                           miss=hf.SHAPE_MISS)
    within = sum(1 for k in SCORED
                 if k in block["tgt"] and block["sd"].get(k)
                 and np.isfinite(obs.get(k, np.nan))
                 and abs(obs[k] - block["tgt"][k]) <= 2.0 * block["sd"][k])
    n = sum(1 for k in SCORED if k in block["tgt"] and block["sd"].get(k))
    return dict(moment=d["total"], shape=sh["total"], within_2sd=f"{within}/{n}",
                collisions=obs.get("collisions", 0), distance=d,
                observables=obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=[
        "out/normal_calibration_lon.json", "out/normal_calibration_lat.json",
        "out/normal_calibration_lc.json", "out/normal_calibration_joint.json"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[21, 22, 23, 24, 25],
                    help="fresh seeds, never used in any fit")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="out/normal_validation.json")
    args = ap.parse_args()
    if args.quick:
        args.seeds = [21, 22]

    print("fitted parameters")
    theta = merged_theta(args.blocks)
    base = highd_params().copy(dt=0.06, A_pass=4.0, A_keep_right=2.0)
    p_fit = base.copy(**theta)
    p_prev = highd_params().copy(dt=0.06)          # the 2026-07-29 fit
    p_def = Params().copy(dt=0.06)

    t0 = time.time()
    with RunLogger("normal_validation",
                   note=f"fresh seeds {args.seeds}, EV-free, "
                        f"blocks={[os.path.basename(b) for b in args.blocks]}",
                   params=dict(seeds=",".join(map(str, args.seeds)))) as rl:
        print("\n1. acceptance bar: highD's own leave-one-carriageway-out distance")
        bars = {}
        for r in ("freeflow", "dense", "congested"):
            b = self_distance(r)
            if b:
                bars[r] = b
                print(f"  {r:10s} {b['n_cells']:3d} carriageways   "
                      f"mean {b['mean']:.4f}  median {b['med']:.4f}")

        print(f"\n2-4. model on fresh seeds {args.seeds}")
        blocks = {}
        for name in ("train:freeflow", "train:dense", "holdout:freeflow",
                     "all:congested"):
            regime = name.split(":")[1]
            try:
                blk = hf.target_block(name)
            except KeyError:
                print(f"  (no target block {name})")
                continue
            spec = highd_regime(regime)
            window = blk.get("obs_dur_med", {}).get("value")
            tgt = dict(tgt=hf.targets_of_spec(blk, hf.OBS_SPEC_NORMAL),
                       sd=hf.spread_of_spec(blk, hf.OBS_SPEC_NORMAL))
            row = {}
            for label, p in (("fitted", p_fit), ("previous", p_prev),
                             ("default", p_def)):
                row[label] = score_block(p, tgt, regime, tuple(args.seeds),
                                         spec, window)
                print(f"  {name:18s} {label:9s} moment {row[label]['moment']:7.3f}"
                      f"  shape {row[label]['shape']:6.3f}"
                      f"  within2sd {row[label]['within_2sd']:>7s}"
                      f"  coll {row[label]['collisions']:3d}"
                      f"   ({time.time()-t0:.0f} s)", flush=True)
            row["bar"] = bars.get(regime, {}).get("mean")
            blocks[name] = row

        res = dict(theta=theta, seeds=list(args.seeds), bars=bars, blocks=blocks,
                   wall_s=round(time.time() - t0, 1))
        path = os.path.join(ROOT, args.out)
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(f"\nwrote {path}")
        m = {f"{k.replace(':', '_')}_moment": v["fitted"]["moment"]
             for k, v in blocks.items()}
        rl.finish(metrics=dict(m, bar_freeflow=bars.get("freeflow", {}).get("mean")),
                  outputs=[args.out])


if __name__ == "__main__":
    main()
