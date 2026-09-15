"""Stage 2: fit the emergency-vehicle repulsion block on the frozen normal model.

    python experiments/run_ev_calibration.py --gt-dir ~/emergency-vehicle-dataset-pipeline/output
    python experiments/run_ev_calibration.py --ablate      # E-stage term influence
    python experiments/run_ev_calibration.py --quick
    python experiments/run_ev_calibration.py --dry-run

**Exactly four parameters are fitted**: `A_ev`, `B_ev` (the point repulsion the
emergency vehicle exerts) and `A_c`, `B_c` (the repulsion from its predicted
corridor). Everything else - the whole normal-traffic block from stage 1, and the
EV's own perception timing, urgency and longitudinal merge terms - is frozen.
That is the requested experiment: model and tune ordinary traffic first, then
change only the parameters describing how other cars are repelled by the
emergency vehicle.

Why this can be done without SUMO: the ground truth is the dashcam dataset, and
the whole scoring path is standalone - `groundtruth.ego_stream` (lifted out of
run_surrogate_sumo.py) -> `annotate_stream` -> `fingerprint` ->
`fingerprint_distance`. The scenario is `make_highd_like` with the EV active, so
stage 2 sits on the *same road, fleet and demand* that stage 1 was fitted to,
rather than on a different scenario whose traffic was never calibrated.

The objective is the behaviour-fingerprint distance to the ground truth plus
safety and comfort guards taken from `emv/calibrate.py` (collisions, minimum
time-to-collision, comfortable deceleration). Without those guards a fingerprint
can be improved by driving the traffic harder than any real driver would, which
is not a better model of yielding - it is a worse model that scores well.

Writes out/ev_calibration.json (+ out/ev_ablation.json with --ablate).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import calibrate as cal
from emv import groundtruth as gt
from emv import metrics
from emv import terms
from emv.params import (Params, bluelight_params, highd_params,
                        normal_params)
from emv.runlog import RunLogger
from emv.scenarios import highd_regime, make_highd_like

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: The four fitted parameters, with the bounds of the original EV calibration
#: (emv/calibrate.py:THETA_SPEC) so the search space is the published one.
EV_SPEC = [
    ("A_ev", 1.5, 8.0),
    ("B_ev", 8.0, 40.0),
    ("A_c",  1.5, 8.0),
    ("B_c",  0.6, 3.0),
]
TRAIN_SEEDS = (11, 12, 13)
VAL_SEEDS = (21, 22, 23)
#: Run long enough for the EV to traverse the section and the traffic to merge
#: back, on the same 3 km road the normal calibration used.
T_RUN, ROAD_LEN = 110.0, 3000.0


def patch_theta_spec(spec):
    cal.THETA_SPEC = list(spec)
    cal.NAMES = [s[0] for s in cal.THETA_SPEC]
    cal.LO = np.array([s[1] for s in cal.THETA_SPEC], float)
    cal.HI = np.array([s[2] for s in cal.THETA_SPEC], float)
    return cal.NAMES, cal.LO, cal.HI


def base_params(fallback_lc: dict, fitted: bool = False) -> tuple:
    """The frozen stage-1 block, or the best available stand-in.

    `fitted=True` starts from TUNED_BLUELIGHT instead - i.e. stage 1 frozen *and*
    the stage-2 EV block at its fitted values. That is the right operating point
    for the **ablations**: a term has to be judged where the model actually sits,
    not at an arbitrary starting point. Measured on the N-stage side, the two give
    opposite answers - switching the discretionary term off *improves* the
    objective before calibration and costs 2.97 after it.
    """
    if fitted:
        try:
            return bluelight_params().copy(dt=0.06), "TUNED_BLUELIGHT"
        except RuntimeError as exc:
            print(f"  ! {exc}\n  ! falling back to the stage-1 base")
    try:
        return normal_params().copy(dt=0.06), "TUNED_NORMAL"
    except RuntimeError as exc:
        print(f"  ! {exc}")
        print("  ! using the 2026-07-29 highD fit + the hand-set incentive "
              "instead; re-run once stage 1 has been calibrated.")
        return highd_params().copy(dt=0.06, **fallback_lc), "highd+LC_START"


def load_gt(gt_dir: str) -> dict:
    """Pooled dashcam ground truth, relabelled with the pure kinematic rules.

    Rules-only labels are the apples-to-apples reference: a simulated stream can
    only ever receive rule labels, whereas the stored ground-truth labels carry
    ~20 % manual-review corrections. Both are reported; the rules-only one is the
    primary score, exactly as in the survey benchmark.
    """
    vids = gt.gt_videos(gt_dir)
    if not vids:
        raise SystemExit(f"no processed videos under {gt_dir}")
    # Pool every processed video with per-video id prefixes, exactly as
    # run_survey_compare.py:pooled_gt does - without the prefix, track ids
    # collide across videos and the track-level shares are wrong.
    frames, by_video = [], {}
    for v in vids:
        tag = os.path.basename(v)
        own = []
        for fr in gt.load_gt_stream(v):
            for veh in fr["vehicles"]:
                veh["id"] = f"{tag}:{veh['id']}"
            own.append(fr)
        by_video[tag] = own
        frames.extend(own)
    if not frames:
        raise SystemExit(f"no emergency-active frames under {gt_dir}")
    fp_reviewed = gt.fingerprint(frames)
    fp_rules = gt.fingerprint(gt.reannotate(frames))
    print(f"  ground truth: {len(vids)} video(s), {len(frames)} frames, "
          f"{fp_rules['n_obs']} observations, {fp_rules['n_tracks']} tracks")

    # Leave-one-video-out self-distance: how far one real recording sits from the
    # pooled reference of the others. This is the acceptance bar - the same
    # construction highD's leave-one-carriageway-out distance uses - and it is
    # needed here for two reasons. The corpus has grown to 5 videos / ~145 k
    # observations since the survey benchmark ran on 1 video / 2528, so the
    # fingerprint distances are NOT comparable with the 0.646-0.812 in
    # PROJECT_LOG sec. 5; and the pooled corpus carries visible tracking noise
    # (5th-percentile acceleration -16 m/s2, which no vehicle produces), so a fit
    # can only sensibly be asked to get as close as the data is to itself.
    per_video, self_d = {}, []
    if len(by_video) > 1:
        for tag, own in by_video.items():
            if len(own) < 20:
                continue
            fp_own = gt.fingerprint(gt.reannotate(own))
            d = gt.fingerprint_distance(fp_own, fp_rules)["total"]
            per_video[tag] = dict(n_frames=len(own), n_obs=fp_own["n_obs"],
                                  distance_to_pooled=round(float(d), 4))
            self_d.append(float(d))
    bar = round(float(np.mean(self_d)), 4) if self_d else None
    if bar is not None:
        print(f"  acceptance bar (mean distance of one video to the pooled "
              f"reference): {bar:.4f}")
    return dict(rules=fp_rules, reviewed=fp_reviewed, n_frames=len(frames),
                videos=[os.path.basename(v) for v in vids],
                per_video=per_video, acceptance_bar=bar)


def run_seeds(p: Params, seeds, regime: str, spec: dict):
    """Pooled ego-frame stream + safety/comfort measures over seeds."""
    pooled, mm = [], []
    for s in seeds:
        sim = make_highd_like(seed=s, p=p, regime=regime, spec=spec,
                              road_len=ROAD_LEN)
        h = sim.run(T_RUN, rec_dt=0.1, stop_when_ev_x=ROAD_LEN - 50.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = metrics.evaluate(h)
        frames = gt.ego_stream(h)
        for fr in frames:                     # keep track ids unique across seeds
            for v in fr["vehicles"]:
                v["id"] = f"{s}:{v['id']}"
        pooled.extend(gt.annotate_stream(
            frames, n_lanes=int(spec["n_lanes"]),
            lane_width=float(spec["lane_width"])))
        mm.append(m)
    agg = {k: float(np.nanmean([m[k] for m in mm]))
           for k in ("ev_speed_ratio", "min_ttc", "p95_decel", "p95_alat",
                     "clearance_mean", "react_dist_mean", "disruption")}
    agg["collisions"] = int(sum(m["collisions"] for m in mm))
    return gt.fingerprint(pooled), agg


def score(fp, agg, tgt) -> tuple:
    """Fingerprint distance plus the safety/comfort guards of calibrate.loss."""
    parts = dict(
        fingerprint=float(gt.fingerprint_distance(fp, tgt["rules"])["total"]),
        safety=float(4.0 * max(0.0, 1.2 - (agg["min_ttc"]
                                           if np.isfinite(agg["min_ttc"])
                                           else 10.0)) ** 2),
        collisions=float(min(5.0 * agg["collisions"], 50.0)),
        comfort=float(1.5 * max(0.0, agg["p95_decel"] - 3.2) ** 2
                      + 1.0 * max(0.0, agg["p95_alat"] - 2.5) ** 2))
    return float(sum(parts.values())), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=gt.PIPELINE_OUT)
    ap.add_argument("--regime", default="freeflow")
    ap.add_argument("--n-lhs", type=int, default=40)
    ap.add_argument("--n-nm", type=int, default=45)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    ap.add_argument("--ablate", action="store_true",
                    help="score the stage-2 term ablations instead of fitting")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="out/ev_calibration.json")
    args = ap.parse_args()
    if args.quick:
        args.n_lhs, args.n_nm, args.seeds = 6, 8, [11]

    gt_dir = os.path.expanduser(args.gt_dir)
    tgt = load_gt(gt_dir)
    spec = highd_regime(args.regime)
    base, base_label = base_params(dict(A_pass=4.0, A_keep_right=2.0),
                                   fitted=args.ablate)
    names, LO, HI = patch_theta_spec(EV_SPEC)
    print(f"  frozen normal block: {base_label}")
    print(f"  fitting {names} on {args.regime} "
          f"(density {spec['density']} veh/km/lane), seeds {args.seeds}\n")

    _n, _t0 = [0], [time.time()]

    def evaluate(p):
        fp, agg = run_seeds(p, tuple(args.seeds), args.regime, spec)
        val, parts = score(fp, agg, tgt)
        return val, parts, fp, agg

    def f(theta):
        p = base.copy(**dict(zip(names, np.clip(theta, LO, HI))))
        val, parts, _, _ = evaluate(p)
        _n[0] += 1
        if _n[0] % 10 == 0:
            print(f"    eval {_n[0]:4d}  last {val:.4f}  "
                  f"({time.time()-_t0[0]:.0f} s)", flush=True)
        return val

    # ---- baselines ---------------------------------------------------------
    # Three reference points, because two of them coincide and that is worth
    # seeing: TUNED_NORMAL deliberately carries no EV parameters, so the frozen
    # normal block leaves them at the dataclass defaults - "inherited" and
    # "defaults" are therefore the same numbers. `tuned_ev` is the published
    # 2026-07-14 EV calibration (params.TUNED), which is the reference stage 2
    # actually has to beat.
    from emv.params import TUNED
    baselines = {}
    for label, th in (("inherited", [getattr(base, n) for n in names]),
                      ("defaults", [getattr(Params(), n) for n in names]),
                      ("tuned_ev", [TUNED.get(n, getattr(Params(), n))
                                    for n in names])):
        p = base.copy(**dict(zip(names, th)))
        val, parts, fp, agg = evaluate(p)
        baselines[label] = dict(theta=dict(zip(names, [round(float(v), 4) for v in th])),
                                loss=round(val, 4), parts=parts,
                                fingerprint=fp, measures=agg)
        print(f"baseline {label:10s} loss {val:7.4f}  "
              f"(fp {parts['fingerprint']:.3f}, coll {agg['collisions']}, "
              f"TTC {agg['min_ttc']:.2f}, EV ratio {agg['ev_speed_ratio']:.3f})")

    if args.ablate:
        print("\nstage-2 term ablations (EV present, dashcam objective)")
        res = dict(baseline=baselines["inherited"], base_label=base_label,
                   regime=args.regime, seeds=list(args.seeds), ablation={})
        L0 = baselines["inherited"]["loss"]
        for name in terms.E_ABLATIONS:
            over, term, stage, why = terms.ABLATIONS[name]
            val, parts, fp, agg = evaluate(base.copy(**over))
            res["ablation"][name] = dict(
                term=term, why=why, overrides=over, loss=round(val, 4),
                d_loss=round(val - L0, 4), parts=parts, measures=agg,
                fingerprint=fp)
            print(f"  {name:20s} loss {val:7.3f}  d {val-L0:+7.3f}   "
                  f"fp {parts['fingerprint']:.3f}  coll {agg['collisions']:2d}  "
                  f"EV ratio {agg['ev_speed_ratio']:.3f}  "
                  f"clear {agg['clearance_mean']:6.1f} m", flush=True)
        with RunLogger("ev_ablation", note=f"base={base_label}") as rl:
            path = os.path.join(ROOT, "out/ev_ablation.json")
            json.dump(res, open(path, "w"), indent=1, default=float)
            print(f"\nwrote {path}")
            rl.finish(metrics=dict(baseline_loss=L0), outputs=["out/ev_ablation.json"])
        return

    if args.dry_run:
        print(json.dumps({k: v["loss"] for k, v in baselines.items()}, indent=1))
        return

    with RunLogger("ev_calibration",
                   note=f"4-param EV repulsion on {base_label}, "
                        f"regime={args.regime} seeds={args.seeds}",
                   params=dict(names=",".join(names), gt_dir=gt_dir)) as rl:
        rng = np.random.default_rng(2026)
        X = cal.latin_hypercube(args.n_lhs, rng)
        X = np.vstack([X, np.array([[getattr(base, n) for n in names]])])
        fv = np.array([f(x) for x in X])
        k = int(np.argmin(fv))
        print(f"\nLHS best {fv[k]:.4f} at {dict(zip(names, X[k].round(3).tolist()))}")

        x_best, f_best, trace = cal.nelder_mead(f, X[k], maxiter=args.n_nm)
        theta = dict(zip(names, [round(float(v), 4) for v in x_best]))
        p_fit = base.copy(**theta)
        val, parts, fp, agg = evaluate(p_fit)
        print(f"\nfit {f_best:.4f} at {theta}")
        print(f"  parts {parts}")

        # ---- out-of-sample validation on fresh seeds ----------------------
        fp_v, agg_v = run_seeds(p_fit, VAL_SEEDS, args.regime, spec)
        val_v, parts_v = score(fp_v, agg_v, tgt)
        print(f"  validation (seeds {VAL_SEEDS}): loss {val_v:.4f} "
              f"(fp {parts_v['fingerprint']:.3f}, coll {agg_v['collisions']}, "
              f"TTC {agg_v['min_ttc']:.2f})")

        res = dict(theta=theta, loss=round(float(f_best), 4), parts=parts,
                   base_label=base_label, regime=args.regime,
                   seeds=list(args.seeds), val_seeds=list(VAL_SEEDS),
                   baselines=baselines, fingerprint=fp, measures=agg,
                   validation=dict(loss=round(val_v, 4), parts=parts_v,
                                   fingerprint=fp_v, measures=agg_v),
                   gt=dict(videos=tgt["videos"], n_frames=tgt["n_frames"],
                           rules=tgt["rules"], reviewed=tgt["reviewed"],
                           # the acceptance bar and its per-video breakdown are
                           # the only reference the fitted distance can be judged
                           # against, so they belong in the record, not just in
                           # the console output
                           acceptance_bar=tgt["acceptance_bar"],
                           per_video=tgt["per_video"]),
                   n_evals=int(X.shape[0] + len(trace)),
                   nm_trace=[round(float(v), 5) for v in trace],
                   frozen={n: getattr(base, n) for n in
                           ("a_pin", "zeta_lat", "v_lat_max", "a_lat_max",
                            "sigma_off", "het_lat", "k_rho", "A_pass",
                            "A_keep_right", "T_frust", "s_veto", "tau",
                            "T_hw_lo", "T_hw_hi", "s0", "A_v", "Bx_v",
                            "a_max", "b_comf", "gamma_c", "T_react", "R_front")})
        path = os.path.join(ROOT, args.out)
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(f"\nwrote {path}")
        rl.finish(metrics=dict(loss=res["loss"],
                               loss_inherited=baselines["inherited"]["loss"],
                               loss_tuned_ev=baselines["tuned_ev"]["loss"],
                               fingerprint=parts["fingerprint"],
                               val_loss=res["validation"]["loss"],
                               n_evals=res["n_evals"], **theta),
                  outputs=[args.out])


if __name__ == "__main__":
    main()
