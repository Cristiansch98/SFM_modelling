"""Fit the social-force model as a physics-inspired SURROGATE of SUMO.

Here SUMO's native EV operation (bluelight device + sublane model, the
regulator-accepted microsimulation baseline) is treated as the ground truth,
and the standalone force framework is calibrated to reproduce its behaviour
and performance - a surrogate that can then replace SUMO where SUMO is
impractical (see 'Why this is reasonable' below).

Protocol
  1. Ground truth = the recorded SUMO bluelight runs (survey seeds 42/43/44,
     out/survey_compare.json): EV segment speed, corridor clearance, and the
     pooled behaviour fingerprint of the 1 Hz ego-frame stream - measured by
     the exact annotator/kinematics chain of the dashcam dataset pipeline.
  2. Surrogate scenario mirrors the SUMO network 1:1: straight 3 x 3.5 m
     motorway, 3 km, no shoulders, 100 km/h limit, speedFactor
     normc(0.92, 0.08) desired speeds, ~1700 veh/h/lane, EV desired speed
     min(41, 1.40 x 27.78) = 38.9 m/s entering behind the traffic. The
     corridor line is the boundary between the two leftmost lanes - the
     same place SUMO's bluelight rescue gap forms, so the surrogate matches
     the mechanism, not just the numbers.
  3. Fit the six calibration parameters (A_ev, B_ev, A_c, B_c, T_react,
     gamma_c; bounds from emv.calibrate.THETA_SPEC) by LHS + Nelder-Mead on
     a loss = fingerprint distance + |seg speed error|/10 km/h
     + 0.5 |clearance error|/30 m + collision penalty, over 2 train seeds.
  4. Validate on 3 fresh surrogate seeds, against BOTH the training ground
     truth and 2 held-out SUMO seeds (45/46, run here): the surrogate is
     accepted if its deviation from SUMO is within SUMO's own seed-to-seed
     spread.

Why SUMO-as-ground-truth is reasonable
  * SUMO's car-following/lane-change stack is independently validated and
    widely accepted; treating it as GT gives unlimited, perfectly observed,
    collision-checked trajectories, where real dashcam data is scarce
    (1 video), short-ranged (~35 m) and noisy.
  * The fingerprint chain used for fitting is the same one already used to
    compare both SUMO and this model against real dashcam data, so
    'behaviour' means the same thing in every comparison.
  * The surrogate is physics-structured (forces, not curve fits): its
    parameters stay interpretable and it extrapolates by mechanism.

What the surrogate buys (advantages)
  * Speed: one force step is a few numpy ops; no TraCI IPC, no SUMO
    process - orders of magnitude faster per simulated second, and
    embeddable directly in the UE5 trainer loop.
  * Differentiability/analytics: smooth closed-form fields give the d*
    clearance law and phase boundary analytically - usable for control,
    reward shaping, and PIDL-style learning; SUMO is a discrete black box.
  * Portability: no SUMO_HOME, no netconvert, no XML scenario files.
  * Unified behaviour: the same fields drive the standalone sim, the SUMO
    bridge and (via this fit) a SUMO-equivalent - one parameter set to
    maintain.

Outputs: out/surrogate_sumo.json (fit + validation), out/fig_surrogate_sumo.png,
out/anim_surrogate_sumo.gif. Self-logs kind='surrogate_sumo'.

    python experiments/run_surrogate_sumo.py               # full (~30 min)
    python experiments/run_surrogate_sumo.py --quick       # tiny budgets
    python experiments/run_surrogate_sumo.py --skip-fit    # reuse stored fit
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import Params, TUNED
from emv.road import Road
from emv.state import VehState, blank_state
from emv.simulate import Sim
from emv.metrics import evaluate
import emv.calibrate as cal
from emv.calibrate import latin_hypercube, nelder_mead
from emv import groundtruth as gt
from emv import viz
from emv.scenarios import make_sumo_like

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

# SUMO scenario constants mirrored (emv/sumo/make_scenario.py + bridge.SEG)
N_LANES, LANE_W, EDGE_LEN, V_MAX = 3, 3.5, 3000.0, 27.78
SEG = (400.0, 2600.0)
FLOW = 1700.0                      # veh/h/lane
EV_V0 = min(41.0, 1.40 * V_MAX)    # SUMO ev type: maxSpeed 41, speedFactor 1.4
TRAIN_SEEDS = (5, 6)               # surrogate traffic realisations for the fit
VAL_SEEDS = (7, 8, 9)
GT_HOLDOUT_SEEDS = (45, 46)        # fresh SUMO bluelight runs


# ------------------------------------------------------------ scenario
# make_sumo_like now lives in emv/scenarios.py (shared with the emv.ue game
# bridge) and is imported above - single source of truth for the GT mirror.


#: --extended adds the lane-discipline parameters to the search: SUMO's
#: bluelight produces sustained sub-lane edging, which the washboard pin
#: (all-or-nothing lane hops) cannot express at its default stiffness.
EXTRA_SPEC = [("a_pin", 0.3, 1.6), ("urgency_pin_relief", 0.2, 0.95)]


def extend_theta_spec():
    cal.THETA_SPEC.extend(EXTRA_SPEC)
    cal.NAMES = [n for n, _, _ in cal.THETA_SPEC]
    cal.LO = np.array([lo for _, lo, _ in cal.THETA_SPEC])
    cal.HI = np.array([hi for _, _, hi in cal.THETA_SPEC])


def make_params(theta) -> Params:
    """Base params mirror the GT regime, fitted params on top: SUMO's
    bluelight yields with 100 % compliance (p_noncomply=0) and its cars have
    maxSpeedLat=1.5 m/s (v_lat_max) - matching the ground truth's compliance
    and vehicle capability is part of being its surrogate, not tuning."""
    over = dict(zip(cal.NAMES, np.clip(theta, cal.LO, cal.HI)))
    return Params().copy(**over, dt=0.06, p_noncomply=0.0, v_lat_max=1.5)


# ------------------------------------------------------------- measures
#: Moved to emv/groundtruth.py on 2026-08-06 so the stage-2 EV calibration can
#: reach it without importing this experiment (and without SUMO). Re-exported
#: here under its original name: this module's public surface is unchanged.
ego_stream = gt.ego_stream


def run_stats(p: Params, seed: int):
    """One surrogate run -> (seg_kmh, clearance_mean, labelled frames, m, h)."""
    h = make_sumo_like(seed, p).run(115.0, rec_dt=0.1,
                                    stop_when_ev_x=SEG[1] + 60.0)
    m = evaluate(h)
    x_ev = h.x[:, h.ev]
    inside = (x_ev >= SEG[0]) & (x_ev <= SEG[1])
    if inside.sum() > 5:
        seg_kmh = 3.6 * float((x_ev[inside][-1] - x_ev[inside][0])
                              / max(h.t[inside][-1] - h.t[inside][0], 1e-6))
        clr = float(np.mean(m["clearance_trace"][inside]))
    else:
        seg_kmh, clr = 0.0, 0.0
    labelled = gt.annotate_stream(ego_stream(h), n_lanes=N_LANES,
                                  lane_width=LANE_W)
    for fr in labelled:
        for v in fr["vehicles"]:
            v["id"] = f"{seed}:{v['id']}"
    return seg_kmh, clr, labelled, m, h


def gt_targets():
    """Ground-truth targets from the recorded SUMO bluelight runs."""
    with open(os.path.join(OUT, "survey_compare.json")) as fh:
        bl = json.load(fh)["modes"]["bluelight"]
    ps = list(bl["per_seed"].values())
    return dict(
        seg_kmh=float(np.mean([3.6 * s["seg_speed"] for s in ps])),
        seg_kmh_per_seed=[3.6 * s["seg_speed"] for s in ps],
        clearance=float(np.mean([s["clearance_mean"] for s in ps])),
        fingerprint=bl["fingerprint"],
        ev_v_trace=bl.get("ev_v_trace"), t_trace=bl.get("t_trace"))


def score(seg_kmh, clr, fp, coll, tgt):
    d_fp = gt.fingerprint_distance(fp, tgt["fingerprint"])["total"]
    parts = dict(
        fingerprint=float(d_fp),
        speed=abs(seg_kmh - tgt["seg_kmh"]) / 10.0,
        clearance=0.5 * abs(clr - tgt["clearance"]) / 30.0,
        collisions=min(5.0 * coll, 25.0))
    return float(sum(parts.values())), parts


# ------------------------------------------------------------------ fit
def fit(tgt, n_lhs, n_nm, verbose=True):
    rng = np.random.default_rng(2026)
    X = latin_hypercube(n_lhs, rng)
    base = Params()
    X = np.vstack([X, [[TUNED.get(n, getattr(base, n)) for n in cal.NAMES]],
                   [[getattr(base, n) for n in cal.NAMES]]])
    evals = []

    def f(theta):
        p = make_params(theta)
        pooled, seg, clr, coll = [], [], [], 0
        for s in TRAIN_SEEDS:
            sk, ck, frames, m, _ = run_stats(p, s)
            pooled.extend(frames)
            seg.append(sk)
            clr.append(ck)
            coll += m["collisions"]
        fp = gt.fingerprint(pooled)
        val, _ = score(float(np.mean(seg)), float(np.mean(clr)), fp, coll, tgt)
        evals.append((list(map(float, theta)), val))
        return val

    fv = []
    for i, x in enumerate(X):
        fv.append(f(x))
        if verbose and (i + 1) % 5 == 0:
            print(f"  LHS {i+1}/{len(X)}  best {min(fv):.3f}", flush=True)
    best0 = X[int(np.argmin(fv))]
    theta, fbest, trace = nelder_mead(f, best0, maxiter=n_nm)
    return theta, fbest, dict(n_evals=len(evals),
                              loss_tuned=float(fv[len(X) - 2]),
                              loss_default=float(fv[len(X) - 1]),
                              nm_trace=[round(v, 4) for v in trace])


# ------------------------------------------------------------ validation
def validate(theta, tgt):
    p = make_params(theta)
    out = dict(per_seed={}, pooled={})
    pooled, hist0 = [], None
    for s in VAL_SEEDS:
        sk, ck, frames, m, h = run_stats(p, s)
        pooled.extend(frames)
        fp1 = gt.fingerprint(frames)
        out["per_seed"][s] = dict(
            seg_kmh=round(sk, 1), clearance=round(ck, 1),
            collisions=m["collisions"],
            fp_dist=gt.fingerprint_distance(fp1, tgt["fingerprint"])["total"])
        if hist0 is None:
            hist0 = h
    fp = gt.fingerprint(pooled)
    val, parts = score(
        float(np.mean([d["seg_kmh"] for d in out["per_seed"].values()])),
        float(np.mean([d["clearance"] for d in out["per_seed"].values()])),
        fp, sum(d["collisions"] for d in out["per_seed"].values()), tgt)
    out["pooled"] = dict(loss=round(val, 4), parts=parts,
                         fingerprint={k: (round(v, 4)
                                          if isinstance(v, float) else v)
                                      for k, v in fp.items()})
    return out, fp, hist0


def sumo_holdout(tgt):
    """Fresh SUMO bluelight seeds: how far is SUMO from ITSELF? The
    surrogate only has to sit inside this seed-to-seed spread."""
    from emv.sumo.bridge import SumoBridge
    out = {}
    pooled = []
    for s in GT_HOLDOUT_SEEDS:
        m = SumoBridge("bluelight", p=Params(), seed=s,
                       out_dir=os.path.join(OUT, "sumo")).run()
        labelled = gt.annotate_stream(m["ego_frames"], n_lanes=N_LANES,
                                      lane_width=LANE_W)
        for fr in labelled:
            for v in fr["vehicles"]:
                v["id"] = f"{s}:{v['id']}"
        pooled.extend(labelled)
        out[s] = dict(seg_kmh=round(3.6 * m.get("seg_speed", 0.0), 1),
                      clearance=round(m.get("clearance_mean", 0.0), 1),
                      collisions=m["collisions"])
        print(f"  SUMO holdout seed {s}: {out[s]['seg_kmh']} km/h", flush=True)
    fp = gt.fingerprint(pooled)
    out["fp_dist_to_train_gt"] = gt.fingerprint_distance(
        fp, tgt["fingerprint"])["total"]
    return out


# --------------------------------------------------------------- figure
def make_figure(tgt, val, fp_sur, traces_sur, path):
    import matplotlib.pyplot as plt
    pal, cat = viz.LIGHT, viz.CAT_LIGHT
    fig = viz._fig(11.5, 4.4, pal)
    ax1, ax2 = fig.subplots(1, 2)

    viz._style_ax(ax1, pal)
    if tgt["ev_v_trace"] and tgt["t_trace"]:
        t0 = tgt["t_trace"][0]
        ax1.plot(np.array(tgt["t_trace"]) - t0,
                 3.6 * np.array(tgt["ev_v_trace"]), color=pal["ink2"],
                 lw=2.2, label="SUMO bluelight (ground truth)")
    for i, (ts, vs) in enumerate(traces_sur):
        ax1.plot(ts, 3.6 * vs, color=cat["blue"], lw=1.2, alpha=0.75,
                 label="force surrogate" if i == 0 else None)
    ax1.axhline(tgt["seg_kmh"], color=pal["muted"], lw=0.9, ls=(0, (4, 4)))
    ax1.annotate("GT segment mean", (1, tgt["seg_kmh"] + 1.5), fontsize=7.5,
                 color=pal["muted"])
    ax1.set_xlabel("t since EV departure (s)")
    ax1.set_ylabel("EV speed (km/h)")
    ax1.set_title("EV progress: surrogate vs SUMO", fontsize=10.5, loc="left")
    ax1.legend(loc="lower right", fontsize=8, frameon=False,
               labelcolor=pal["ink2"])

    viz._style_ax(ax2, pal)
    ax2.grid(axis="x", visible=False)
    keys = ["obs_yielded", "obs_failed", "obs_braked", "trk_yielded",
            "onset_dist_med", "lat_speed_med"]
    lab = ["yielded\n(obs)", "failed\n(obs)", "braked\n(obs)",
           "yielded\n(tracks)", "onset\n(m/100)", "lat speed\n(m/s)"]
    sc = [1, 1, 1, 1, 0.01, 1]
    w = 0.38
    for j, (k, s) in enumerate(zip(keys, sc)):
        a, b = tgt["fingerprint"].get(k), fp_sur.get(k)
        ax2.bar(j - w / 2, (a or 0) * s, w, color=pal["muted"], zorder=3)
        ax2.bar(j + w / 2, (b or 0) * s, w, color=cat["blue"], zorder=3)
        for dx, v in ((-w / 2, a), (w / 2, b)):
            if v is not None and np.isfinite(v):
                ax2.annotate(f"{v * s:.2f}", (j + dx, v * s), fontsize=6.5,
                             xytext=(0, 2), textcoords="offset points",
                             ha="center", color=pal["ink2"])
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(lab, fontsize=7.5)
    ax2.set_title("Behaviour fingerprint: SUMO (grey) vs surrogate (blue)",
                  fontsize=10.5, loc="left")
    fig.suptitle("Force-model surrogate of SUMO's native EV operation",
                 fontsize=12, color=pal["ink"], x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    viz._save(fig, path)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny fit budgets (smoke test)")
    ap.add_argument("--skip-fit", action="store_true",
                    help="reuse theta from out/surrogate_sumo.json")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--n-lhs", type=int, default=22)
    ap.add_argument("--n-nm", type=int, default=38)
    ap.add_argument("--extended", action="store_true",
                    help="also fit lane-discipline params "
                         "(a_pin, urgency_pin_relief)")
    args = ap.parse_args()
    if args.extended:
        extend_theta_spec()
    n_lhs, n_nm = (4, 4) if args.quick else (args.n_lhs, args.n_nm)

    from emv.runlog import RunLogger
    with RunLogger("surrogate_sumo",
                   note="SFM fitted as surrogate of SUMO bluelight GT "
                        "(fingerprint+speed+clearance loss) + validation",
                   params=dict(n_lhs=n_lhs, n_nm=n_nm,
                               train_seeds=str(TRAIN_SEEDS))) as rl:
        t0 = time.time()
        tgt = gt_targets()
        print(f"GT (SUMO bluelight, seeds 42/43/44): "
              f"{tgt['seg_kmh']:.1f} km/h, clearance {tgt['clearance']:.0f} m",
              flush=True)

        jpath = os.path.join(OUT, "surrogate_sumo.json")
        if args.skip_fit and os.path.exists(jpath):
            with open(jpath) as fh:
                theta = np.array([json.load(fh)["theta"][n] for n in cal.NAMES])
            fbest, fit_info = float("nan"), dict(reused=True)
        else:
            theta, fbest, fit_info = fit(tgt, n_lhs, n_nm)
        print("theta:", {n: round(float(v), 3)
                         for n, v in zip(cal.NAMES, theta)}, flush=True)

        val, fp_sur, hist0 = validate(theta, tgt)
        holdout = sumo_holdout(tgt)

        res = dict(
            gt=dict(seg_kmh=tgt["seg_kmh"], clearance=tgt["clearance"],
                    fingerprint=tgt["fingerprint"]),
            theta=dict(zip(cal.NAMES, map(float, theta))),
            fit_loss=None if np.isnan(fbest) else round(fbest, 4),
            fit=fit_info, validation=val, sumo_holdout=holdout,
            train_seeds=list(TRAIN_SEEDS), val_seeds=list(VAL_SEEDS))
        with open(jpath, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote out/surrogate_sumo.json", flush=True)

        # figure needs per-seed EV speed traces
        p = make_params(theta)
        traces = []
        for s in VAL_SEEDS:
            h = make_sumo_like(s, p).run(115.0, rec_dt=0.2,
                                         stop_when_ev_x=SEG[1] + 60.0)
            traces.append((h.t, h.vx[:, h.ev]))
        fpath = os.path.join(OUT, "fig_surrogate_sumo.png")
        make_figure(tgt, val, fp_sur, traces, fpath)
        outputs = [jpath, fpath]

        if not args.no_anim:
            apath = os.path.join(OUT, "anim_surrogate_sumo.gif")
            viz.animate(hist0, apath)
            outputs.append(apath)

        print(f"\nsurrogate seg {np.mean([d['seg_kmh'] for d in val['per_seed'].values()]):.1f} "
              f"km/h vs GT {tgt['seg_kmh']:.1f}; "
              f"fp dist {val['pooled']['parts']['fingerprint']:.3f} "
              f"(SUMO holdout self-dist {holdout['fp_dist_to_train_gt']:.3f}); "
              f"total {time.time()-t0:.0f}s", flush=True)
        rl.finish(
            metrics=dict(
                seg_kmh=float(np.mean([d["seg_kmh"]
                                       for d in val["per_seed"].values()])),
                gt_kmh=tgt["seg_kmh"],
                fp_dist=val["pooled"]["parts"]["fingerprint"],
                fp_selfdist=holdout["fp_dist_to_train_gt"],
                fit_loss=res["fit_loss"]),
            outputs=outputs)
