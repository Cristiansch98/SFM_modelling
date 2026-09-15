"""Survey benchmark: three EV-yielding setups on identical traffic, scored
against real ambulance interactions (the dataset-pipeline ground truth).

The three conditions of the upcoming human survey. Everything about their
setup is identical - same network (3-lane, 3 km, 100 km/h), same demand
(1700 veh/h/lane), same seeds, same EV type/departure, same 0.1 s steps,
same actuation path where applicable - ONLY the yielding mechanism differs:

  A force     - this framework: physics fields + perception/urgency model
                via the hybrid TraCI bridge (R_front 140 m, lognormal
                reaction delay, 5% non-compliers)
  B bluelight - SUMO's native bluelight/rescue-lane device,
                reactiondist 100 m (the standard baseline)
  C rule      - scripted Rettungsgasse: fixed 100 m trigger (= B's
                reactiondist), fixed 1 s delay, lane-keyed edge-hugging,
                slowdown to 60% of the limit, 100% compliance - what a
                trainer typically hard-codes
 (+ none      - null reference, no yielding; not a survey condition, it
                anchors the fingerprint scale)

Ground truth: the emergency-vehicle-dataset-pipeline's processed dashcam
run(s) (real German ambulance, A2 Autobahn, reviewed labels). Each sim
setup's 1 Hz ego-frame stream is labelled with the SAME annotator rules
that labelled the real data (emv/groundtruth.py) and reduced to a
behaviour fingerprint; the weighted fingerprint distance to the real data
is the behavioural-fidelity score reported per setup.

Writes out/survey_compare.json (+ labelled streams in
out/survey_streams.json), out/fig_survey_behaviour.png,
out/fig_survey_ev.png, and full trajectories of each setup's first seed
(out/survey_traj_<mode>.npz - render figures/videos from these with
experiments/make_survey_media.py). Self-logs kind='survey_compare'.

    python experiments/run_survey_compare.py            # 3 seeds + none
    python experiments/run_survey_compare.py --quick    # 1 seed, no none
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import tuned_params
from emv.sumo.bridge import SumoBridge
from emv.sumo import make_scenario as sc
from emv.sumo import traj as trajrec
from emv import groundtruth as gt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

SURVEY_MODES = ("force", "bluelight", "rule")
#: fixed entity colours (emv.viz categorical slots; consistent with
#: fig_sumo_compare: force=blue, bluelight=yellow; real data gets violet).
#: real_rules = the same real data relabelled by the pure kinematic rules
#: (no manual-review upgrades) - the apples-to-apples reference for sims,
#: whose streams only ever get rule labels.
ENTITY_COLOR = dict(real="violet", real_rules="orange", force="blue",
                    bluelight="yellow", rule="aqua", none=None)
ENTITY_LABEL = dict(real="real (reviewed)", real_rules="real (rules-only)",
                    force="force model (A)", bluelight="SUMO bluelight (B)",
                    rule="scripted rule (C)", none="no yielding (ref)")


# ---------------------------------------------------------------------------
def pooled_gt(gt_dir):
    """Pool all processed ground-truth videos (ids prefixed per video)."""
    frames, videos = [], gt.gt_videos(gt_dir)
    for vd in videos:
        tag = os.path.basename(vd)
        for fr in gt.load_gt_stream(vd):
            for v in fr["vehicles"]:
                v["id"] = f"{tag}:{v['id']}"
            frames.append(fr)
    return frames, [os.path.basename(v) for v in videos]


def run_mode(mode, seeds, p, record_traj=True):
    """Run one setup over all seeds; label its ego streams like the GT."""
    res = dict(mode=mode, seeds=list(seeds), per_seed={}, pooled_frames=[])
    for seed in seeds:
        t0 = time.time()
        m = SumoBridge(mode, p=p, seed=seed, out_dir=os.path.join(OUT, "sumo"),
                       record_traj=record_traj and seed == seeds[0]).run()
        if "traj" in m:                         # full trajectories, seed[0]
            res["traj_path"] = trajrec.save_traj(
                os.path.join(OUT, f"survey_traj_{mode}.npz"), m.pop("traj"),
                meta=dict(mode=mode, seed=seed, n_lanes=sc.N_LANES,
                          lane_w=sc.LANE_W, edge_len=sc.EDGE_LEN,
                          v_max=sc.V_MAX, ev_depart=sc.EV_DEPART,
                          flow=sc.FLOW_PER_LANE))
            print(f"  {mode:10s} seed {seed}: wrote "
                  f"{os.path.relpath(res['traj_path'], ROOT)}", flush=True)
        labelled = gt.annotate_stream(m["ego_frames"], n_lanes=sc.N_LANES,
                                      lane_width=sc.LANE_W)
        for fr in labelled:                     # avoid id collisions in pool
            for v in fr["vehicles"]:
                v["id"] = f"{seed}:{v['id']}"
        res["pooled_frames"].extend(labelled)
        res["per_seed"][seed] = dict(
            seg_speed=m.get("seg_speed"), seg_time=m.get("seg_time"),
            clearance_mean=m.get("clearance_mean"),
            collisions=m["collisions"], wall_s=round(time.time() - t0, 1))
        if seed == seeds[0]:                    # traces for the figure
            res["t_trace"] = m["t_trace"]
            res["ev_v_trace"] = m["ev_v_trace"]
        print(f"  {mode:10s} seed {seed}: "
              f"{(m.get('seg_speed') or 0) * 3.6:6.1f} km/h  "
              f"clr {m.get('clearance_mean', float('nan')):5.1f} m  "
              f"coll {m['collisions']}  [{time.time() - t0:.0f}s]", flush=True)
    res["fingerprint"] = gt.fingerprint(res["pooled_frames"])
    return res


# ---------------------------------------------------------------------------
def _bar_group(ax, cats, entities, values, colors, pal, ylabel, title,
               as_pct=True, tick_labels=None):
    """Grouped bars, fixed entity colours, direct value labels."""
    n = len(entities)
    w = 0.8 / n
    vmax = 0.0
    for j, ent in enumerate(entities):
        xs = np.arange(len(cats)) + (j - (n - 1) / 2) * w
        vs = [values[ent][c] * (100.0 if as_pct else 1.0) for c in cats]
        vmax = max(vmax, np.nanmax(vs))
        ax.bar(xs, vs, width=w * 0.92, color=colors[ent],
               label=ENTITY_LABEL[ent])
        for x, v in zip(xs, vs):
            if np.isfinite(v):
                ax.annotate(f"{v:.0f}" if as_pct else f"{v:.1f}",
                            (x, v), ha="center", va="bottom", fontsize=7,
                            color=pal["ink2"])
    ax.set_xticks(range(len(cats)), tick_labels or cats, fontsize=8.5)
    ax.set_ylim(0, vmax * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", visible=False)


def _dot_range(ax, entities, med, p90, colors, pal, xlabel, title):
    """Median dot + p90 whisker per entity, direct-labelled."""
    ys = np.arange(len(entities))[::-1]
    for y, ent in zip(ys, entities):
        m, q = med.get(ent, float("nan")), p90.get(ent, float("nan"))
        if not np.isfinite(m):
            ax.annotate("no yields", (0.02, y), ha="left", va="center",
                        fontsize=8, color=pal["muted"],
                        xycoords=("axes fraction", "data"))
            continue
        ax.plot([m, q], [y, y], color=colors[ent], lw=2, solid_capstyle="round")
        ax.plot([m], [y], marker="o", ms=8, color=colors[ent],
                mec=pal["surface"], mew=1.2)
        ax.plot([q], [y], marker="|", ms=9, color=colors[ent], mew=2)
        ax.annotate(f"{m:.2g}", (m, y + 0.28), ha="center", fontsize=7.5,
                    color=pal["ink2"])
    ax.set_yticks(ys, [ENTITY_LABEL[e] for e in entities], fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title + "  (dot = median, tick = p90)", fontsize=10,
                 loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(entities) - 0.4)


def plot_behaviour(fps, entities, path):
    import matplotlib
    matplotlib.use("Agg")
    from emv.viz import _fig, _style_ax, LIGHT, CAT_LIGHT

    pal = LIGHT
    colors = {e: (CAT_LIGHT[ENTITY_COLOR[e]] if ENTITY_COLOR[e] else
                  pal["axis"]) for e in entities}
    fig = _fig(11.5, 8.0, pal)
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
    for a in (ax1, ax2, ax3, ax4):
        _style_ax(a, pal)

    cats = ["yielded", "failed_to_yield", "braked_abruptly", "normal"]
    vals = {e: dict(yielded=fps[e]["obs_yielded"],
                    failed_to_yield=fps[e]["obs_failed"],
                    braked_abruptly=fps[e]["obs_braked"],
                    normal=fps[e]["obs_normal"]) for e in entities}
    _bar_group(ax1, cats, entities, vals, colors, pal,
               "share of observations (%)",
               "Behaviour labels within 50 m - same annotator for all",
               tick_labels=["yielded", "failed", "braked", "normal"])

    tcats = ["ever yielded", "failed (never yielded)"]
    tvals = {e: {"ever yielded": fps[e]["trk_yielded"],
                 "failed (never yielded)": fps[e]["trk_failed"]}
             for e in entities}
    _bar_group(ax2, tcats, entities, tvals, colors, pal,
               "share of tracks (%)",
               "Per-vehicle outcome (tracks with >= 3 frames in range)",
               tick_labels=["ever yielded", "never yielded"])

    _dot_range(ax3, entities,
               {e: fps[e]["lat_speed_med"] for e in entities},
               {e: fps[e]["lat_speed_p90"] for e in entities},
               colors, pal, "|lateral speed| while yielding (m/s)",
               "How briskly vehicles move aside")
    _dot_range(ax4, entities,
               {e: fps[e]["onset_dist_med"] for e in entities},
               {e: fps[e]["onset_dist_p90"] for e in entities},
               colors, pal, "distance to EV at first 'yielded' label (m)",
               "How early vehicles react")

    fig.suptitle("Yielding behaviour: real ambulance data vs the three "
                 "survey setups", fontsize=12, color=pal["ink"], x=0.02,
                 y=0.995, ha="left")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, frameon=False,
               fontsize=7.5, labelcolor=pal["ink2"],
               bbox_to_anchor=(0.99, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", path, flush=True)


def plot_ev(results, modes, path):
    import matplotlib
    matplotlib.use("Agg")
    from emv.viz import _fig, _style_ax, LIGHT, CAT_LIGHT
    from emv.sumo.bridge import SEG

    pal = LIGHT
    colors = {m: (CAT_LIGHT[ENTITY_COLOR[m]] if ENTITY_COLOR[m] else
                  pal["axis"]) for m in modes}
    fig = _fig(11.0, 4.2, pal)
    ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.4, 1])
    for a in (ax1, ax2):
        _style_ax(a, pal)
    for m in modes:
        r = results[m]
        t = np.array(r["t_trace"]) - r["t_trace"][0]
        ax1.plot(t, np.array(r["ev_v_trace"]) * 3.6, color=colors[m],
                 lw=1.9, label=ENTITY_LABEL[m])
    ax1.set_xlabel("time since EV departure (s)")
    ax1.set_ylabel("EV speed (km/h)")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False,
               labelcolor=pal["ink2"])
    ax1.set_title("EV speed, identical traffic (seed "
                  f"{results[modes[0]]['seeds'][0]})", fontsize=10.5,
                  loc="left")

    means, sds = [], []
    for m in modes:
        v = [s["seg_speed"] * 3.6 for s in results[m]["per_seed"].values()
             if s["seg_speed"]]
        means.append(np.mean(v) if v else float("nan"))
        sds.append(np.std(v) if len(v) > 1 else 0.0)
    ax2.bar(range(len(modes)), means, yerr=sds, capsize=3,
            color=[colors[m] for m in modes], width=0.62,
            error_kw=dict(ecolor=pal["ink2"], lw=1))
    for i, v in enumerate(means):
        ax2.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom",
                     fontsize=10, color=pal["ink"])
    ax2.set_xticks(range(len(modes)),
                   [ENTITY_LABEL[m].split(" (")[0] for m in modes],
                   fontsize=8)
    ax2.set_ylabel(f"mean speed, x = {SEG[0]:.0f}-{SEG[1]:.0f} m (km/h)")
    n_seeds = len(results[modes[0]]["seeds"])
    ax2.set_title(f"Measured segment, {n_seeds} seed"
                  f"{'s' if n_seeds > 1 else ''}", fontsize=10.5, loc="left")
    ax2.grid(axis="x", visible=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", path, flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--gt-dir", default=gt.PIPELINE_OUT)
    ap.add_argument("--quick", action="store_true",
                    help="one seed, survey modes only")
    ap.add_argument("--no-none", action="store_true",
                    help="skip the no-yielding null reference")
    ap.add_argument("--no-traj", action="store_true",
                    help="skip full-trajectory recording (seed[0] per mode)")
    ap.add_argument("--flow", type=int, default=sc.FLOW_PER_LANE,
                    help="background demand (veh/h/lane), default %(default)s")
    args = ap.parse_args()
    seeds = args.seeds[:1] if args.quick else args.seeds
    sc.FLOW_PER_LANE = args.flow
    modes = list(SURVEY_MODES)
    if not (args.quick or args.no_none):
        modes.append("none")

    from emv.runlog import RunLogger
    with RunLogger("survey_compare",
                   note=f"survey setups {'+'.join(modes)} vs dashcam GT",
                   params=dict(seeds=",".join(map(str, seeds)),
                               flow=args.flow, gt_dir=args.gt_dir)) as rl:
        gt_frames, gt_names = pooled_gt(args.gt_dir)
        if not gt_frames:
            raise SystemExit(f"no ground-truth output found in {args.gt_dir}")
        fp_gt = gt.fingerprint(gt_frames)
        fp_gt_rules = gt.fingerprint(gt.reannotate(gt_frames))
        print(f"ground truth: {len(gt_names)} video(s), "
              f"{fp_gt['n_frames']} s, {fp_gt['n_obs']} obs <= 50 m, "
              f"{fp_gt['n_tracks']} tracks", flush=True)

        p = tuned_params()
        results = {}
        for mode in modes:
            results[mode] = run_mode(mode, seeds, p,
                                     record_traj=not args.no_traj)

        fps = {"real": fp_gt, "real_rules": fp_gt_rules}
        dists, dists_reviewed = {}, {}
        for mode in modes:
            fps[mode] = results[mode]["fingerprint"]
            # primary fidelity: same annotator on both sides (rules-only);
            # secondary: against the human-reviewed labels
            dists[mode] = gt.fingerprint_distance(fps[mode], fp_gt_rules)
            dists_reviewed[mode] = gt.fingerprint_distance(fps[mode], fp_gt)

        print("\nbehaviour fingerprints (obs shares within 50 m):", flush=True)
        hdr = ("setup", "yield", "fail", "brake", "onset_med",
               "latv_med", "fid_rules", "fid_revw")
        print(f"{hdr[0]:>12s} {hdr[1]:>6s} {hdr[2]:>6s} {hdr[3]:>6s} "
              f"{hdr[4]:>9s} {hdr[5]:>8s} {hdr[6]:>9s} {hdr[7]:>8s}")
        for e in ["real", "real_rules"] + modes:
            f = fps[e]
            d1 = dists.get(e, {}).get("total")
            d2 = dists_reviewed.get(e, {}).get("total")
            print(f"{e:>12s} {f['obs_yielded']:6.2f} {f['obs_failed']:6.2f} "
                  f"{f['obs_braked']:6.2f} {f['onset_dist_med']:9.1f} "
                  f"{f['lat_speed_med']:8.2f} "
                  f"{'-' if d1 is None else format(d1, '9.3f')} "
                  f"{'-' if d2 is None else format(d2, '8.3f')}", flush=True)

        payload = dict(
            gt=dict(videos=gt_names, fingerprint=fp_gt,
                    fingerprint_rules_only=fp_gt_rules, dir=args.gt_dir),
            modes={m: dict(per_seed={str(k): v for k, v in
                                     results[m]["per_seed"].items()},
                           fingerprint=fps[m], distance=dists[m],
                           distance_reviewed=dists_reviewed[m],
                           t_trace=results[m]["t_trace"],
                           ev_v_trace=results[m]["ev_v_trace"])
                   for m in modes},
            setup=dict(seeds=seeds, n_lanes=sc.N_LANES, lane_w=sc.LANE_W,
                       flow_per_lane=args.flow, edge_len=sc.EDGE_LEN,
                       ev_depart=sc.EV_DEPART, sim_end=sc.SIM_END,
                       identical="network, demand, seeds, EV type/departure,"
                                 " step length, ego recorder, annotator;"
                                 " only the yielding mechanism differs"))
        with open(os.path.join(OUT, "survey_compare.json"), "w") as fh:
            json.dump(payload, fh)
        with open(os.path.join(OUT, "survey_streams.json"), "w") as fh:
            json.dump({m: results[m]["pooled_frames"] for m in modes}, fh)

        entities = ["real", "real_rules"] + modes
        plot_behaviour(fps, entities, os.path.join(OUT,
                                                   "fig_survey_behaviour.png"))
        plot_ev(results, modes, os.path.join(OUT, "fig_survey_ev.png"))

        rl.finish(
            metrics=dict(
                gt=dict(obs=fp_gt["n_obs"], trk_yield=fp_gt["trk_yielded"],
                        onset=fp_gt["onset_dist_med"]),
                **{m: dict(
                    kmh=float(np.mean([s["seg_speed"] * 3.6 for s in
                                       results[m]["per_seed"].values()
                                       if s["seg_speed"]] or [float("nan")])),
                    coll=sum(s["collisions"] for s in
                             results[m]["per_seed"].values()),
                    fid=dists[m]["total"]) for m in modes}),
            outputs=["out/survey_compare.json", "out/survey_streams.json",
                     "out/fig_survey_behaviour.png", "out/fig_survey_ev.png"]
                    + [results[m]["traj_path"] for m in modes
                       if "traj_path" in results[m]])
