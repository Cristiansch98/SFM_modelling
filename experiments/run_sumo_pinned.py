"""SUMO bridge with the EV pinned to its lane: the emergency vehicle just
drives (laneChangeMode 0 - car-following only, never a lane change) and ALL
corridor-making comes from the surrounding traffic. `--mode force` (default)
uses the force-model yielding; `--mode bluelight` uses SUMO's native
rescue-lane device, which then only pushes the other cars aside while the
ego never moves sideways. This is the `ev_fixed_lane` ablation of the
bridge: the standard conditions additionally give the EV SUMO's native lane
changing to weave around stragglers (PROJECT_LOG gotcha 3).

Runs the survey seeds (42/43/44), records full trajectories for the first
seed, and renders the same media as the survey cases side by side with the
matching free-lane-change run (<mode> = force | bluelight):

  out/sumo_pinned_<mode>.json                        per-seed metrics
  out/survey_traj_<mode>_pinned.npz                  full trajectories, seed 42
  out/fig_survey_traj_spacetime_<mode>_pinned.png    free vs pinned, space-time
  out/fig_survey_traj_lateral_<mode>_pinned.png      free vs pinned, corridor
  out/anim_survey_<mode>_pinned.gif                  dark top-down video

Self-logs kind='sumo_pinned'.

    python experiments/run_sumo_pinned.py                    # force, all
    python experiments/run_sumo_pinned.py --mode bluelight   # bluelight
    python experiments/run_sumo_pinned.py --no-video --quick # fast check
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from emv.params import tuned_params
from emv.sumo.bridge import SumoBridge
from emv.sumo import make_scenario as sc
from emv.sumo import traj as trajrec
import make_survey_media as media

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

SEEDS = [42, 43, 44]
SCALARS = ("seg_speed", "seg_time", "seg_complete", "clearance_mean",
           "ev_mean_speed", "collisions")


def main(args):
    p = tuned_params()
    seeds = SEEDS[:1] if args.quick else SEEDS
    tag = f"{args.mode}_pinned"
    res = {"mode": tag, "seeds": seeds, "per_seed": {}}
    traj_path = os.path.join(OUT, f"survey_traj_{tag}.npz")

    for seed in seeds:
        t0 = time.time()
        m = SumoBridge(args.mode, p=p, seed=seed,
                       out_dir=os.path.join(OUT, "sumo"),
                       record_traj=(seed == seeds[0]),
                       ev_fixed_lane=True).run()
        if "traj" in m:
            trajrec.save_traj(traj_path, m.pop("traj"),
                              meta=dict(mode=tag, seed=seed,
                                        n_lanes=sc.N_LANES, lane_w=sc.LANE_W,
                                        edge_len=sc.EDGE_LEN, v_max=sc.V_MAX,
                                        ev_depart=sc.EV_DEPART,
                                        flow=sc.FLOW_PER_LANE))
            print(f"  wrote {os.path.relpath(traj_path, ROOT)}", flush=True)
        row = {k: m.get(k) for k in SCALARS}
        res["per_seed"][seed] = row
        kmh = 3.6 * row["seg_speed"] if row["seg_speed"] else float("nan")
        print(f"  seed {seed}: {kmh:.1f} km/h in segment, "
              f"clearance {row['clearance_mean'] or float('nan'):.0f} m, "
              f"{row['collisions']} collisions  [{time.time()-t0:.0f}s]",
              flush=True)

    kmhs = [3.6 * r["seg_speed"] for r in res["per_seed"].values()
            if r["seg_speed"]]
    res["seg_kmh_mean"] = float(np.mean(kmhs)) if kmhs else float("nan")
    res["collisions_total"] = int(sum(r["collisions"]
                                      for r in res["per_seed"].values()))

    # context: the moving-LC force condition + references from the survey run
    ref_path = os.path.join(OUT, "survey_compare.json")
    if os.path.exists(ref_path):
        with open(ref_path) as fh:
            ref = json.load(fh)["modes"]
        res["reference"] = {}
        for m in ("force", "bluelight", "rule", "none"):
            if m not in ref:
                continue
            ps = ref[m]["per_seed"].values()
            ks = [3.6 * s["seg_speed"] for s in ps if s.get("seg_speed")]
            res["reference"][m] = dict(
                seg_kmh_mean=float(np.mean(ks)) if ks else float("nan"),
                collisions=int(sum(s["collisions"] for s in ps)))

    json_path = os.path.join(OUT, f"sumo_pinned_{args.mode}.json")
    with open(json_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {os.path.relpath(json_path, ROOT)}", flush=True)

    # ---- media: same renderers as the survey cases ----------------------
    outputs = [json_path, traj_path]
    trs = {}
    free_path = os.path.join(OUT, f"survey_traj_{args.mode}.npz")
    if os.path.exists(free_path):
        trs[args.mode] = trajrec.load_traj(free_path)
    trs[tag] = trajrec.load_traj(traj_path)

    if not args.no_figs:
        f1 = os.path.join(OUT, f"fig_survey_traj_spacetime_{tag}.png")
        f2 = os.path.join(OUT, f"fig_survey_traj_lateral_{tag}.png")
        media.fig_spacetime(trs, f1)
        media.fig_lateral(trs, f2)
        outputs += [f1, f2]
    if not args.no_video:
        t0 = time.time()
        gif = os.path.join(OUT, f"anim_survey_{tag}.gif")
        n = media.render_video(trs[tag], p, gif,
                               frame_dt=args.frame_dt, fps=args.fps)
        print(f"video: {n} frames in {time.time()-t0:.0f}s", flush=True)
        outputs.append(gif)
    return res, outputs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="force", choices=("force", "bluelight"))
    ap.add_argument("--quick", action="store_true", help="seed 42 only")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--frame-dt", type=float, default=0.3)
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()

    from emv.runlog import RunLogger
    with RunLogger("sumo_pinned", params=dict(mode=args.mode),
                   note=f"{args.mode} bridge with EV pinned to its lane (no "
                        "EV lane changes; traffic makes the corridor) + media") as rl:
        res, outputs = main(args)
        rl.finish(
            metrics=dict(
                seg_kmh_mean=res["seg_kmh_mean"],
                collisions=res["collisions_total"],
                ref_free_kmh=(res.get("reference", {})
                              .get(args.mode, {}).get("seg_kmh_mean")),
                seeds=len(res["seeds"])),
            outputs=outputs)
