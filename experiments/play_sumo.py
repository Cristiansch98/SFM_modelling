"""Playable SUMO demo: watch the force model clear a corridor, live.

Opens sumo-gui with the hybrid TraCI bridge attached. The camera tracks the
emergency vehicle; surrounding vehicles are recoloured by awareness state
(grey unaware, amber noticed, orange yielding, teal hold; the EV is red).

    python experiments/play_sumo.py                    # force model (default)
    python experiments/play_sumo.py --mode bluelight   # SUMO native device
    python experiments/play_sumo.py --mode rule        # scripted Rettungsgasse
    python experiments/play_sumo.py --mode none        # no yielding
    python experiments/play_sumo.py --seed 7 --delay 40

sumo-gui controls while it runs:
    Ctrl+A run / resume        Ctrl+S stop (pause)
    Ctrl+D single step         mouse wheel: zoom
    Delay box (toolbar): real-time pacing, ms per 0.1 s step
    right-click a vehicle for parameters; left panel: locate objects
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emv.params import tuned_params
from emv.sumo.bridge import SumoBridge
from emv.sumo import make_scenario as sc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Playable SUMO demo of EV yielding")
    ap.add_argument("--mode", default="force",
                    choices=["force", "bluelight", "rule", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=int, default=70,
                    help="GUI pacing in ms per 0.1 s step (0 = as fast as possible)")
    ap.add_argument("--test", action="store_true",
                    help="short auto-closing run (validation)")
    args = ap.parse_args()

    if args.test:
        sc.SIM_END = 150.0
        args.delay = 0

    print(f"mode={args.mode}  seed={args.seed}", flush=True)
    print("The EV departs at t = 120 s (the toolbar 'Time' box); the camera")
    print("locks onto it. Colours: grey=unaware, amber=noticed,")
    print("orange=yielding, teal=hold, red=EV. Ctrl+S pauses, Ctrl+A resumes.")
    print("NOTE: on first launch Windows Firewall may ask about sumo-gui")
    print("(TraCI uses a local socket) - click Allow.", flush=True)
    from emv.runlog import RunLogger
    _rl = RunLogger("play_sumo", note=f"mode={args.mode} seed={args.seed}")
    m = SumoBridge(args.mode, p=tuned_params(), seed=args.seed,
                   out_dir=os.path.join(os.path.dirname(__file__), "..", "out", "sumo"),
                   gui=True, delay_ms=args.delay).run()
    _rl.finish(metrics=dict(seg_kmh=m.get("seg_speed", 0) * 3.6,
                            clearance=m.get("clearance_mean"),
                            collisions=m.get("collisions")))
    if "seg_speed" in m:
        print(f"\nEV over the 2.2 km segment: {m['seg_speed']*3.6:.1f} km/h "
              f"in {m['seg_time']:.1f} s | mean corridor clearance "
              f"{m['clearance_mean']:.0f} m | collisions {m['collisions']}")
    else:
        print("\nrun ended before the EV completed the segment", m.get("collisions", 0),
              "collisions")
