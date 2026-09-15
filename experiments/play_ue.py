"""Run the Python force-model "brain" for the UE5.8 first-person EV game.

The surrounding traffic behaves exactly like out/anim_surrogate_sumo.gif; you
drive the emergency vehicle in Unreal Engine 5.8, which connects to this server
over a local TCP socket (see docs/UE_BRIDGE.md for the wire + coordinate
contract and the UE-side setup).

    python experiments/play_ue.py                 # serve on 127.0.0.1:7777
    python experiments/play_ue.py --view          # + live top-down 2D window
    python experiments/play_ue.py --seed 8 --window 500
    python experiments/play_ue.py --params tuned  # standalone calibration
    python experiments/play_ue.py --endless       # keep traffic flowing
    python experiments/play_ue.py --mock          # offline self-test (no UE) +
                                                  # out/anim_ue_mock.gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emv.ue import UEBridge
from emv.runlog import RunLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--params", default="surrogate",
                    choices=["game", "surrogate", "tuned", "default"],
                    help="surrogate = the params behind anim_surrogate_sumo.gif; "
                         "game = retuned for a human-driven EV")
    ap.add_argument("--window", type=float, default=600.0,
                    help="metres ahead/behind the EV to stream NPCs for")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--endless", action="store_true",
                    help="recycle NPCs that fall behind (continuous traffic)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many simulated seconds")
    ap.add_argument("--view", action="store_true",
                    help="open the live top-down 2D window (like anim_surrogate_sumo.gif)")
    ap.add_argument("--view-fps", type=float, default=12.0,
                    help="redraw rate of the live 2D window")
    ap.add_argument("--mock", action="store_true",
                    help="offline self-test: replay the surrogate EV, no UE")
    args = ap.parse_args()

    bridge = UEBridge(seed=args.seed, params=args.params, window=args.window,
                      port=args.port, host=args.host, endless=args.endless)

    with RunLogger("ue_play",
                   note=f"{'mock' if args.mock else 'serve'} "
                        f"params={args.params} seed={args.seed}",
                   params=dict(seed=args.seed, params=args.params,
                               window=args.window, endless=args.endless,
                               mock=args.mock)) as rl:
        if args.mock:
            gif = os.path.join(ROOT, "out", "anim_ue_mock.gif")
            m = bridge.mock(duration=args.duration, out_gif=gif)
            print(f"\nmock parity: NPC-brain seg {m['seg_kmh']} km/h vs "
                  f"surrogate {m['ref_seg_kmh']} km/h | clearance "
                  f"{m['clearance']} m vs {m['ref_clearance']} m | "
                  f"collisions {m['collisions']} vs {m['ref_collisions']} | "
                  f"{m['n_frames']} frames", flush=True)
            print(f"rendered {gif}", flush=True)
            rl.finish(metrics=m, outputs=[gif])
        else:
            if args.view:
                from emv.ue.liveview import serve_with_view
                m = serve_with_view(bridge, duration=args.duration,
                                    fps=args.view_fps)
            else:
                m = bridge.serve(duration=args.duration)
            if m:
                print(f"\nserved {m['steps']} steps / {m['sim_time']:.1f} s sim in "
                      f"{m['wall_s']:.1f} s wall; EV mean {m['ev_mean_kmh']} km/h",
                      flush=True)
            else:
                print("\nsession ended before any frame was served", flush=True)
            rl.finish(metrics=m)
