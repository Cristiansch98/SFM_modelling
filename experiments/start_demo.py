"""Start-all launcher for the UE5.8 first-person EV demo.

One command brings up the whole demo in the right order:

    1. build the NPC brain and open its live top-down 2D window,
    2. bind + listen on the bridge socket,
    3. only THEN launch Unreal (so the editor never races ahead of the host),
    4. serve the real-time co-simulation until you close either side.

    python experiments/start_demo.py              # brain + 2D view + UE editor
    python experiments/start_demo.py --game       # standalone game, no PIE click
    python experiments/start_demo.py --no-ue      # brain only (attach UE yourself)
    python experiments/start_demo.py --no-view    # no 2D window (headless brain)
    python experiments/start_demo.py --seed 8 --params tuned

Windows users can just double-click START_DEMO.bat at the repo root.
"""
import argparse
import glob
import os
import subprocess
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emv.ue import UEBridge
from emv.runlog import RunLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT = os.path.join(ROOT, "ue_game", "EmvHighway", "EmvHighway.uproject")
#: newest-first search order for the editor binary; $UE_ROOT wins if set.
EDITOR_GLOBS = [
    r"C:\Program Files\Epic Games\UE_5.*\Engine\Binaries\Win64\UnrealEditor.exe",
    r"D:\Program Files\Epic Games\UE_5.*\Engine\Binaries\Win64\UnrealEditor.exe",
    r"D:\Epic Games\UE_5.*\Engine\Binaries\Win64\UnrealEditor.exe",
]


def find_editor(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    ue_root = os.environ.get("UE_ROOT")
    if ue_root:
        cand = os.path.join(ue_root, "Engine", "Binaries", "Win64",
                            "UnrealEditor.exe")
        if os.path.isfile(cand):
            return cand
    hits: list[str] = []
    for pat in EDITOR_GLOBS:
        hits.extend(glob.glob(pat))
    return sorted(hits)[-1] if hits else None       # highest UE_5.x


def launch_ue(editor: str, project: str, game: bool) -> subprocess.Popen:
    """-game = standalone windowed play (drops you straight into driving);
    otherwise the editor opens and you press Play (PIE) yourself."""
    args = [editor, project, "-culture=en"]
    if game:
        args += ["-game", "-windowed", "-ResX=1600", "-ResY=900"]
    print(f"[start] launching Unreal: {os.path.basename(editor)} "
          f"{'(-game)' if game else '(editor, press Play)'}", flush=True)
    return subprocess.Popen(args, cwd=os.path.dirname(project))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--mode", default="physics",
                    choices=["physics", "bluelight", "none"],
                    help="who decides how traffic reacts: physics = this "
                         "project's force model (default); bluelight = SUMO's "
                         "native rescue-lane device; none = no yielding")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--params", default="game",
                    choices=["game", "surrogate", "tuned", "default"],
                    help="force-model preset (--mode physics only); "
                         "game = surrogate retuned for a human-driven EV")
    ap.add_argument("--warmup", type=float, default=90.0,
                    help="seconds of SUMO traffic to build before the EV joins "
                         "(--mode bluelight/none only)")
    ap.add_argument("--window", type=float, default=600.0,
                    help="metres ahead/behind the EV to stream NPCs for")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="brain tick rate (= model dt). 0 keeps the preset's dt")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many simulated seconds")
    ap.add_argument("--finite", action="store_true",
                    help="disable endless traffic recycling (endless is the default here)")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="don't open the live 2D window")
    ap.add_argument("--view-fps", type=float, default=12.0)
    ap.add_argument("--view-inproc", action="store_true",
                    help="run the 2D view in the brain process (costs real-time "
                         "pacing; only useful for debugging the viewer)")
    ap.add_argument("--no-ue", dest="ue", action="store_false",
                    help="don't launch Unreal; just host the brain")
    ap.add_argument("--game", action="store_true",
                    help="launch standalone -game instead of the editor")
    ap.add_argument("--editor-exe", default=None,
                    help="path to UnrealEditor.exe (else $UE_ROOT / autodetect)")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    editor = None
    if args.ue:
        editor = find_editor(args.editor_exe)
        if editor is None:
            sys.exit("[start] could not find UnrealEditor.exe - pass --editor-exe "
                     "or set UE_ROOT (or use --no-ue and start Unreal yourself)")
        if not os.path.isfile(args.project):
            sys.exit(f"[start] project not found: {args.project}")

    if args.mode == "physics":
        bridge = UEBridge(seed=args.seed, params=args.params, window=args.window,
                          port=args.port, host=args.host,
                          endless=not args.finite, hz=args.hz or None)
    else:
        from emv.ue.sumo_server import SumoUEBridge
        bridge = SumoUEBridge(mode=args.mode, seed=args.seed, window=args.window,
                              port=args.port, host=args.host, hz=args.hz or None,
                              warmup=args.warmup)
    print(f"[start] mode={args.mode}"
          f"{' params=' + args.params if args.mode == 'physics' else ''}"
          f" seed={args.seed} hz={args.hz or 'preset'}", flush=True)

    proc: subprocess.Popen | None = None
    started = threading.Event()

    def on_listening():
        """Called the moment the socket accepts - safe to start Unreal now."""
        global proc
        if editor is not None and not started.is_set():
            started.set()
            proc = launch_ue(editor, args.project, args.game)

    with RunLogger("ue_play",
                   note=f"start_demo mode={args.mode} params={args.params} "
                        f"seed={args.seed} "
                        f"ue={'game' if args.game else 'editor' if args.ue else 'none'}",
                   params=dict(mode=args.mode, seed=args.seed, params=args.params,
                               hz=args.hz, window=args.window,
                               endless=not args.finite,
                               view=args.view, ue=args.ue, game=args.game)) as rl:
        try:
            if args.view:
                from emv.ue.liveview import serve_with_view
                m = serve_with_view(bridge, duration=args.duration,
                                    fps=args.view_fps, on_listening=on_listening,
                                    in_process=args.view_inproc)
            else:
                m = bridge.serve(duration=args.duration, on_listening=on_listening)
        finally:
            if proc is not None and proc.poll() is None:
                # the brain stopped first (window closed / Ctrl-C): leave the
                # editor alive so nothing unsaved is lost, but say so.
                print("[start] brain stopped; Unreal is still running "
                      f"(pid {proc.pid}) - close it yourself", flush=True)
        if m:
            print(f"\nserved {m['steps']} steps / {m['sim_time']:.1f} s sim in "
                  f"{m['wall_s']:.1f} s wall; EV mean {m['ev_mean_kmh']} km/h",
                  flush=True)
        else:
            print("\n[start] session ended before any frame was served", flush=True)
        rl.finish(metrics=m)
