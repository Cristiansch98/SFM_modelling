"""Three-way SUMO comparison on the same seeded traffic:

  none      - EV without special rights (lower bound)
  bluelight - SUMO's native rescue-lane device [17] (reference behaviour)
  force     - this framework via the hybrid TraCI bridge

Writes out/sumo_compare.json and out/fig_sumo_compare.png.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import tuned_params
from emv.sumo.bridge import SumoBridge, SEG

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from emv.viz import _fig, _style_ax, LIGHT, CAT_LIGHT, EV_DARK

    pal = LIGHT
    fig = _fig(11.0, 4.2, pal)
    ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.4, 1])
    for a in (ax1, ax2):
        _style_ax(a, pal)
    colors = {"none": pal["axis"], "bluelight": CAT_LIGHT["yellow"],
              "force": CAT_LIGHT["blue"]}
    labels = {"none": "no yielding", "bluelight": "SUMO bluelight [17]",
              "force": "force model (bridge)"}
    for mode, m in results.items():
        t = np.array(m["t_trace"]) - m["t_trace"][0]
        ax1.plot(t, np.array(m["ev_v_trace"]) * 3.6, color=colors[mode],
                 lw=1.9, label=labels[mode])
    ax1.set_xlabel("time since EV departure (s)")
    ax1.set_ylabel("EV speed (km/h)")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False,
               labelcolor=pal["ink2"])
    ax1.set_title("EV speed through congested traffic (SUMO)", fontsize=10.5,
                  loc="left")

    modes = list(results)
    seg_v = [results[m].get("seg_speed", float("nan")) * 3.6 for m in modes]
    bars = ax2.bar(range(len(modes)), seg_v,
                   color=[colors[m] for m in modes], width=0.62)
    for i, v in enumerate(seg_v):
        ax2.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=10,
                     color=pal["ink"])
    ax2.set_xticks(range(len(modes)), [labels[m].split(" [")[0] for m in modes],
                   fontsize=8.5)
    ax2.set_ylabel(f"mean speed, x = {SEG[0]:.0f}-{SEG[1]:.0f} m (km/h)")
    ax2.set_title("Measured segment", fontsize=10.5, loc="left")
    ax2.grid(axis="x", visible=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", path, flush=True)


if __name__ == "__main__":
    from emv.runlog import RunLogger
    _rl = RunLogger("sumo_compare")
    p = tuned_params()
    results = {}
    for mode in ("none", "bluelight", "force"):
        t0 = time.time()
        m = SumoBridge(mode, p=p, seed=42, out_dir=os.path.join(OUT, "sumo")).run()
        results[mode] = m
        print(f"{mode:10s} seg_time {m.get('seg_time', float('nan')):6.1f}s  "
              f"seg_speed {m.get('seg_speed', float('nan'))*3.6:6.1f} km/h  "
              f"clearance {m.get('clearance_mean', float('nan')):6.1f} m  "
              f"collisions {m['collisions']}  [{time.time()-t0:.0f}s wall]",
              flush=True)
    with open(os.path.join(OUT, "sumo_compare.json"), "w") as fh:
        json.dump(results, fh)
    plot(results, os.path.join(OUT, "fig_sumo_compare.png"))
    _rl.finish(metrics={m: dict(kmh=results[m].get("seg_speed", 0) * 3.6,
                                clr=results[m].get("clearance_mean"),
                                coll=results[m]["collisions"])
                        for m in results},
               outputs=["out/sumo_compare.json", "out/fig_sumo_compare.png"])
