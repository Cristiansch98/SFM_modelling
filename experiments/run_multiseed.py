"""Multi-seed experiment program for the paper:

  * 10 seeds x {overtake, jam} x {model, baseline}  -> Table I mean +- s.d.
  * pooled yield-onset distances across seeds
  * 10-seed pooled bow wave (paper figure)
  * 3-seed-averaged phase sweep (ratio: mean, TTC: min = conservative)
  * compare figure with 95% CI error bars (paper size)

Writes out/multiseed.json, out/phase_ms.npz, out/paper/{compare,bowwave,phase}.png
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import tuned_params
from emv.scenarios import make_overtake, make_jam
from emv.metrics import evaluate
from emv import viz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
POUT = os.path.join(OUT, "paper")
os.makedirs(POUT, exist_ok=True)

SEEDS = list(range(1, 11))
KEYS = ["ev_mean_speed", "ev_speed_ratio", "t_clear", "min_ttc", "collisions",
        "react_dist_mean", "p95_decel", "p95_alat", "oscillation", "disruption"]


def paper_fig_patch():
    import matplotlib.pyplot as plt
    orig = viz._fig
    def pf(w, h, pal=viz.LIGHT):
        fig = plt.figure(figsize=(w * 0.62, h * 0.66), dpi=200)
        fig.patch.set_facecolor(pal["page"])
        return fig
    viz._fig = pf
    return orig


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), lo=float("nan"),
                    hi=float("nan"), n=0)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                lo=float(v.min()), hi=float(v.max()), n=int(v.size))


def main():
    p = tuned_params()
    t0 = time.time()
    per = {c: {k: [] for k in KEYS} for c in
           ("overtake", "overtake_base", "jam", "jam_base")}
    onsets = {"overtake": [], "jam": []}
    hists_overtake = []

    for s in SEEDS:
        for cond, build, kw, T, stop in [
            ("overtake", make_overtake, dict(seed=s), 78.0, 2320.0),
            ("overtake_base", make_overtake, dict(seed=s, yielding=False), 78.0, 2320.0),
            ("jam", make_jam, dict(seed=s), 70.0, 700.0),
            ("jam_base", make_jam, dict(seed=s, yielding=False), 70.0, 700.0),
        ]:
            h = build(p=p, **kw).run(T, rec_dt=0.1, stop_when_ev_x=stop)
            m = evaluate(h)
            for k in KEYS:
                per[cond][k].append(m[k])
            if cond == "overtake":
                onsets["overtake"].extend(m["react_dists"].tolist())
                hists_overtake.append(h)
            elif cond == "jam":
                onsets["jam"].extend(m["react_dists"].tolist())
        print(f"seed {s} done  [{time.time()-t0:.0f}s]", flush=True)

    summary = {c: {k: agg(v) for k, v in d.items()} for c, d in per.items()}
    for c in per:
        summary[c]["collisions_total"] = int(np.nansum(per[c]["collisions"]))
    summary["onsets_pooled"] = {sc: agg(v) for sc, v in onsets.items()}
    summary["n_seeds"] = len(SEEDS)

    # ---- phase sweep over 3 seeds --------------------------------------
    densities = np.array([8, 12, 16, 20, 24, 28, 32, 36, 40], float)
    Acs = np.array([0.35, 0.7, 1.05, 1.4, 2.1, 2.8, 4.2, 5.6, 7.0])
    pseeds = (5, 6, 7)
    ratios = np.zeros((len(pseeds), Acs.size, densities.size))
    ttcs = np.full_like(ratios, 9.9)
    for si, ps in enumerate(pseeds):
        for j, dnn in enumerate(densities):
            for i, ac in enumerate(Acs):
                pp = p.copy(A_c=float(ac), dt=0.06)
                sim = make_overtake(seed=ps, p=pp, density=float(dnn), road_len=1300.0)
                h = sim.run(50.0, rec_dt=0.15, stop_when_ev_x=1250.0)
                m = evaluate(h)
                ratios[si, i, j] = m["ev_speed_ratio"]
                ttcs[si, i, j] = m["min_ttc"] if np.isfinite(m["min_ttc"]) else 9.9
        print(f"phase seed {ps} done  [{time.time()-t0:.0f}s]", flush=True)
    ratio_mean = ratios.mean(axis=0)
    ttc_min = ttcs.min(axis=0)
    np.savez(os.path.join(OUT, "phase_ms.npz"), densities=densities, Acs=Acs,
             ratio=ratio_mean, ratio_sd=ratios.std(axis=0, ddof=1), ttc=ttc_min,
             seeds=np.array(pseeds))
    summary["phase_seeds"] = list(pseeds)

    with open(os.path.join(OUT, "multiseed.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote out/multiseed.json", flush=True)

    # ---- paper figures ---------------------------------------------------
    orig = paper_fig_patch()
    # compare with 95% CI error bars (t ~ 2.262 for n=10)
    tcrit = 2.262
    def ci(c):
        a = summary[c]["ev_mean_speed"]
        return tcrit * a["sd"] / np.sqrt(a["n"]) * 3.6
    mk = lambda c: dict(ev_mean_speed=summary[c]["ev_mean_speed"]["mean"])
    viz.plot_compare(
        [dict(scenario="motorway overtake", baseline=mk("overtake_base"),
              model=mk("overtake"), v0_ev=36.0,
              base_err=ci("overtake_base"), model_err=ci("overtake")),
         dict(scenario="rescue lane in jam", baseline=mk("jam_base"),
              model=mk("jam"), v0_ev=9.0,
              base_err=ci("jam_base"), model_err=ci("jam"))],
        os.path.join(POUT, "compare.png"))
    viz.plot_bowwave(hists_overtake, os.path.join(POUT, "bowwave.png"))
    d = np.load(os.path.join(OUT, "phase_ms.npz"))
    viz.plot_phase(d["densities"], d["Acs"], d["ratio"], d["ttc"], p,
                   os.path.join(POUT, "phase.png"))
    viz._fig = orig

    # ---- console table for the paper ------------------------------------
    def row(k, f=1, scale=1.0):
        out = []
        for c in ("overtake_base", "overtake", "jam_base", "jam"):
            a = summary[c][k]
            out.append(f"{a['mean']*scale:.{f}f}+-{a['sd']*scale:.{f}f}")
        return out
    print("\n--- Table I data (mean +- sd over 10 seeds) ---")
    for k, f, sc in [("ev_mean_speed", 1, 1), ("ev_speed_ratio", 1, 100),
                     ("t_clear", 1, 1), ("min_ttc", 1, 1), ("p95_decel", 2, 1),
                     ("oscillation", 2, 1)]:
        print(f"{k:>15s}: " + "  ".join(row(k, f, sc)))
    print("collisions_total:", {c: summary[c]["collisions_total"] for c in per})
    print("onsets pooled:", {sc: (round(a['mean'],1), round(a['sd'],1), a['n'])
                             for sc, a in summary["onsets_pooled"].items()})
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    from emv.runlog import RunLogger
    with RunLogger("multiseed", note=f"{len(SEEDS)} seeds + 3-seed phase") as rl:
        main()
        try:
            import json as _json
            s = _json.load(open(os.path.join(OUT, "multiseed.json")))
            rl.finish(metrics={c: dict(v=s[c]["ev_mean_speed"]["mean"],
                                       sd=s[c]["ev_mean_speed"]["sd"],
                                       coll=s[c]["collisions_total"])
                               for c in ("overtake", "overtake_base", "jam", "jam_base")},
                      outputs=["out/multiseed.json", "out/phase_ms.npz",
                               "out/paper/compare.png", "out/paper/bowwave.png",
                               "out/paper/phase.png"])
        except Exception:
            pass
