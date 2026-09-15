"""Produce all results into out/: metrics, figures, phase diagram, animations.

Usage:
    python experiments/run_all.py               # everything
    python experiments/run_all.py --no-anim     # skip the (slow) GIFs
    python experiments/run_all.py --stage figs  # just static figures
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import Params, tuned_params
from emv.scenarios import make_overtake, make_jam
from emv.metrics import evaluate, summary_line
from emv import viz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def _scalars(m):
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in m.items()
            if not isinstance(v, np.ndarray)}


def main_runs(p):
    hs, ms = {}, {}
    for key, build, kw, T, stop in [
        ("overtake", make_overtake, dict(seed=1), 78.0, 2320.0),
        ("overtake_base", make_overtake, dict(seed=1, yielding=False), 78.0, 2320.0),
        ("jam", make_jam, dict(seed=3), 70.0, 700.0),
        ("jam_base", make_jam, dict(seed=3, yielding=False), 70.0, 700.0),
    ]:
        sim = build(p=p, **kw)
        hs[key] = sim.run(T, rec_dt=0.1, stop_when_ev_x=stop)
        ms[key] = evaluate(hs[key])
        print(summary_line(ms[key]), flush=True)
    return hs, ms


def phase_sweep(p, densities, Acs, T=50.0):
    ratio = np.zeros((Acs.size, densities.size))
    ttc = np.zeros_like(ratio)
    t0 = time.time()
    for j, d in enumerate(densities):
        for i, ac in enumerate(Acs):
            pp = p.copy(A_c=float(ac), dt=0.06)
            sim = make_overtake(seed=5, p=pp, density=float(d), road_len=1300.0)
            h = sim.run(T, rec_dt=0.15, stop_when_ev_x=1250.0)
            m = evaluate(h)
            ratio[i, j] = m["ev_speed_ratio"]
            ttc[i, j] = m["min_ttc"] if np.isfinite(m["min_ttc"]) else 9.9
        print(f"  phase sweep density {d:g}: ratios "
              f"{np.array2string(ratio[:, j], precision=2)}  [{time.time()-t0:.0f}s]",
              flush=True)
    return ratio, ttc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "runs", "figs", "phase", "anim"])
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = tuned_params()
    print("params:", {k: round(getattr(p, k), 3) for k in
                      ("A_ev", "B_ev", "A_c", "B_c", "T_react", "gamma_c")},
          flush=True)

    hs, ms = main_runs(p)

    if args.stage in ("all", "figs"):
        viz.plot_force_field(p, os.path.join(OUT, "fig_force_field.png"),
                             y_ev=hs["overtake"].y_corr,
                             y_corr=hs["overtake"].y_corr,
                             road=hs["overtake"].road)
        viz.plot_spacetime(hs["overtake"], os.path.join(OUT, "fig_spacetime.png"))
        viz.plot_bowwave(hs["overtake"], os.path.join(OUT, "fig_bowwave_overtake.png"))
        viz.plot_bowwave(hs["jam"], os.path.join(OUT, "fig_bowwave_jam.png"))
        viz.plot_reactions(ms["overtake"], os.path.join(OUT, "fig_reaction_onset.png"))
        viz.plot_compare(
            [dict(scenario="motorway overtake", baseline=ms["overtake_base"],
                  model=ms["overtake"], v0_ev=float(hs["overtake"].v0[hs["overtake"].ev])),
             dict(scenario="rescue lane in jam", baseline=ms["jam_base"],
                  model=ms["jam"], v0_ev=float(hs["jam"].v0[hs["jam"].ev]))],
            os.path.join(OUT, "fig_compare.png"))
        calib_path = os.path.join(OUT, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path) as fh:
                viz.plot_calibration(json.load(fh),
                                     os.path.join(OUT, "fig_calibration.png"))

    phase = None
    if args.stage in ("all", "phase"):
        densities = np.array([8, 12, 16, 20, 24, 28, 32, 36, 40], float)
        Acs = np.array([0.35, 0.7, 1.05, 1.4, 2.1, 2.8, 4.2, 5.6, 7.0])
        ratio, ttc = phase_sweep(p, densities, Acs)
        phase = dict(densities=densities.tolist(), Acs=Acs.tolist(),
                     ratio=ratio.tolist(), ttc=ttc.tolist())
        viz.plot_phase(densities, Acs, ratio, ttc, p,
                       os.path.join(OUT, "fig_phase_diagram.png"))
        np.savez(os.path.join(OUT, "phase.npz"), densities=densities, Acs=Acs,
                 ratio=ratio, ttc=ttc)

    if args.stage in ("all", "anim") and not args.no_anim:
        viz.animate(hs["overtake"], os.path.join(OUT, "anim_overtake.gif"),
                    t_end=min(56.0, float(hs["overtake"].t[-1])),
                    title="EV overtake, force model")
        viz.animate(hs["jam"], os.path.join(OUT, "anim_jam_rescue_lane.gif"),
                    t_end=min(62.0, float(hs["jam"].t[-1])), span=(60, 140),
                    title="Rescue-lane formation in jam")

    summary = dict(
        params={k: float(getattr(p, k)) for k in
                ("A_ev", "B_ev", "lam_ev", "A_c", "B_c", "T_react", "gamma_c",
                 "a_pin", "T_pred", "R_front", "R_urgent")},
        metrics={k: _scalars(m) for k, m in ms.items()},
        react_dists=ms["overtake"]["react_dists"].tolist(),
        ev_speed_traces={k: dict(t=hs[k].t.tolist(),
                                 v=ms[k]["ev_speed_trace"].tolist())
                         for k in hs},
        clearance_traces={k: ms[k]["clearance_trace"].tolist() for k in hs},
        phase=phase,
    )
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh)
    print("wrote out/summary.json", flush=True)


if __name__ == "__main__":
    from emv.runlog import RunLogger
    t0 = time.time()
    with RunLogger("run_all", note=" ".join(sys.argv[1:]) or "full") as rl:
        main()
        try:
            with open(os.path.join(OUT, "summary.json")) as fh:
                s = json.load(fh)
            rl.finish(metrics={k: dict(v_ev=s["metrics"][k]["ev_mean_speed"],
                                       ratio=s["metrics"][k]["ev_speed_ratio"],
                                       coll=s["metrics"][k]["collisions"])
                               for k in s["metrics"]},
                      outputs=["out/summary.json", "out/fig_*.png", "out/anim_*.gif"])
        except Exception:
            pass
    print(f"total {time.time()-t0:.0f}s", flush=True)
