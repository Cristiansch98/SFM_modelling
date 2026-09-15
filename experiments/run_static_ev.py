"""Static-EV comparison: park the emergency vehicle, move only the traffic.

Question: how does yielding behaviour differ around a *static* (parked, siren
on) EV versus the standard moving EV?  The model's fields are direction-of-
travel based: the corridor projects from the predicted path (only L_back=6 m
behind the nose) and the point field is forward-focused (lam_ev=0.1), so a
parked EV is nearly silent toward the traffic approaching it from behind, and
detection there is R_rear=60 m (mirror/siren geometry of a *moving* EV).
The static case is therefore run twice:

  *_static       stock parameters - the model's naive answer (queueing expected)
  *_static_inc   'incident mode': approaching drivers see the parked lightbar
                 through the windshield (R_rear -> R_front) and the protected
                 corridor covers the approach zone (L_back -> 120 m); the
                 move-over-law reading - two param overrides, no new physics
  *_static_base  yielding=False - silent-obstacle control
  *_moving       the paper-protocol run for reference

Behaviour is compared on EV-frame metrics that exist in both arms: passing
lateral gap and passing speed while alongside the EV, yield-onset distance,
p95 braking / lateral accel, min TTC, collisions, background disruption.

Writes out/static_ev_compare.json, out/fig_static_compare.png,
out/fig_static_spacetime_{ov,jam}{,_inc}.png; ledger kind=static_ev_compare.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import tuned_params
from emv.scenarios import make_overtake, make_jam
from emv.simulate import Sim
from emv.metrics import evaluate
from emv import viz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1] if "--quick" in sys.argv else list(range(1, 11))
KEYS =["collisions", "min_ttc", "react_dist_mean", "p95_decel", "p95_alat",
        "oscillation", "disruption", "n_yielded", "lat_disp_mean",
        "ev_mean_speed"]
PASS_KEYS = ["n_passed", "gap_med", "gap_min", "speed_med", "ratio_med",
             "coll_pairs", "coll_ev"]


class StaticEVSim(Sim):
    """Sim whose EV is pinned in place: position restored and velocity zeroed
    after every step, so only the surrounding traffic moves."""

    def step(self, diag: bool = False):
        st, i = self.st, self.st.ev
        x0, y0 = float(st.x[i]), float(st.y[i])
        out = super().step(diag)
        st.x[i], st.y[i] = x0, y0
        st.vx[i] = 0.0
        st.vy[i] = 0.0
        return out


def park(sim: Sim, x_target: float, y_ev: float | None = None) -> Sim:
    """Move the EV into the widest longitudinal gap near x_target in the lane
    band around y_ev, zero its speed and intent, aim the corridor at itself.
    v0=0 floors the corridor at L_pred_min (pure 'intent' projection)."""
    st, i = sim.st, sim.st.ev
    y_ev = float(st.y[i]) if y_ev is None else float(y_ev)
    lane = ~st.is_ev & (np.abs(st.y - y_ev) < 0.6 * sim.road.lane_width)
    xs = np.sort(st.x[lane])
    x_park = x_target
    if xs.size >= 2:
        mids, widths = 0.5 * (xs[:-1] + xs[1:]), np.diff(xs)
        near = np.abs(mids - x_target) < 200.0
        if near.any():
            j = np.flatnonzero(near)[np.argmax(widths[near])]
            x_park = float(mids[j])
    st.x[i], st.y[i] = x_park, y_ev
    st.vx[i] = 0.0
    st.vy[i] = 0.0
    st.v0[i] = 0.0
    # Cars already inside the no-escape braking envelope at t=0 would be
    # doomed by construction (the parked EV materialises in front of them).
    # The queue predates the observation window: cap their initial speed to
    # what a comfortable-braking stop from the current gap allows.
    gap = (st.x[i] - 0.5 * st.L[i]) - (st.x + 0.5 * st.L)
    approach = lane & (gap > 0.0) & (gap < 130.0)
    v_safe = np.sqrt(2.0 * sim.p.b_comf * np.maximum(gap - 4.0, 0.0))
    st.vx[approach] = np.minimum(st.vx[approach], v_safe[approach])
    sim.y_corr = y_ev
    sim.__class__ = StaticEVSim
    return sim


def passing_stats(h) -> dict:
    """EV-frame metrics for every vehicle that crosses the EV: minimum lateral
    body gap while alongside, and mean speed there (abs + fraction of own v0).
    Works identically for a moving EV (EV passes car) and a static one."""
    ev = h.ev
    gaps, speeds, ratios = [], [], []
    for i in range(h.x.shape[1]):
        if i == ev:
            continue
        dx = h.x[:, i] - h.x[:, ev]
        if not (dx[0] < 0.0 < dx[-1] or dx[0] > 0.0 > dx[-1]):
            continue                                    # never crossed the EV
        alongside = np.abs(dx) < 0.5 * (h.L[ev] + h.L[i]) + 1.0
        if not alongside.any():
            continue
        lat = np.abs(h.y[alongside, i] - h.y[alongside, ev]) \
            - 0.5 * (h.W[ev] + h.W[i])
        gaps.append(float(lat.min()))
        v = float(h.vx[alongside, i].mean())
        speeds.append(v)
        ratios.append(v / max(float(h.v0[i]), 0.5))
    g = np.array(gaps)
    return dict(
        n_passed=len(gaps),
        gap_med=float(np.median(g)) if g.size else float("nan"),
        gap_min=float(g.min()) if g.size else float("nan"),
        speed_med=float(np.median(speeds)) if speeds else float("nan"),
        ratio_med=float(np.median(ratios)) if ratios else float("nan"),
        gaps=gaps)


def collision_pairs(h) -> tuple[int, int]:
    """Unique colliding vehicle pairs (the paper metric counts overlap
    *frames*, which over-weights a car wedged against a parked EV), plus how
    many of those pairs involve the EV itself."""
    ev, seen = h.ev, set()
    for k in range(h.n_frames):
        x, y = h.x[k], h.y[k]
        dx = x[None, :] - x[:, None]
        dy = np.abs(y[None, :] - y[:, None])
        gap = dx - 0.5 * (h.L[None, :] + h.L[:, None])
        hit = (dx > 0.0) & (dy < 0.5 * (h.W[None, :] + h.W[:, None]) - 0.05) \
            & (gap < 0.0)
        for i, j in zip(*np.nonzero(hit)):
            seen.add((min(i, j), max(i, j)))
    return len(seen), sum(ev in pr for pr in seen)


def build(cond: str, seed: int, p):
    """Return (sim, T, stop_when_ev_x) for one condition."""
    inc = p.copy(R_rear=p.R_front, L_back=120.0)
    if cond == "ov_moving":
        return make_overtake(seed=seed, p=p), 78.0, 2320.0
    if cond == "ov_static":
        return park(make_overtake(seed=seed, p=p), 1400.0), 100.0, None
    if cond == "ov_static_inc":
        return park(make_overtake(seed=seed, p=inc), 1400.0), 100.0, None
    if cond == "ov_static_base":
        return park(make_overtake(seed=seed, p=p, yielding=False),
                    1400.0), 100.0, None
    if cond == "jam_moving":
        return make_jam(seed=seed, p=p), 70.0, 700.0
    # static jam: EV parked in the middle-lane centre (a stalled ambulance
    # occupies a lane, not the Rettungsgasse boundary)
    if cond == "jam_static":
        sim = make_jam(seed=seed, p=p)
    elif cond == "jam_static_inc":
        sim = make_jam(seed=seed, p=inc)
    elif cond == "jam_static_base":
        sim = make_jam(seed=seed, p=p, yielding=False)
    else:
        raise ValueError(cond)
    return park(sim, 400.0, y_ev=sim.road.lane_center(1)), 140.0, None


CONDS = ["ov_moving", "ov_static", "ov_static_inc", "ov_static_base",
         "jam_moving", "jam_static", "jam_static_inc", "jam_static_base"]


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), lo=float("nan"),
                    hi=float("nan"), n=0)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                lo=float(v.min()), hi=float(v.max()), n=int(v.size))


def make_figure(summary, path):
    """Grouped-bar comparison, emv.viz palette. Fixed condition colours."""
    import matplotlib.pyplot as plt
    pal, cat = viz.LIGHT, viz.CAT_LIGHT
    colors = {"moving": cat["blue"], "static": cat["orange"],
              "static_inc": cat["yellow"], "static_base": pal["muted"]}
    labels = {"moving": "moving EV", "static": "parked (stock)",
              "static_inc": "parked (incident mode)",
              "static_base": "parked (no siren)"}
    panels = [
        ("gap_med", "passing lateral gap (m)", 1.0),
        ("ratio_med", "passing speed (% of own desired)", 100.0),
        ("react_dist_abs", "yield onset distance from EV (m)", 1.0),
        ("p95_decel", "p95 deceleration (m/s$^2$)", 1.0),
    ]
    fig = viz._fig(9.2, 6.4, pal)
    axs = fig.subplots(2, 2)
    for ax, (key, title, scale) in zip(axs.ravel(), panels):
        viz._style_ax(ax, pal)
        ax.grid(axis="x", visible=False)
        for gi, sc in enumerate(("ov", "jam")):
            for ci, variant in enumerate(colors):
                a = summary[f"{sc}_{variant}"].get(key)
                x = gi * (len(colors) + 1) + ci
                if a is None or not np.isfinite(a["mean"]):
                    ax.text(x, 0.02, "n/a", ha="center", va="bottom",
                            color=pal["muted"], fontsize=7, rotation=90,
                            transform=ax.get_xaxis_transform())
                    continue
                m, sd = a["mean"] * scale, a["sd"] * scale
                ax.bar(x, m, 0.78, color=colors[variant], zorder=3,
                       yerr=(sd if a["n"] > 1 else None), error_kw=dict(
                           ecolor=pal["ink2"], elinewidth=0.9, capsize=2))
                ax.annotate(f"{m:.3g}", (x, m), xytext=(0, 3 + (7 if a['n'] > 1 else 0)),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=pal["ink2"])
        ax.set_xticks([1.5, len(colors) + 2.5])
        ax.set_xticklabels(["motorway overtake", "rescue lane in jam"])
        ax.set_title(title, fontsize=9.5, pad=6)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors.values()]
    fig.legend(handles, labels.values(), ncol=4, loc="lower center",
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02),
               labelcolor=pal["ink2"])
    fig.suptitle("Traffic behaviour around a static vs. moving EV "
                 f"({len(SEEDS)} seeds, mean $\\pm$ s.d.)",
                 fontsize=11, color=pal["ink"], y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    viz._save(fig, path)


def plot_spacetime_static(h, path, title):
    """Space-time diagram cropped around the parked EV (viz.plot_spacetime
    frames a moving EV and titles it accordingly)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    pal, ev = viz.LIGHT, h.ev
    fig = viz._fig(10.5, 5.2, pal)
    ax = fig.add_subplot(111)
    viz._style_ax(ax, pal)
    v_ref = np.median(h.v0[np.arange(h.v0.size) != ev])
    segs, cols = [], []
    for i in range(h.x.shape[1]):
        if i == ev:
            continue
        pts = np.column_stack([h.t, h.x[:, i]])
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        cong = np.clip(1.0 - h.vx[:, i] / v_ref, 0.0, 1.0)
        cols.append(cong[:-1])
    lc = LineCollection(np.concatenate(segs), cmap=viz.CMAP_SEQ,
                        norm=plt.Normalize(0, 1), lw=0.75, alpha=0.85)
    lc.set_array(np.concatenate(cols))
    ax.add_collection(lc)
    x_ev = float(h.x[0, ev])
    ax.axhline(x_ev, color=viz.CAT_LIGHT["red"], lw=2.4, label="parked EV",
               zorder=5)
    ax.set_xlim(h.t[0], h.t[-1])
    ax.set_ylim(x_ev - 350, x_ev + 250)
    cb = fig.colorbar(lc, ax=ax, pad=0.012)
    cb.set_label("congestion  1 - v/v$_0$", color=pal["ink2"], fontsize=9)
    cb.ax.tick_params(colors=pal["muted"], labelsize=8)
    cb.outline.set_visible(False)
    ax.set_xlabel("t (s)"); ax.set_ylabel("x (m)")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False,
              labelcolor=pal["ink2"])
    viz._save(fig, path)


def main():
    p = tuned_params()
    t0 = time.time()
    per = {c: {k: [] for k in KEYS + PASS_KEYS} for c in CONDS}
    pooled = {c: dict(gaps=[], onsets=[]) for c in CONDS}
    keep = {}

    for s in SEEDS:
        for cond in CONDS:
            sim, T, stop = build(cond, s, p)
            sim.name = f"{cond}_s{s}"
            h = sim.run(T, rec_dt=0.1, stop_when_ev_x=stop)
            m = evaluate(h)
            ps = passing_stats(h)
            ps["coll_pairs"], ps["coll_ev"] = collision_pairs(h)
            for k in KEYS:
                per[cond][k].append(m[k])
            for k in PASS_KEYS:
                per[cond][k].append(ps[k])
            pooled[cond]["gaps"].extend(ps["gaps"])
            pooled[cond]["onsets"].extend(m["react_dists"].tolist())
            if s == SEEDS[0] and cond in ("ov_static", "ov_static_inc",
                                          "jam_static", "jam_static_inc"):
                keep[cond] = h
        print(f"seed {s} done  [{time.time()-t0:.0f}s]", flush=True)

    summary = {c: {k: agg(v) for k, v in d.items()} for c, d in per.items()}
    for c in CONDS:
        summary[c]["collisions_total"] = int(np.nansum(per[c]["collisions"]))
        summary[c]["gaps_pooled"] = agg(pooled[c]["gaps"])
        summary[c]["onsets_pooled"] = agg(pooled[c]["onsets"])
        # onset distance as unsigned distance from the EV (moving EV reacts
        # ahead of itself: +s; a parked EV is approached from behind: -s)
        summary[c]["react_dist_abs"] = agg([abs(v) for v in
                                            per[c]["react_dist_mean"]])
    summary["n_seeds"] = len(SEEDS)
    summary["protocol"] = dict(
        ov_static=dict(x_park=1400.0, T=100.0),
        jam_static=dict(x_park=400.0, lane="middle centre", T=140.0),
        incident_mode=dict(R_rear="R_front (140)", L_back=120.0),
        moving="paper protocol (multiseed)")

    with open(os.path.join(OUT, "static_ev_compare.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote out/static_ev_compare.json", flush=True)

    make_figure(summary, os.path.join(OUT, "fig_static_compare.png"))
    titles = {
        "ov_static": "Parked EV, stock model: approaching traffic notices at "
                     "R_rear=60 m and piles up (line shade = congestion)",
        "ov_static_inc": "Parked EV, incident mode: early sighting + approach "
                         "corridor produce a smooth move-over",
        "jam_static": "Parked EV in the jam, stock model",
        "jam_static_inc": "Parked EV in the jam, incident mode"}
    for cond, tag in (("ov_static", "ov"), ("ov_static_inc", "ov_inc"),
                      ("jam_static", "jam"), ("jam_static_inc", "jam_inc")):
        plot_spacetime_static(
            keep[cond], os.path.join(OUT, f"fig_static_spacetime_{tag}.png"),
            titles[cond])

    print(f"\n--- static vs moving ({len(SEEDS)} seeds, mean+-sd) ---")
    for c in CONDS:
        g, r, o, d = (summary[c][k] for k in
                      ("gap_med", "ratio_med", "react_dist_abs", "p95_decel"))
        print(f"{c:<16s} gap {g['mean']:5.2f}+-{g['sd']:4.2f} m  "
              f"passv {r['mean']*100:5.1f}%  onset {o['mean']:6.1f} m  "
              f"p95dec {d['mean']:4.2f}  TTCmin {summary[c]['min_ttc']['lo']:4.2f}  "
              f"collpairs {summary[c]['coll_pairs']['mean']:4.1f} "
              f"(EV {summary[c]['coll_ev']['mean']:3.1f})  "
              f"npass {summary[c]['n_passed']['mean']:5.1f}")
    print(f"total {time.time()-t0:.0f}s")
    return summary


if __name__ == "__main__":
    from emv.runlog import RunLogger
    with RunLogger("static_ev_compare",
                   note=f"{len(SEEDS)} seeds x 8 conds: parked EV (stock / "
                        "incident-mode / no-siren) vs moving EV") as rl:
        s = main()
        rl.finish(
            metrics={c: dict(gap=s[c]["gap_med"]["mean"],
                             passv=s[c]["ratio_med"]["mean"],
                             coll=s[c]["collisions_total"])
                     for c in CONDS},
            outputs=["out/static_ev_compare.json", "out/fig_static_compare.png",
                     "out/fig_static_spacetime_ov.png",
                     "out/fig_static_spacetime_ov_inc.png",
                     "out/fig_static_spacetime_jam.png",
                     "out/fig_static_spacetime_jam_inc.png"])
