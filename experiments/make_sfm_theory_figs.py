"""Figures for the quantitative-tests study (experiments/run_sfm_theory_tests.py).

Reads out/sfm_theory_tests.json, writes into paper/figs/:

  thm_depin.png     (a) measured cleared half-width vs the closed form
                    d* = w_need + B_c ln(A_c u / a_pin_eff); (b) depinning
                    order parameter phi vs the reduced control x = A_c u /
                    a_pin_eff, overtake + jam, logistic fit, transition at x=1.
  thm_predpath.png  EV progress and corridor clearance vs the corridor
                    look-ahead T_pred; the calibrated 8 s marked.
  thm_envelope.png  operating envelope: EV progress, clearance and collisions
                    vs traffic density (free-flow) and vs jam spacing.
  thm_converge.png  relative change of the headline metrics as the integrator
                    step dt is halved; production dt = 0.05 s marked.

Self-logs kind='paper_figs'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv import viz
from emv.params import bluelight_params
from emv.runlog import log_run

plt = viz.plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PFIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(PFIGS, exist_ok=True)
PAL, CAT = viz.LIGHT, viz.CAT_LIGHT


def _save(fig, name):
    p = os.path.join(PFIGS, name)
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print("wrote", p)
    return p


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (np.log(x) - np.log(x0))))


def _fit_logistic(x, y):
    """Least-squares logistic in log(x); tiny Gauss-Newton, scipy-free."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    lx0, k = 0.0, 2.0                       # log x0 = 0  -> x0 = 1
    for _ in range(200):
        z = k * (np.log(x) - lx0)
        s = 1.0 / (1.0 + np.exp(-z))
        r = s - y
        ds = s * (1 - s)
        g0 = np.sum(r * ds * (-k))
        gk = np.sum(r * ds * (np.log(x) - lx0))
        h0 = np.sum((ds * k) ** 2) + 1e-9
        hk = np.sum((ds * (np.log(x) - lx0)) ** 2) + 1e-9
        lx0 -= g0 / h0
        k -= gk / hk
    return float(np.exp(lx0)), float(k)


# ---------------------------------------------------------------- thm_depin
def fig_depin(js):
    rows = js["depin"]
    ot = [r for r in rows if r["scen"] == "overtake"]
    f = js.get("depin_fit", {})
    tr = js.get("depin_transition", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), dpi=200)
    fig.patch.set_facecolor(PAL["page"])

    # (a) measured vs analytic d*
    ax = axes[0]
    Bcs = sorted({r["B_c"] for r in ot})
    cols = dict(zip(Bcs, [CAT["blue"], CAT["aqua"], CAT["violet"]]))
    rel = [r for r in ot if np.isfinite(r["d_meas"]) and r["phi_pooled"] > 0.55
           and r["x_ctrl"] > 1.3]
    sub = [r for r in rel if r["d_star"] < 4.2]
    for r in rel:
        mk = "o" if r in sub else "o"
        fc = cols[r["B_c"]] if r in sub else "none"
        ax.plot(r["d_star"], r["d_meas"], mk, ms=5.2 if r in sub else 4.2,
                mfc=fc, mec=cols[r["B_c"]] if r not in sub else "white",
                mew=0.8 if r not in sub else 0.5,
                alpha=0.9 if r in sub else 0.6)
    lo, hi = 3.2, 8.4
    ax.plot([lo, hi], [lo, hi], color=PAL["ink2"], lw=1.0, ls="--",
            label="identity")
    if f:
        ax.axhline(f["d_meas_saturation"], color=PAL["muted"], lw=0.9, ls=":")
        ax.text(hi - 0.1, f["d_meas_saturation"] + 0.08,
                f"saturates $\\approx${f['d_meas_saturation']:.1f} m "
                "(next lane + edge)", fontsize=6.2, color=PAL["ink2"],
                ha="right")
        xs = np.array([lo, 4.2])
        ax.plot(xs, f["slope"] * xs + f["intercept"], color=CAT["red"], lw=1.5,
                label=f"OLS, $d^*\\!<\\!4.2$: RMSE {f['rmse_vs_identity_sub']:.2f} m")
        ax.text(0.03, 0.96, f"Spearman $\\rho$ = {f['spearman_rho']:.2f}  "
                            f"(n={f['n_reliable']})",
                transform=ax.transAxes, fontsize=6.8, color=PAL["ink2"],
                va="top")
    viz._style_ax(ax, PAL)
    ax.set_xlabel(r"analytic $d^{*}=w_{need}+B_c\ln(A_c u/a^{\rm eff}_{pin})$ (m)",
                  fontsize=7.7)
    ax.set_ylabel("measured cleared half-width (m)", fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(3.0, 6.2)
    hs = [plt.Line2D([], [], marker="o", ls="none", mfc=cols[b], mec="white",
                     ms=5, label=f"$B_c$={b} m") for b in Bcs]
    hs += [plt.Line2D([], [], marker="o", ls="none", mfc="none",
                      mec=PAL["muted"], ms=4, label="saturated ($d^*\\!>\\!4.2$)"),
           plt.Line2D([], [], ls="--", color=PAL["ink2"], label="identity")]
    ax.legend(handles=hs, fontsize=6.0, frameon=False, loc="lower right", ncol=1)
    ax.set_title("(a) cleared half-width vs the closed form", fontsize=8.3,
                 color=PAL["ink"])

    # (b) phi vs reduced control -- settled (lane exit) and onset (any move)
    ax = axes[1]
    otx = np.array([r["x_ctrl"] for r in ot])
    ax.plot(otx, [r["phi"] for r in ot], "o", ms=4.3, mfc=CAT["blue"],
            mec="white", mew=0.4, label=r"left the lane ($\phi$)")
    ax.plot(otx, [r["phi_onset"] for r in ot], "^", ms=3.6, mfc="none",
            mec=CAT["aqua"], mew=0.9, label=r"began to move ($\phi_{onset}$)")
    jm = [r for r in rows if r["scen"] == "jam"]
    if jm:
        ax.plot([r["x_ctrl"] for r in jm], [r["phi"] for r in jm], "s", ms=3.2,
                mfc="none", mec=PAL["muted"], mew=0.8, alpha=0.7,
                label="jam (geometry shifts level)")
    if tr:
        gx = np.geomspace(otx.min(), otx.max(), 100)
        ax.plot(gx, _logistic(gx, tr["x_mid"], tr["slope_k"]), color=PAL["ink"],
                lw=1.3, label=f"logistic, $x_{{1/2}}$={tr['x_mid']:.1f}")
    ax.axvline(1.0, color=CAT["red"], lw=1.1, ls=":")
    ax.text(1.06, 0.9, "threshold\n$x=1$", fontsize=6.4, color=CAT["red"])
    ax.set_xscale("log")
    viz._style_ax(ax, PAL)
    ax.set_xlabel(r"reduced control  $x = A_c\,u / a^{\rm eff}_{pin}$",
                  fontsize=8)
    ax.set_ylabel(r"fraction of vehicles that had to move", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6.0, frameon=False, loc="center left")
    ax.set_title("(b) corridor formation is a depinning transition",
                 fontsize=8.3, color=PAL["ink"])
    fig.tight_layout()
    return _save(fig, "thm_depin.png")


# ------------------------------------------------------------- thm_predpath
def fig_predpath(js):
    pp = js["predpath"]
    fig, ax = plt.subplots(figsize=(3.45, 2.8), dpi=200)
    fig.patch.set_facecolor(PAL["page"])
    ax2 = ax.twinx()
    styles = {"overtake_d22": ("-", r"$\rho$=22"),
              "overtake_d30": ((0, (5, 2)), r"$\rho$=30")}
    for name, rows in pp.items():
        ls, lab = styles.get(name, ("-", name))
        tp = [r["T_pred"] for r in rows]
        clr = [r["clearance_mean"] for r in rows]
        spd = [r["ev_speed_ratio"] for r in rows]
        cerr = [r.get("clearance_sd", 0) for r in rows]
        ax.errorbar(tp, clr, yerr=cerr, ls=ls, color=CAT["blue"], lw=1.5,
                    marker="o", ms=4, capsize=2)
        ax2.plot(tp, spd, ls=ls, color=CAT["red"], lw=1.3, marker="s", ms=3.5)
    ax.axvline(8.0, color=PAL["muted"], lw=1.0, ls=":")
    ax.text(7.4, 73.5, "calibrated 8 s", fontsize=6.2, color=PAL["ink2"],
            ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    viz._style_ax(ax, PAL)
    ax.set_xlabel(r"corridor look-ahead $T_{pred}$ (s) — "
                  r"1 s $\approx$ instantaneous", fontsize=7.3)
    ax.set_ylabel("corridor clearance (m)", fontsize=8, color=CAT["blue"])
    ax.tick_params(axis="y", labelsize=7, colors=CAT["blue"])
    ax2.set_ylabel("EV speed / desired", fontsize=8, color=CAT["red"])
    ax2.tick_params(axis="y", labelsize=7, colors=CAT["red"])
    ax.tick_params(axis="x", labelsize=7)
    proxies = [
        plt.Line2D([], [], color=CAT["blue"], marker="o", ms=4,
                   label="corridor clearance"),
        plt.Line2D([], [], color=CAT["red"], marker="s", ms=3.5,
                   label="EV speed / desired"),
        plt.Line2D([], [], color=PAL["ink2"], ls="-", label=r"$\rho$=22 veh/km/ln"),
        plt.Line2D([], [], color=PAL["ink2"], ls=(0, (5, 2)),
                   label=r"$\rho$=30 veh/km/ln")]
    ax.legend(handles=proxies, fontsize=5.9, frameon=False, loc="lower right",
              ncol=1, labelspacing=0.3)
    ax.set_title("The predicted path, not the instantaneous position,\n"
                 "drives yielding", fontsize=7.8, color=PAL["ink"])
    fig.tight_layout()
    return _save(fig, "thm_predpath.png")


# ------------------------------------------------------------- thm_envelope
def fig_envelope(js):
    env = js["envelope"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(PAL["page"])
    for ax, (key, xlab, xget) in zip(
            axes,
            (("overtake", r"traffic density $\rho$ (veh/km/lane)",
              lambda r: r["density"]),
             ("jam", r"equivalent density $1000/{\rm spacing}$ (veh/km/lane)",
              lambda r: r["density_eq"]))):
        rows = sorted(env[key], key=xget)
        x = [xget(r) for r in rows]
        spd = [r["ev_speed_ratio"] for r in rows]
        clr = [r["clearance_mean"] / 100.0 for r in rows]
        coll = [r["collisions"] for r in rows]
        l1, = ax.plot(x, spd, "-o", color=CAT["blue"], lw=1.5, ms=4,
                      label="EV speed / desired")
        l2, = ax.plot(x, clr, "-s", color=CAT["aqua"], lw=1.2, ms=3.5,
                      label="clearance / 100 m")
        ax.set_ylim(0, 1.05)
        ax3 = ax.twinx()
        l3, = ax3.plot(x, coll, "-^", color=CAT["red"], lw=1.2, ms=4,
                       label="collisions / run")
        # free-flow contacts are noise (<=0.3/run); cap the axis so they read flat
        ax3.set_ylim(0, 3.0 if key == "overtake" else None)
        ax3.tick_params(axis="y", labelsize=7, colors=CAT["red"])
        ax3.set_ylabel("collisions / run", fontsize=8, color=CAT["red"])
        good = [xi for xi, s, c in zip(x, spd, coll) if c < 0.5 and s > 0.9]
        if good:
            ax.axvspan(min(x), max(good), color=CAT["aqua"], alpha=0.09, lw=0)
        viz._style_ax(ax, PAL)
        ax.set_xlabel(xlab, fontsize=7.3)
        ax.set_title({"overtake": "(a) free-flow: graceful, collision-free",
                      "jam": "(b) jam: outside the calibrated range"}[key],
                     fontsize=8, color=PAL["ink"])
        ax.tick_params(labelsize=7)
        if ax is axes[0]:
            ax.legend(handles=[l1, l2, l3], fontsize=6.0, frameon=False,
                      loc="lower left")
    axes[0].set_ylabel("EV speed / desired,  clearance / 100 m", fontsize=8)
    fig.tight_layout()
    return _save(fig, "thm_envelope.png")


# ------------------------------------------------------------- thm_converge
def fig_converge(js):
    cv = js["converge"]
    keys = ("ev_speed_ratio", "clearance_mean", "min_ttc", "react_dist_mean")
    klab = {"ev_speed_ratio": "EV speed ratio", "clearance_mean": "clearance",
            "min_ttc": "min TTC", "react_dist_mean": "yield onset"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), dpi=200, sharey=True)
    fig.patch.set_facecolor(PAL["page"])
    cols = [CAT["blue"], CAT["aqua"], CAT["violet"], CAT["orange"]]
    for ax, sc in zip(axes, ("overtake", "jam")):
        rows = sorted(cv[sc], key=lambda r: -r["dt"])
        dts = [r["dt"] for r in rows]
        ref = rows[-1]                       # finest dt = reference
        for c, k in zip(cols, keys):
            rel = [100.0 * (r[k] - ref[k]) / abs(ref[k]) for r in rows]
            ax.plot(dts, rel, "-o", ms=4, color=c, lw=1.3, label=klab[k])
        ax.axhline(0, color=PAL["muted"], lw=0.8)
        ax.axvline(0.05, color=PAL["ink2"], lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(dts)
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter())
        viz._style_ax(ax, PAL)
        ax.set_xlabel("integrator step dt (s)", fontsize=8)
        ax.set_title(sc, fontsize=8.5, color=PAL["ink"])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(r"deviation from dt=%.4g s  (%%)" %
                       cv["overtake"][-1]["dt"], fontsize=8)
    axes[0].legend(fontsize=6.3, frameon=False, loc="best")
    axes[0].text(0.052, axes[0].get_ylim()[1] * 0.8, "production", fontsize=6.4,
                 color=PAL["ink2"])
    fig.suptitle("Numerical convergence of the headline metrics", fontsize=9,
                 color=PAL["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "thm_converge.png")


def main():
    with open(os.path.join(OUT, "sfm_theory_tests.json")) as fh:
        js = json.load(fh)
    outs = []
    if "depin" in js:
        outs.append(fig_depin(js))
    if "predpath" in js:
        outs.append(fig_predpath(js))
    if "envelope" in js:
        outs.append(fig_envelope(js))
    if "converge" in js:
        outs.append(fig_converge(js))     # kept for the record; the paper uses
        #                                   a table for convergence, not this fig
    log_run("paper_figs", note="SFM formulation-test figures (thm_*)",
            outputs=[os.path.relpath(o, ROOT) for o in outs])
    print("done:", len(outs))


if __name__ == "__main__":
    main()
