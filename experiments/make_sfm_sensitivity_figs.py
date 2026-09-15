"""Figures for the SFM two-stage parameter-sensitivity study.

Reads out/sfm_param_sensitivity.{json,npz} (experiments/run_sfm_param_sensitivity.py)
and writes into paper/figs/:

  sens_tornado.png    OAT swing (max-min of the seed mean over the search
                      range) of EV speed ratio, every parameter, sorted, both
                      scenarios; bars coloured by parameter block.
  sens_curves.png     OAT response curves of the nine highest-swing parameters,
                      EV speed ratio with a bootstrap CI band, calibrated value
                      marked.
  sens_gsa.png        Spearman rank correlation of every headline metric on
                      every parameter (global LHS sample), with the linear
                      surrogate R2 per metric.
  sens_reference.png  calibrated (blue-light) vs dataclass-default vs a
                      no-yielding control: EV progress, clearance, min TTC,
                      disruption, collisions.
  sens_surface.png    A_c x a_pin plane: EV progress vs the analytic escape
                      boundary A_c u > a_pin(1 - eps u).

Self-logs kind='paper_figs'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from emv import viz
from emv.params import bluelight_params
from emv.runlog import log_run
from run_sfm_param_sensitivity import FULL_SPEC, HEADLINE, PCLASS, PNAMES

plt = viz.plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PFIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(PFIGS, exist_ok=True)
PAL, CAT = viz.LIGHT, viz.CAT_LIGHT

CLASS_C = {"stage1-lon": CAT["blue"], "stage1-lat": CAT["aqua"],
           "stage1-lc": CAT["violet"], "stage2-ev": CAT["red"],
           "handset-perc": CAT["orange"]}
CLASS_LAB = {"stage1-lon": "stage 1 - car following",
             "stage1-lat": "stage 1 - lane keeping",
             "stage1-lc": "stage 1 - lane change",
             "stage2-ev": "stage 2 - EV block",
             "handset-perc": "hand-set (perception)"}
PLABEL = {
    "tau": r"$\tau$", "T_hw_lo": r"$T_{hw}^{lo}$", "T_hw_hi": r"$T_{hw}^{hi}$",
    "s0": r"$s_0$", "A_v": r"$A_v$", "Bx_v": r"$B_x$", "a_max": r"$a_{max}$",
    "b_comf": r"$b$", "a_pin": r"$a_{pin}$", "zeta_lat": r"$\zeta$",
    "v_lat_max": r"$v_{lat}^{max}$", "a_lat_max": r"$a_{lat}^{max}$",
    "sigma_off": r"$\sigma_{off}$", "het_lat": r"$het_{lat}$", "k_rho": r"$k_\rho$",
    "A_pass": r"$A_{pass}$", "A_keep_right": r"$A_{keep}$", "T_frust": r"$T_{frust}$",
    "s_veto": r"$s_{veto}$", "A_ev": r"$A_{ev}$", "B_ev": r"$B_{ev}$",
    "A_c": r"$A_c$", "B_c": r"$B_c$", "gamma_c": r"$\gamma_c$",
    "T_react": r"$T_{react}$", "p_noncomply": r"$p_{nc}$", "R_front": r"$R_{front}$",
    "delay_med": r"$t_{delay}$", "urgency_pin_relief": r"$\varepsilon$"}
MLABEL = {"ev_speed_ratio": "EV speed/desired", "ev_med_speed": "EV median speed",
          "clearance_mean": "corridor clearance", "min_ttc": "min TTC",
          "react_dist_mean": "yield onset dist.", "disruption": "traffic disruption",
          "lc_per_veh_km": "lane changes/veh-km", "collisions": "collisions"}
SCEN = (("o_", "free-flow overtake"), ("j_", "stop-and-go jam"))


def _load():
    with open(os.path.join(OUT, "sfm_param_sensitivity.json")) as fh:
        js = json.load(fh)
    npz = np.load(os.path.join(OUT, "sfm_param_sensitivity.npz"))
    return js, npz


def _save(fig, name):
    p = os.path.join(PFIGS, name)
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=170)
    plt.close(fig)
    print("wrote", p)
    return p


# ------------------------------------------------------------- tornado
def fig_tornado(js):
    swing = js["oat"]["swing"]
    order = sorted(PNAMES, key=lambda n: -max(swing[n]["o_ev_speed_ratio"],
                                              swing[n]["j_ev_speed_ratio"]))
    n = len(order)
    y = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.30 * n + 1.1), dpi=170,
                             sharey=True)
    fig.patch.set_facecolor(PAL["page"])
    for ax, (pre, title) in zip(axes, SCEN):
        vals = [swing[nm][pre + "ev_speed_ratio"] for nm in order]
        cols = [CLASS_C[PCLASS[nm]] for nm in order]
        ax.barh(y, vals, color=cols, height=0.72)
        viz._style_ax(ax, PAL)
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([PLABEL.get(nm, nm) for nm in order], fontsize=8)
        ax.set_xlabel("swing of EV speed / desired  (max $-$ min over range)",
                      fontsize=8.5)
        ax.set_title(title, fontsize=10, color=PAL["ink"])
        ax.tick_params(labelsize=7.5)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=7, mfc=CLASS_C[c],
                          mec="none", label=CLASS_LAB[c]) for c in CLASS_C]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("One-at-a-time influence of every tunable parameter on EV "
                 "progress", fontsize=12, color=PAL["ink"], x=0.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.10,
                        wspace=0.06)
    return _save(fig, "sens_tornado.png")


def fig_tornado_col(js, k=16):
    """Column-width single-panel tornado for the paper: jam EV progress, the
    k most influential parameters, both scenarios shown as paired bars."""
    swing = js["oat"]["swing"]
    order = sorted(PNAMES, key=lambda n: -max(swing[n]["o_ev_speed_ratio"],
                                              swing[n]["j_ev_speed_ratio"]))[:k]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 0.238 * len(order) + 0.85), dpi=200)
    fig.patch.set_facecolor(PAL["page"])
    h = 0.38
    ax.barh(y + h / 2, [swing[n]["j_ev_speed_ratio"] for n in order], height=h,
            color=[CLASS_C[PCLASS[n]] for n in order], label="stop-and-go jam")
    ax.barh(y - h / 2, [swing[n]["o_ev_speed_ratio"] for n in order], height=h,
            color=[CLASS_C[PCLASS[n]] for n in order], alpha=0.42,
            label="free-flow overtake")
    viz._style_ax(ax, PAL)
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([PLABEL.get(n, n) for n in order], fontsize=7.5)
    ax.set_xlabel("swing of EV speed / desired over the parameter's range",
                  fontsize=7.5)
    ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=6,
                          mfc=CLASS_C[c], mec="none", label=CLASS_LAB[c])
               for c in ("stage1-lat", "stage2-ev", "stage1-lon",
                         "handset-perc")]
    ax.legend(handles=handles, fontsize=6.2, frameon=False, loc="lower right",
              handletextpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    return _save(fig, "sens_tornado_col.png")


# ------------------------------------------------------------- OAT curves
def fig_curves(js, npz, k=9):
    swing = js["oat"]["swing"]
    order = sorted(PNAMES, key=lambda n: -max(swing[n]["o_ev_speed_ratio"],
                                              swing[n]["j_ev_speed_ratio"]))[:k]
    keys = js["oat"]["keys"]
    idx = {kk: keys.index(kk) for kk in ("o_ev_speed_ratio", "j_ev_speed_ratio")}
    nr = int(np.ceil(k / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(11.4, 2.5 * nr), dpi=170)
    fig.patch.set_facecolor(PAL["page"])
    for ax, nm in zip(axes.flat, order):
        grid = np.asarray(js["oat"]["grids"][nm])
        Y = npz[f"oat_{nm}_Y"]                       # levels x seeds x metrics
        for (pre, lab), c in zip(SCEN, (CAT["red"], CAT["blue"])):
            col = idx[pre + "ev_speed_ratio"]
            m = np.nanmean(Y[:, :, col], axis=1)
            sd = np.nanstd(Y[:, :, col], axis=1)
            se = sd / max(np.sqrt(Y.shape[1]), 1)
            ax.fill_between(grid, m - se, m + se, color=c, alpha=0.18, lw=0)
            ax.plot(grid, m, color=c, lw=1.6, label=lab)
        ax.axvline(js["oat"]["base"][nm], color=PAL["ink2"], lw=1.0, ls="--")
        viz._style_ax(ax, PAL)
        ax.set_title(PLABEL.get(nm, nm) + f"  ({PCLASS[nm].split('-')[-1]})",
                     fontsize=9, color=PAL["ink"])
        ax.tick_params(labelsize=7)
    for ax in axes.flat[k:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("EV speed / desired vs each parameter (dashed = calibrated "
                 "value; band = $\\pm$1 s.e. over seeds)", fontsize=11,
                 color=PAL["ink"])
    fig.supylabel("EV speed / desired", fontsize=9, color=PAL["ink2"])
    fig.tight_layout(rect=(0.02, 0.0, 1, 0.95))
    return _save(fig, "sens_curves.png")


# ------------------------------------------------------------- GSA heatmap
def fig_gsa(js):
    g = js["gsa"]
    cols = [pre + m for m in HEADLINE for pre, _ in SCEN]
    cols = [c for c in cols if c in g["keys"]]
    M = np.array([[g["rho"][c][p] for c in cols] for p in g["params"]])
    r2 = [g["r2"][c] for c in cols]
    order = sorted(range(len(g["params"])),
                   key=lambda i: -np.nanmax(np.abs(M[i])))
    M, params = M[order], [g["params"][i] for i in order]
    fig, ax = plt.subplots(figsize=(0.52 * len(cols) + 3.2,
                                    0.29 * len(params) + 1.6), dpi=170)
    fig.patch.set_facecolor(PAL["page"])
    im = ax.imshow(M, cmap=viz.CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([("O " if c.startswith("o_") else "J ")
                        + MLABEL.get(c[2:], c[2:]) for c in cols],
                       rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([PLABEL.get(p, p) for p in params], fontsize=7.5)
    for i in range(len(params)):
        for j in range(len(cols)):
            if abs(M[i, j]) >= 0.30:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if abs(M[i, j]) > 0.55 else PAL["ink"])
    for j, r in enumerate(r2):
        ax.text(j, -0.9, f"R$^2$\n{r:.2f}", ha="center", va="center",
                fontsize=5.6, color=PAL["muted"])
    ax.set_title("Global sensitivity: Spearman $\\rho$ (LHS over all "
                 f"{len(g['params'])} parameters)", fontsize=10.5,
                 color=PAL["ink"], pad=18)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("rank correlation", fontsize=8, color=PAL["ink2"])
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, "sens_gsa.png")


# ------------------------------------------------------------- reference bars
def fig_reference(js):
    ref = js["reference"]
    cfgs = ["no_yield", "default", "bluelight"]
    cc = {"no_yield": PAL["muted"], "default": CAT["orange"],
          "bluelight": CAT["aqua"]}
    mets = ["ev_speed_ratio", "clearance_mean", "min_ttc", "disruption",
            "collisions"]
    fig, axes = plt.subplots(2, len(mets), figsize=(12.0, 4.4), dpi=170)
    fig.patch.set_facecolor(PAL["page"])
    for r, (pre, stitle) in enumerate(SCEN):
        sc = "overtake" if pre == "o_" else "jam"
        for cax, mk in zip(axes[r], mets):
            xs = np.arange(len(cfgs))
            vals, los, his = [], [], []
            for cfg in cfgs:
                v, lo, hi = ref[cfg][sc][mk]
                vals.append(v)
                los.append(v - lo if np.isfinite(lo) else 0)
                his.append(hi - v if np.isfinite(hi) else 0)
            cax.bar(xs, vals, color=[cc[c] for c in cfgs],
                    yerr=[los, his], capsize=2, error_kw=dict(lw=0.8))
            viz._style_ax(cax, PAL)
            cax.set_xticks(xs)
            cax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=6.5)
            if r == 0:
                cax.set_title(MLABEL.get(mk, mk), fontsize=8.5,
                              color=PAL["ink"])
            if mk == mets[0]:
                cax.set_ylabel(stitle, fontsize=8.5, color=PAL["ink2"])
            cax.tick_params(labelsize=7)
    fig.suptitle("Performance: calibrated blue-light model vs default vs "
                 "no-yielding control", fontsize=12, color=PAL["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "sens_reference.png")


# ------------------------------------------------------------- surface
def fig_surface(npz):
    A_c, a_pin = npz["surf_A_c"], npz["surf_a_pin"]
    p = bluelight_params()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=170)
    fig.patch.set_facecolor(PAL["page"])
    fig.subplots_adjust(left=0.08, right=0.87, top=0.82, bottom=0.14,
                        wspace=0.13)
    vmin = float(np.floor(min(npz["surf_overtake"][:, :, 0].min(),
                              npz["surf_jam"][:, :, 0].min()) * 20) / 20)
    im = None
    for ax, (key, title) in zip(axes, (("surf_overtake", "free-flow overtake"),
                                       ("surf_jam", "stop-and-go jam"))):
        Z = npz[key][:, :, 0]
        viz._style_ax(ax, PAL)
        ax.grid(False)
        im = ax.pcolormesh(a_pin, A_c, Z, cmap=viz.CMAP_SEQ, vmin=vmin, vmax=1.0,
                           shading="nearest")
        levels = [l for l in (0.5, 0.7, 0.85, 0.95) if vmin < l < 1.0]
        cs = ax.contour(a_pin, A_c, Z, levels=levels, colors="#f9f9f7",
                        linewidths=0.9, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f")
        xs = np.linspace(a_pin[0], a_pin[-1], 100)
        ax.plot(xs, xs * (1.0 - p.urgency_pin_relief), color="#8f1d1d", lw=2.0,
                label=r"escape boundary $A_c=a_{pin}(1-\varepsilon)$")
        ax.plot(p.a_pin, p.A_c, marker="*", ms=14, mfc=CAT["yellow"],
                mec="#3a2a00", mew=0.8, ls="none", label="calibrated point")
        ax.set_xlim(a_pin[0], a_pin[-1])
        ax.set_ylim(A_c[0], A_c[-1])
        ax.set_xlabel(r"$a_{pin}$  lane-keeping strength (m/s$^2$)", fontsize=9)
        ax.set_title(title, fontsize=10, color=PAL["ink"])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"$A_c$  corridor push (m/s$^2$)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7.8, frameon=False,
                   labelcolor=PAL["ink2"])
    cax = fig.add_axes([0.89, 0.14, 0.014, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("EV speed / desired", fontsize=8.5, color=PAL["ink2"])
    cb.ax.tick_params(labelsize=7.5, colors=PAL["muted"])
    cb.outline.set_visible(False)
    fig.suptitle("Depinning: EV progress over the corridor-push / "
                 "lane-keeping plane", fontsize=12, color=PAL["ink"], x=0.08,
                 ha="left", y=0.96)
    return _save(fig, "sens_surface.png")


def main():
    js, npz = _load()
    outs = []
    if "oat" in js:
        outs += [fig_tornado(js), fig_tornado_col(js), fig_curves(js, npz)]
    if "gsa" in js:
        outs.append(fig_gsa(js))
    if "reference" in js:
        outs.append(fig_reference(js))
    if "surf_A_c" in npz.files:
        outs.append(fig_surface(npz))
    log_run("paper_figs",
            note="SFM parameter-sensitivity figures (sens_*)",
            outputs=[os.path.relpath(o, ROOT) for o in outs])
    print("done:", len(outs))


if __name__ == "__main__":
    main()
