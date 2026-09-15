"""Figures for the parameter-sensitivity / behavioural-realism study.

Reads out/param_stats.{json,npz} (experiments/run_param_stats.py) and writes:

  out/fig_param_sensitivity.png   Spearman rho of every metric on every
                                  parameter, per scenario, + linear-surrogate R2
  out/fig_param_response.png      one-at-a-time response of EV progress with
                                  bootstrap CIs, calibrated point marked
  out/fig_depinning_surface.png   A_c x a_pin surface vs the analytic escape
                                  boundary and cleared half-width d*
  out/fig_behaviour_vs_literature.png  model observables against dashcam ground
                                  truth and naturalistic-driving statistics

Style follows emv.viz (same palette as the papers): sequential = one hue for
magnitude, diverging blue-grey-red for signed correlation, categorical hues in
fixed order for the model configurations.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from emv import empirical, viz
from emv.params import Params, tuned_params
from emv.runlog import log_run
from run_param_stats import METRIC_SPEC, MNAMES, PARAM_SPEC, PNAMES, boot_ci

plt = viz.plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PAL = viz.LIGHT
CAT = viz.CAT_LIGHT

MLABEL = {k: lab for k, lab, _ in METRIC_SPEC}
PLABEL = {n: lab for n, _, _, lab in PARAM_SPEC}
UNITS = dict(A_c="(m/s$^2$)", B_c="(m)", a_pin="(m/s$^2$)", A_ev="(m/s$^2$)",
             B_ev="(m)", T_react="(s)", R_front="(m)", p_noncomply="(-)")
SCEN = (("o_", "free-flow overtake"), ("j_", "stop-and-go jam"))

# --------------------------------------------------------------- references
#: Empirical anchors. Naturalistic-driving statistics describe *normal* motorway
#: driving, so an evasive yielding manoeuvre is expected in the upper tail - the
#: bands are context for reading the model, not calibration targets.
#: Now served by emv/empirical.py, which is the single source of truth for both
#: the cited anchors and the highD values measured by this repo. figure_bands()
#: returns the cited values only and is byte-identical to the dict that used to
#: live here, so this migration changes no existing figure (asserted in
#: tests/test_highd.py).
LIT = empirical.figure_bands()
#: Dashcam ground truth (out/survey_compare.json, reviewed and rules-only
#: relabelled). Ego-frame, 1 Hz, and bounded by the annotator's 50 m range.
GT = empirical.gt()
CFG_ORDER = ("tuned", "default", "game", "no_yield")
CFG_COLOR = {"tuned": CAT["blue"], "default": CAT["aqua"], "game": CAT["yellow"],
             "no_yield": CAT["red"]}
CFG_LABEL = {"tuned": "calibrated", "default": "hand-set defaults",
             "game": "game preset", "no_yield": "no-yielding control"}


def _load():
    with open(os.path.join(OUT, "param_stats.json")) as fh:
        js = json.load(fh)
    npz = np.load(os.path.join(OUT, "param_stats.npz"))
    return js, npz


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=160)
    plt.close(fig)
    print("wrote", path, flush=True)
    return path


# ============================================================ 1. sensitivity
def fig_sensitivity(js, npz):
    rho, r2 = npz["gsa_rho"], npz["gsa_r2"]
    keys = js["gsa"]["keys"]
    fig = plt.figure(figsize=(13.4, 6.4), dpi=160)
    fig.patch.set_facecolor(PAL["page"])
    gs = fig.add_gridspec(2, 2, height_ratios=[7, 1.2], hspace=0.08, wspace=0.10,
                          left=0.085, right=0.88, top=0.855, bottom=0.30)
    im = None
    for c, (pre, title) in enumerate(SCEN):
        cols = [keys.index(pre + m) for m in MNAMES if pre + m in keys]
        labels = [MLABEL[m] for m in MNAMES if pre + m in keys]
        R = rho[:, cols]
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(R, cmap=viz.CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([])                            # labelled under the R2 strip
        ax.set_yticks(range(len(PNAMES)))
        ax.set_yticklabels([PLABEL[p] for p in PNAMES] if c == 0 else [],
                           fontsize=9, color=PAL["ink2"])
        ax.set_title(title, fontsize=10, color=PAL["ink"], pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        for i in range(R.shape[0]):                  # label only what matters
            for j in range(R.shape[1]):
                if np.isfinite(R[i, j]) and abs(R[i, j]) >= 0.25:
                    ax.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6.6, color="#ffffff" if abs(R[i, j]) > 0.62
                            else PAL["ink"])
        axr = fig.add_subplot(gs[1, c])
        viz._style_ax(axr, PAL)
        axr.bar(range(len(cols)), np.nan_to_num(r2[cols]), color=CAT["blue"],
                width=0.66)
        axr.set_ylim(0, 1)
        axr.set_xlim(-0.5, len(cols) - 0.5)
        axr.set_xticks(range(len(labels)))
        axr.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.6,
                            color=PAL["ink2"])
        axr.set_yticks([0, 0.5, 1.0])
        axr.axhline(0.5, color=PAL["axis"], lw=0.8)
        axr.set_ylabel("$R^2$" if c == 0 else "", fontsize=8.5)
        axr.tick_params(labelsize=7.5)
    cax = fig.add_axes([0.895, 0.42, 0.013, 0.43])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman $\\rho$  (parameter $\\to$ metric)", fontsize=8.5,
                 color=PAL["ink2"])
    cb.ax.tick_params(labelsize=7.5, colors=PAL["muted"])
    cb.outline.set_visible(False)
    fig.suptitle("Global sensitivity: rank correlation of each metric with each "
                 "parameter", fontsize=12, color=PAL["ink"], x=0.085, ha="left",
                 y=0.978)
    n = npz["gsa_X"].shape[0]
    sig = 1.96 / np.sqrt(max(n - 1, 2))          # ~95 % threshold for rho = 0
    fig.text(0.085, 0.935,
             f"Latin-hypercube design, N = {n} parameter vectors x "
             f"{len(js['design']['seeds_gsa'])} common random seeds; "
             f"|$\\rho$| > {sig:.2f} is significant at 95 %. Cells labelled where "
             "|$\\rho$| $\\geq$ 0.25. Bars: $R^2$ of the linear surrogate - where "
             "it is low the effect is non-linear and only the rank correlation "
             "and the response curves are meaningful.",
             fontsize=8, color=PAL["muted"], ha="left")
    return _save(fig, "fig_param_sensitivity.png")


# ============================================================= 2. OAT curves
def fig_response(js, npz):
    keys = js["oat"]["keys"]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 5.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(PAL["page"])
    for ax, (name, lo, hi, _lab) in zip(axes.ravel(), PARAM_SPEC):
        viz._style_ax(ax, PAL)
        grid = npz[f"oat_{name}_grid"]
        Y = npz[f"oat_{name}_Y"]                     # (levels, seeds, metrics)
        for (pre, label), col in zip(SCEN, (CAT["blue"], CAT["orange"])):
            ci = keys.index(pre + "ev_speed_ratio")
            med = np.array([np.nanmean(Y[i, :, ci]) for i in range(len(grid))])
            lohi = np.array([boot_ci(Y[i, :, ci]) for i in range(len(grid))])
            ax.fill_between(grid, lohi[:, 0], lohi[:, 1], color=col, alpha=0.18,
                            lw=0)
            ax.plot(grid, med, color=col, lw=2.0, marker="o", ms=3.5,
                    label=label.split()[0])
        ax.axvline(js["oat"]["base"][name], color=PAL["muted"], lw=1.0,
                   ls=(0, (4, 4)))
        ax.set_title(f"{PLABEL[name]}   {UNITS[name]}", fontsize=9.5,
                     color=PAL["ink"])
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(0, 1.05)
    for ax in axes[:, 0]:
        ax.set_ylabel("EV speed / desired", fontsize=8.5)
    axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=False,
                      labelcolor=PAL["ink2"])
    fig.suptitle("Response of EV progress to each parameter, others held at the "
                 "calibrated value", fontsize=12, color=PAL["ink"], x=0.055,
                 ha="left", y=1.005)
    fig.text(0.055, 0.955,
             f"Bands: 95 % percentile bootstrap over "
             f"{len(js['design']['seeds_oat'])} seeds - indicative spread, not an "
             "inferential interval at this sample size. Dashed line: calibrated "
             "value (emv/params.py TUNED). Flat panels mean the calibrated point "
             "sits on a plateau for that parameter.",
             fontsize=8, color=PAL["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig_param_response.png")


# ======================================================== 3. depinning surface
def fig_surface(js, npz):
    A_c, a_pin = npz["surf_A_c"], npz["surf_a_pin"]
    p = tuned_params()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=160)
    fig.patch.set_facecolor(PAL["page"])
    fig.subplots_adjust(left=0.075, right=0.875, top=0.78, bottom=0.135,
                        wspace=0.13)
    im = None
    vmin = float(np.floor(min(npz["surf_overtake"][:, :, 0].min(),
                              npz["surf_jam"][:, :, 0].min()) * 20) / 20)
    for ax, (key, title) in zip(axes, (("surf_overtake", "free-flow overtake"),
                                       ("surf_jam", "stop-and-go jam"))):
        Z = npz[key][:, :, 0]                        # EV speed ratio
        viz._style_ax(ax, PAL)
        ax.grid(False)
        im = ax.pcolormesh(a_pin, A_c, Z, cmap=viz.CMAP_SEQ, vmin=vmin, vmax=1.0,
                           shading="nearest")
        # iso-performance contours: their slope is the empirical counterpart of
        # the analytic boundary, so the two can be compared by eye
        levels = [l for l in (0.5, 0.7, 0.85, 0.95) if vmin < l < 1.0]
        cs = ax.contour(a_pin, A_c, Z, levels=levels, colors="#f9f9f7",
                        linewidths=0.9, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f")
        # analytic escape condition at full urgency: A_c > a_pin (1 - relief)
        xs = np.linspace(a_pin[0], a_pin[-1], 100)
        ax.plot(xs, xs * (1.0 - p.urgency_pin_relief), color="#8f1d1d", lw=2.0,
                label="escape boundary  $A_c = a_{pin}(1-\\lambda_u)$")
        ax.plot(p.a_pin, p.A_c, marker="*", ms=14, mfc=CAT["yellow"],
                mec="#3a2a00", mew=0.8, ls="none", label="calibrated point")
        ax.set_xlim(a_pin[0], a_pin[-1])
        ax.set_ylim(A_c[0], A_c[-1])
        ax.set_xlabel("$a_{pin}$   lane-keeping strength (m/s$^2$)", fontsize=9)
        ax.set_title(title, fontsize=10, color=PAL["ink"])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("$A_c$   corridor push (m/s$^2$)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7.8, frameon=False,
                   labelcolor=PAL["ink2"])
    cax = fig.add_axes([0.895, 0.135, 0.014, 0.645])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("EV speed / desired", fontsize=8.5, color=PAL["ink2"])
    cb.ax.tick_params(labelsize=7.5, colors=PAL["muted"])
    cb.outline.set_visible(False)
    fig.suptitle("Depinning: EV progress over the corridor-push / lane-keeping "
                 "plane", fontsize=12, color=PAL["ink"], x=0.075, ha="left",
                 y=0.975)
    fig.text(0.075, 0.885,
             "Theory (docs/FRAMEWORK.md sec. 4): a yielding car escapes its lane "
             "only where $A_c u > a_{pin}(1-\\lambda_u u)$; at full urgency that "
             "is the red line. Progress collapses below it.",
             fontsize=8, color=PAL["muted"], ha="left")
    return _save(fig, "fig_depinning_surface.png")


# ================================================== 4. behaviour vs literature
def fig_literature(js, _npz):
    ref = js["reference"]
    panels = [
        ("lc_dur_med", "lane-change duration (s)", (0, 8)),
        ("lc_per_veh_km", "lane changes per veh-km", (0, 0.4)),
        ("lc_peak_vy_med", "peak lateral speed (m/s)", (0, 2.4)),
        ("yield_peak_decel_med", "peak deceleration, yielding (m/s$^2$)", (0, 5)),
        ("yield_peak_alat_med", "peak lateral accel., yielding (m/s$^2$)", (0, 3)),
        ("react_dist_mean", "yield onset distance (m)", (0, 200)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.0), dpi=160)
    fig.patch.set_facecolor(PAL["page"])
    for ax, (key, label, xlim) in zip(axes.ravel(), panels):
        viz._style_ax(ax, PAL)
        lit = LIT.get(key, {})
        if lit.get("lo") is not None:
            ax.axvspan(lit["lo"], lit["hi"], color=PAL["grid"], zorder=0)
        if lit.get("mid") is not None:
            ax.axvline(lit["mid"], color=PAL["muted"], lw=1.4, ls=(0, (5, 4)),
                       zorder=1)
        ys = []
        for i, cfg in enumerate(CFG_ORDER):
            if cfg not in ref:
                continue
            y = len(CFG_ORDER) - i
            ys.append((y, cfg))
            # scenario as a secondary encoding: filled = free-flow, open = jam,
            # so colour stays free to identify the configuration
            for sc, dy, filled in (("overtake", 0.17, True), ("jam", -0.17, False)):
                m, lo, hi = ref[cfg][sc][key]
                if not np.isfinite(m):
                    if sc == "overtake":
                        ax.text(xlim[0] + 0.03 * (xlim[1] - xlim[0]), y,
                                "none observed", fontsize=6.4,
                                color=PAL["muted"], va="center", style="italic")
                    continue
                ax.plot([lo, hi], [y + dy] * 2, color=CFG_COLOR[cfg], lw=2.2,
                        solid_capstyle="round", zorder=3)
                ax.plot(m, y + dy, "o", ms=6.5, zorder=4, mew=1.5,
                        mfc=CFG_COLOR[cfg] if filled else PAL["surface"],
                        mec=PAL["surface"] if filled else CFG_COLOR[cfg])
        if key == "react_dist_mean":                 # dashcam GT + its ceiling
            ax.axvline(GT["onset_med"], color="#0b0b0b", lw=1.4, zorder=2)
            ax.axvline(GT["range_m"], color=CAT["red"], lw=1.2, ls=(0, (2, 3)),
                       zorder=2)
            ax.text(GT["range_m"] + 3, 1.28, "dashcam\ndetection limit",
                    fontsize=6.4, color=CAT["red"], va="bottom", linespacing=1.2)
            ax.text(GT["onset_med"] + 3, len(CFG_ORDER) + 0.45, "dashcam GT",
                    fontsize=6.8, color=PAL["ink"])
        if key == "lc_peak_vy_med":
            ax.axvline(GT["lat_speed_p90"], color="#0b0b0b", lw=1.4, zorder=2)
            ax.text(GT["lat_speed_p90"] + 0.05, len(CFG_ORDER) + 0.45,
                    "dashcam GT p90", fontsize=6.8, color=PAL["ink"])
        ax.set_yticks([y for y, _ in ys])
        ax.set_yticklabels([CFG_LABEL[c] for _, c in ys], fontsize=7.8,
                           color=PAL["ink2"])
        ax.set_ylim(0.4, len(CFG_ORDER) + 0.9)
        ax.set_xlim(xlim[0] - 0.02 * (xlim[1] - xlim[0]), xlim[1])
        ax.set_xlabel(label, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", visible=False)
        if lit.get("note"):
            ax.text(0.98, 0.06, lit["note"], transform=ax.transAxes, fontsize=6.6,
                    color=PAL["muted"], ha="right", va="bottom", linespacing=1.3)
    h_fill = plt.Line2D([], [], marker="o", ls="none", ms=6.5, mfc="#52514e",
                        mec=PAL["surface"], mew=1.5, label="free-flow overtake")
    h_open = plt.Line2D([], [], marker="o", ls="none", ms=6.5, mfc=PAL["surface"],
                        mec="#52514e", mew=1.5, label="stop-and-go jam")
    axes[0, 0].legend(handles=[h_fill, h_open], loc="upper right", fontsize=6.8,
                      frameon=False, labelcolor=PAL["ink2"], handletextpad=0.4)
    fig.suptitle("Model behaviour against dashcam ground truth and "
                 "naturalistic-driving statistics", fontsize=12,
                 color=PAL["ink"], x=0.045, ha="left", y=1.005)
    fig.text(0.045, 0.955,
             f"{len(js['design']['seeds_reference'])} seeds per configuration; "
             "dots = mean, bars = 95 % bootstrap CI. Grey band / dashed line = "
             "empirical reference for *normal* driving, so an evasive manoeuvre "
             "is expected in its upper tail.",
             fontsize=8, color=PAL["muted"], ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig_behaviour_vs_literature.png")


# ===================================================================== tables
def write_tables(js, npz):
    """Markdown tables: the figures' numbers, for the thesis/paper text."""
    lines = ["# Parameter-sensitivity and behavioural-realism study", "",
             "Generated by `experiments/make_param_stats_figs.py` from "
             "`out/param_stats.json`. Design: "
             f"LHS N={js['design']['n_lhs']} x {len(js['design']['seeds_gsa'])} "
             f"seeds (GSA), {js['design']['levels']} levels x "
             f"{len(js['design']['seeds_oat'])} seeds (response curves), "
             f"{js['design']['grid']}x{js['design']['grid']} x "
             f"{len(js['design']['seeds_surface'])} seeds (surface), "
             f"{len(js['design']['seeds_reference'])} seeds (reference).", ""]

    if "gsa" in js:
        lines += ["## 1. Global sensitivity (Spearman rho, |rho| >= 0.3)", "",
                  "| metric | scenario | strongest parameters |", "|---|---|---|"]
        for m, label, _ in METRIC_SPEC:
            for pre, scen in SCEN:
                key = pre + m
                if key not in js["gsa"]["rho"]:
                    continue
                r = js["gsa"]["rho"][key]
                top = sorted(r.items(), key=lambda kv: -abs(kv[1]))[:3]
                txt = ", ".join(f"{k} {v:+.2f}" for k, v in top if abs(v) >= 0.3)
                lines.append(f"| {label} | {scen.split()[0]} | {txt or '-'} |")
        lines += ["", "$R^2$ of the linear surrogate per metric: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in
                              sorted(js["gsa"]["r2"].items())[:8]) + " ...", ""]

    if "reference" in js:
        lines += ["## 2. Behavioural observables vs empirical references", "",
                  "Mean [95 % bootstrap CI] over "
                  f"{len(js['design']['seeds_reference'])} seeds.", "",
                  "| observable | scenario | " +
                  " | ".join(CFG_LABEL[c] for c in CFG_ORDER if c in js["reference"])
                  + " | empirical reference |", "|---|---|" +
                  "---|" * (len([c for c in CFG_ORDER if c in js["reference"]]) + 1)]
        show = ["lc_dur_med", "lc_per_veh_km", "lc_peak_vy_med", "lat_speed_med",
                "yield_peak_decel_med", "yield_peak_alat_med", "react_dist_mean",
                "ev_med_speed", "bg_med_speed", "min_ttc"]
        for key in show:
            if key not in MNAMES:
                continue
            for sc in ("overtake", "jam"):
                cells = []
                for cfg in CFG_ORDER:
                    if cfg not in js["reference"]:
                        continue
                    m, lo, hi = js["reference"][cfg][sc][key]
                    if not np.isfinite(m):
                        cells.append("n/a")          # observable never occurred
                    elif not np.isfinite(lo):
                        cells.append(f"{m:.2f} (1 run)")   # too few to bootstrap
                    else:
                        cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
                note = LIT.get(key, {}).get("note", "").replace("\n", " ")
                lines.append(f"| {MLABEL[key]} | {sc} | " + " | ".join(cells)
                             + f" | {note or '-'} |")
        lines += ["", "Ground truth (dashcam, ego-frame, 1 Hz, 50 m range): "
                  f"yield onset median {GT['onset_med']:.1f} m "
                  f"({GT['onset_med_rules']:.1f} m rules-only relabelled), "
                  f"lateral speed median {GT['lat_speed_med']:.2f} m/s "
                  f"(p90 {GT['lat_speed_p90']:.2f}).", ""]

    path = os.path.join(OUT, "param_stats_tables.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path, flush=True)
    return path


if __name__ == "__main__":
    js, npz = _load()
    outs = []
    if "gsa" in js:
        outs.append(fig_sensitivity(js, npz))
    if "oat" in js:
        outs.append(fig_response(js, npz))
    if "surface" in js:
        outs.append(fig_surface(js, npz))
    if "reference" in js:
        outs.append(fig_literature(js, npz))
    outs.append(write_tables(js, npz))
    log_run("param_stats_figs", note="sensitivity/response/surface/literature",
            outputs=outs)
