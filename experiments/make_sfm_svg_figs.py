"""Explanatory graphics of the social force model -> images/*.svg (editable).

    python experiments/make_sfm_svg_figs.py [--png]

Eight standalone figures, each answering one question about the formulation.
They are written as **SVG with live text** (`svg.fonttype='none'`, so every
label stays a `<text>` element that can be re-typed in Inkscape/Illustrator),
and the quantitative panels are drawn from the *code that runs*: every profile,
map and threshold is evaluated with the `emv.forces` / `emv.perception`
expressions at the two-stage calibrated parameters (`params.TUNED_BLUELIGHT`).
Only the plan views are schematic; there, distances are to scale but vehicles
are drawn longer than scale (`CAR_EXAG`), because a 4.6 m car on a 200 m road
is one pixel wide.

  images/sfm_00_symbols.svg         every symbol defined, with its value and origin
  images/sfm_01_overview.svg        all six accelerations acting at once, in situ
  images/sfm_02_composition.svg     how the terms compose into one acceleration
  images/sfm_03_car_car.svg         elliptic vehicle-vehicle repulsion + IDM bound
  images/sfm_04_ev_field.svg        the forward-focused emergency-vehicle field
  images/sfm_05_corridor.svg        the predicted-corridor term: geometry, profiles
  images/sfm_06_depinning.svg       lane keeping as a periodic potential; escape
  images/sfm_07_perception.svg      awareness state machine + the urgency channels
  images/sfm_08_identification.svg  two-stage identification and the adequacy bar

Layout rules (the v1 pass violated all four and had to be redone):
  * a panel heading is an axes title, never free text near another panel;
  * every figure carries its explanation in a caption strip at the bottom, so
    inline annotation can stay short (symbols, not sentences);
  * a map panel keeps `set_aspect('equal')` - a distorted field map is a wrong
    field map - and long-range behaviour goes into a profile panel instead;
  * every label carries a page-coloured halo, because labels sit near lines.

`--png` also writes a raster preview next to each SVG (for eyeballing only).
Self-logs kind='paper_figs'.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from emv import viz
from emv.params import Params, TUNED_BLUELIGHT
from emv.runlog import log_run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG = os.path.join(ROOT, "images")
os.makedirs(IMG, exist_ok=True)

PAL, CAT = viz.LIGHT, viz.CAT_LIGHT
P = Params(**TUNED_BLUELIGHT)          # the two-stage calibrated model

C_EV, C_CAR = CAT["red"], CAT["blue"]
C_CORR, C_LANE, C_CC = CAT["aqua"], CAT["orange"], CAT["blue"]
C_DRIVE, C_IDM = CAT["violet"], CAT["yellow"]
INK, INK2, MUTED = PAL["ink"], PAL["ink2"], PAL["muted"]
STATE_C = {"UNAWARE": MUTED, "NOTICED": CAT["yellow"],
           "YIELDING": CAT["orange"], "HOLD": CAT["aqua"]}

W_LANE = 3.5
CAR_L, CAR_W = 4.6, 1.9
CAR_EXAG = 3.4                          # plan views only: drawn length / true length
HALO = [pe.withStroke(linewidth=2.4, foreground=PAL["page"])]

plt.rcParams.update({
    "svg.fonttype": "none",             # keep text as text: editable afterwards
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
})


# ----------------------------------------------------------------- helpers
def _save(fig, name, png=False):
    out = os.path.join(IMG, name)
    if "symbols" not in name:            # every other sheet points at the key
        fig.text(0.982, 0.020, "symbols are defined in sfm_00_symbols.svg",
                 fontsize=7.2, color=MUTED, ha="right", va="bottom")
    fig.savefig(out, format="svg", facecolor=PAL["page"])
    if png:
        fig.savefig(out[:-4] + ".png", dpi=150, facecolor=PAL["page"])
    plt.close(fig)
    print("wrote", os.path.relpath(out, ROOT), flush=True)
    return out


def _head(fig, title, sub=None, y=0.965):
    fig.text(0.018, y, title, fontsize=11.0, fontweight="bold", color=INK,
             va="top")
    if sub:
        fig.text(0.018, y - 0.052, sub, fontsize=8.4, color=INK2, va="top")


def _caption(fig, lines, y=None):
    """Explanation strip at the foot of the figure (one call per figure).

    The text is re-wrapped here rather than by hand: a 7.9 pt DejaVu line runs
    off a 9.4 in page past ~150 characters, and hand-wrapped captions drifted
    over that limit on every edit. Spaces inside `$...$` are protected first,
    so a line break can never split a mathtext group and leave an unbalanced
    delimiter. Lines are anchored at the bottom edge and grow upwards.
    """
    import re
    import textwrap
    text = " ".join(lines)
    text = re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(" ", "\x00"), text)
    wrapped = textwrap.wrap(text, width=150)
    y0 = 0.020 if y is None else y
    for k, ln in enumerate(reversed(wrapped)):
        fig.text(0.018, y0 + k * 0.036, ln.replace("\x00", " "), fontsize=7.9,
                 color=INK2, va="bottom")


def _title(ax, s, fs=8.8):
    ax.set_title(s, loc="left", fontsize=fs, fontweight="bold", color=INK, pad=5)


def _txt(ax, x, y, s, c=INK, ha="left", va="center", fs=7.8, halo=True,
         weight="normal", rot=0, z=9):
    ax.text(x, y, s, color=c, fontsize=fs, ha=ha, va=va, zorder=z, rotation=rot,
            fontweight=weight, linespacing=1.35,
            path_effects=HALO if halo else None)


def _arrow(ax, p0, p1, c, lw=1.5, ms=9.0, ls="-", z=7, rad=None):
    kw = {} if rad is None else dict(connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(FancyArrowPatch(p0, p1, color=c, lw=lw, arrowstyle="-|>",
                                 mutation_scale=ms, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0, **kw))


def _dim(ax, p0, p1, c=MUTED, lw=0.9, ms=6.0):
    ax.add_patch(FancyArrowPatch(p0, p1, color=c, lw=lw, arrowstyle="<|-|>",
                                 mutation_scale=ms, zorder=7,
                                 shrinkA=0, shrinkB=0))


def _car(ax, x, y, c, L=CAR_L, W=CAR_W, exag=1.0, alpha=1.0, z=6):
    Ld = L * exag
    ax.add_patch(Rectangle((x - Ld / 2, y - W / 2), Ld, W, facecolor=c,
                           edgecolor=c, lw=0.6, alpha=alpha, zorder=z))


def _state_dot(ax, x, y, state, ms=5.5):
    ax.plot([x], [y], "o", ms=ms, color=STATE_C[state], zorder=8,
            markeredgecolor=PAL["page"], markeredgewidth=0.7)


def _road(ax, x0, x1, y_lo=-2.6, y_hi=13.0, n_lanes=3):
    ax.add_patch(Rectangle((x0, 0.0), x1 - x0, n_lanes * W_LANE,
                           facecolor=PAL["surface"], edgecolor="none", zorder=0.4))
    for k in range(1, n_lanes):
        ax.plot([x0, x1], [k * W_LANE] * 2, color=PAL["axis"], lw=0.9,
                ls=(0, (7, 6)), zorder=1)
    for yb in (0.0, n_lanes * W_LANE):
        ax.plot([x0, x1], [yb] * 2, color=PAL["axis"], lw=1.3, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    ax.set_facecolor(PAL["page"])


def _box(ax, x, y, w, h, text, fc, ec=None, fs=8.0, tc=None, lw=1.1, z=4,
         weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec or fc, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, zorder=z + 1, fontweight=weight, linespacing=1.5)


def _light(c, a=0.16):
    return (*matplotlib.colors.to_rgb(c), a)


def _clean(ax, xlabel="", ylabel=""):
    viz._style_ax(ax, PAL)
    ax.tick_params(labelsize=7.6)
    ax.set_xlabel(xlabel, fontsize=8.2, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8.2, labelpad=2)


def _blank(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAL["page"])
    return ax


# =====================================================================
# 1. overview: every term acting at once
# =====================================================================
def fig_overview(png):
    fig = plt.figure(figsize=(9.4, 5.0), facecolor=PAL["page"])
    _head(fig, "The six accelerations of the model, acting at once",
          "One vehicle is shown with the full set; the others carry the "
          "awareness state that switches the two emergency-vehicle terms on.")
    ax = fig.add_axes([0.022, 0.315, 0.955, 0.545])
    x0, x1 = -52.0, 176.0
    _road(ax, x0, x1, y_lo=-6.2, y_hi=15.4)

    y_ev, x_ev = 1.5 * W_LANE, 16.0
    w_need = CAR_W + P.margin_c                    # 0.5(W_i+W_ev)+margin = 2.4 m

    # ---- predicted corridor, running off the right edge ---------------
    ax.add_patch(Rectangle((x_ev - P.L_back, y_ev - w_need), x1 - x_ev + P.L_back,
                           2 * w_need, facecolor=C_CORR, alpha=0.13,
                           edgecolor="none", zorder=2))
    for sgn in (-1, 1):
        ax.plot([x_ev - P.L_back, x1], [y_ev + sgn * w_need] * 2, color=C_CORR,
                lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.plot([x_ev, x1], [y_ev] * 2, color=C_CORR, lw=0.8, ls=(0, (1, 2.4)),
            zorder=3)
    _arrow(ax, (128.0, 12.6), (174.0, 12.6), C_CORR, lw=1.0, ms=7)
    _txt(ax, 126.0, 12.6, "predicted corridor, length $L = 260$ m", C_CORR,
         ha="right", fs=8.0)

    # ---- detection ranges (what turns the perception state machine on) -
    ax.plot([x_ev - P.R_rear, x_ev + P.R_front], [14.4] * 2, color=CAT["yellow"],
            lw=1.2, solid_capstyle="butt", zorder=5)
    for xx in (x_ev - P.R_rear, x_ev + P.R_front):
        ax.plot([xx, xx], [13.9, 14.9], color=CAT["yellow"], lw=1.2, zorder=5)
    _txt(ax, x_ev - P.R_rear + 3, 14.4, f"$R_{{rear}} = {P.R_rear:.0f}$ m",
         CAT["yellow"], fs=7.8)
    _txt(ax, x_ev + P.R_front - 3, 14.4, f"$R_{{front}} = {P.R_front:.0f}$ m",
         CAT["yellow"], ha="right", fs=7.8)

    # ---- vehicles -----------------------------------------------------
    _car(ax, x_ev, y_ev, C_EV, L=5.4, W=2.0, exag=CAR_EXAG)
    _txt(ax, x_ev - 12.0, y_ev, "EV", C_EV, ha="right", fs=8.4, weight="bold")

    xh, yh = 66.0, 4.35                            # the vehicle under study
    _car(ax, xh, yh, C_CAR, exag=CAR_EXAG)
    _state_dot(ax, xh, yh + 1.9, "YIELDING")
    _car(ax, 140.0, 2.5 * W_LANE, C_CAR, exag=CAR_EXAG)      # already clear
    _state_dot(ax, 140.0, 2.5 * W_LANE + 1.9, "YIELDING")
    _car(ax, 166.0, 2.5 * W_LANE, PAL["muted"], exag=CAR_EXAG, alpha=0.8)
    _car(ax, 158.0, 0.5 * W_LANE, PAL["muted"], exag=CAR_EXAG, alpha=0.55)
    _state_dot(ax, 158.0, 0.5 * W_LANE + 1.9, "NOTICED")
    _car(ax, -30.0, 0.5 * W_LANE, C_CAR, exag=CAR_EXAG, alpha=0.75)
    _state_dot(ax, -30.0, 0.5 * W_LANE + 1.9, "HOLD")
    _car(ax, -44.0, 2.5 * W_LANE, PAL["muted"], exag=CAR_EXAG, alpha=0.5)
    _state_dot(ax, -44.0, 2.5 * W_LANE + 1.9, "UNAWARE")

    # ---- the five accelerations on that vehicle (symbols only) --------
    _arrow(ax, (xh + 9.0, yh), (xh + 26.0, yh), C_DRIVE)
    _txt(ax, xh + 27.5, yh - 0.05, r"$F^{drive}$", C_DRIVE, fs=8.4)
    _arrow(ax, (xh - 4.0, yh - 1.1), (xh - 4.0, yh - 4.3), C_CORR)
    _txt(ax, xh - 6.0, yh - 3.6, r"$F^{corr}_{\perp}$", C_CORR, ha="right", fs=8.4)
    _arrow(ax, (xh - 9.5, yh + 0.15), (xh - 27.0, yh + 0.15), C_CORR)
    _txt(ax, xh - 28.5, yh + 0.15, r"$F^{corr}_{\parallel}$", C_CORR, ha="right",
         fs=8.4)
    _arrow(ax, (xh + 9.0, yh + 1.25), (xh + 24.0, yh + 2.35), C_EV)
    _txt(ax, xh + 25.5, yh + 2.5, r"$F^{EV}$", C_EV, fs=8.4)
    _arrow(ax, (xh + 2.0, yh + 1.1), (xh + 2.0, yh + 4.1), C_LANE)
    _txt(ax, xh + 3.5, yh + 3.9, r"$F^{lane}$", C_LANE, fs=8.4)

    # F^cc shown on a different vehicle, so no two arrows share a slot
    _arrow(ax, (140.0 - 9.0, 2.5 * W_LANE - 0.7), (140.0 - 22.0, 2.5 * W_LANE - 1.6),
           C_CC, ls=(0, (3.5, 2)))
    _txt(ax, 140.0 - 23.5, 2.5 * W_LANE - 1.8, r"$F^{cc}$", C_CC, ha="right",
         fs=8.4)

    # ---- geometry -----------------------------------------------------
    _dim(ax, (110.0, y_ev), (110.0, y_ev - w_need), C_CORR)
    _txt(ax, 111.8, y_ev - w_need / 2, r"$w_{need}$", C_CORR, fs=8.0)
    _dim(ax, (140.0, y_ev), (140.0, 2.5 * W_LANE - 1.1), INK2)
    _txt(ax, 141.8, y_ev + 1.1, r"$d_{\perp}$", INK2, fs=8.0)
    _dim(ax, (x_ev, -4.4), (xh, -4.4), INK2)
    _txt(ax, (x_ev + xh) / 2, -5.4, r"$s$", INK2, ha="center", fs=8.0)
    _txt(ax, xh + 4, -5.4, "distance ahead of the EV nose", MUTED, fs=7.6)

    # ---- key strip ----------------------------------------------------
    kax = _blank(fig, [0.022, 0.145, 0.955, 0.135])
    key = [(r"$F^{drive}$", "relaxation to the desired speed", C_DRIVE),
           (r"$F^{cc}$", "repulsion from every neighbour", C_CC),
           (r"$F^{lane}$", "lane keeping (periodic potential)", C_LANE),
           (r"$F^{EV}$", "point field of the emergency vehicle", C_EV),
           (r"$F^{corr}_{\perp}$", "push out of the predicted corridor", C_CORR),
           (r"$F^{corr}_{\parallel}$", "merge brake while still blocking it", C_CORR)]
    for k, (sym, desc, col) in enumerate(key):
        cx, cy = 0.005 + (k % 3) * 0.325, 0.72 - (k // 3) * 0.55
        kax.plot([cx, cx + 0.020], [cy, cy], color=col, lw=2.4, zorder=5,
                 solid_capstyle="round")
        kax.text(cx + 0.028, cy, sym, color=col, fontsize=8.2, va="center")
        kax.text(cx + 0.076, cy, desc, color=INK2, fontsize=7.7, va="center")
    for k, (name, col) in enumerate(STATE_C.items()):
        cx = 0.005 + k * 0.130
        kax.plot([cx], [-0.42], "o", ms=5.5, color=col, zorder=5)
        kax.text(cx + 0.014, -0.42, name, color=INK2, fontsize=7.7, va="center")
    kax.text(0.545, -0.42, "awareness state: gates $F^{EV}$ and $F^{corr}$",
             color=MUTED, fontsize=7.7, va="center")
    kax.set_ylim(-0.75, 1.0)

    _caption(fig, [
        "Distances are to scale; vehicle length is drawn 3.4x true size to stay "
        "visible. The corridor follows the EV's predicted path, not its position, "
        "and runs 120-260 m ahead of it.",
        "Only drivers in YIELDING feel $F^{EV}$ and $F^{corr}$, and 5 % of them "
        "never leave UNAWARE - a corridor that fails to form comes out of the same "
        "equations that make one form.",
    ])
    return _save(fig, "sfm_01_overview.svg", png)


# =====================================================================
# 2. composition: from terms to one acceleration
# =====================================================================
def fig_composition(png):
    fig = plt.figure(figsize=(9.4, 4.9), facecolor=PAL["page"])
    _head(fig, "How the terms compose into one acceleration per vehicle",
          "The longitudinal branch takes a minimum, the lateral branch a sum. "
          "That asymmetry is the safety argument of the model.")
    ax = _blank(fig, [0.0, 0.10, 1.0, 0.78])

    terms = [
        (r"$F^{drive} = (v^0\hat{x}-\vec{v})/\tau$", "relaxation to desired speed",
         C_DRIVE),
        (r"$F^{cc}$", "elliptic repulsion + near-field body term", C_CC),
        (r"$F^{lane},\ F^{pass}$", "lane keeping, discretionary lane change", C_LANE),
        (r"$F^{EV}$", "emergency-vehicle point field", C_EV),
        (r"$F^{corr}_{\perp},\ F^{corr}_{\parallel}$",
         "predicted corridor: push out, merge brake", C_CORR),
    ]
    ys = [0.845, 0.685, 0.525, 0.215, 0.045]
    for (sym, desc, col), y in zip(terms, ys):
        ax.add_patch(FancyBboxPatch((0.055, y), 0.245, 0.115,
                                    boxstyle="round,pad=0.006,rounding_size=0.05",
                                    facecolor=_light(col), edgecolor=col, lw=1.1,
                                    zorder=4))
        ax.text(0.177, y + 0.077, sym, ha="center", va="center", fontsize=8.6,
                color=INK, zorder=5)
        ax.text(0.177, y + 0.032, desc, ha="center", va="center", fontsize=7.5,
                color=INK2, zorder=5)

    # the gate, placed between the two blocks it gates
    _box(ax, 0.055, 0.365, 0.245, 0.105,
         "perception state machine\n"
         r"gate $g\in\{0,1\}$,  urgency $u\in[0,1]$",
         _light(CAT["yellow"], 0.22), CAT["yellow"], fs=7.9)
    for y_to in (0.335, 0.165):
        ax.add_patch(FancyArrowPatch((0.045, 0.418), (0.05, y_to),
                                     connectionstyle="angle,angleA=0,angleB=90,rad=6",
                                     color=CAT["yellow"], lw=1.1, ls=(0, (3, 2)),
                                     arrowstyle="-|>", mutation_scale=8, zorder=6))
    _txt(ax, 0.010, 0.265, "gates\nand\nscales", CAT["yellow"], fs=7.4, ha="left")

    # branches
    _box(ax, 0.375, 0.660, 0.265, 0.135,
         "$\\Sigma$ longitudinal\n$F^{cc}_x + F^{EV}_x + F^{corr}_x$",
         PAL["surface"], PAL["axis"], fs=8.4)
    _box(ax, 0.375, 0.215, 0.265, 0.135,
         "$\\Sigma$ lateral\n$F^{lane}_y + F^{cc}_y + F^{EV}_y + F^{corr}_y$",
         PAL["surface"], PAL["axis"], fs=8.4)
    for y in ys:
        _arrow(ax, (0.302, y + 0.058), (0.372, 0.283), MUTED, lw=0.8, ms=6, z=3)
    for y in (0.685, 0.215, 0.045):
        _arrow(ax, (0.302, y + 0.058), (0.372, 0.728), MUTED, lw=0.8, ms=6, z=3)

    _box(ax, 0.375, 0.870, 0.265, 0.105,
         r"$a^{IDM}$: safety demand from" + "\nthe most restrictive leader",
         _light(C_IDM, 0.24), C_IDM, fs=8.0)

    # composition
    _box(ax, 0.700, 0.660, 0.280, 0.135,
         r"$a_x = F^{drive}_x + \min(\Sigma_x,\ a^{IDM})$",
         _light(C_IDM, 0.10), C_IDM, fs=9.4, lw=1.5)
    _box(ax, 0.700, 0.215, 0.280, 0.135,
         r"$a_y = \Sigma_y$" + "\n(unbounded sum)", PAL["surface"], PAL["axis"],
         fs=9.0)
    _arrow(ax, (0.642, 0.728), (0.697, 0.728), INK2, lw=1.2)
    _arrow(ax, (0.642, 0.283), (0.697, 0.283), INK2, lw=1.2)
    _arrow(ax, (0.642, 0.905), (0.838, 0.800), C_IDM, lw=1.3)
    _txt(ax, 0.700, 0.955, "an envelope, not a term", C_IDM, fs=7.6)
    _txt(ax, 0.700, 0.610,
         "No social force can override a braking demand:\n"
         "collision avoidance is structural, not fitted.", INK2, fs=7.7,
         va="top")

    _box(ax, 0.700, 0.020, 0.280, 0.150,
         r"clamp $|a_y|$, $|v_y|$ to the driver's envelope"
         "\n" + r"$\vec{v} \leftarrow \vec{v} + \vec{a}\,\Delta t$,   "
         r"$\vec{r} \leftarrow \vec{r} + \vec{v}\,\Delta t$"
         "\nsemi-implicit Euler, $\\Delta t = 0.05$ s",
         PAL["surface"], PAL["axis"], fs=8.0)
    _arrow(ax, (0.840, 0.212), (0.840, 0.176), INK2, lw=1.2)
    _arrow(ax, (0.840, 0.657), (0.840, 0.355), INK2, lw=1.2)

    _caption(fig, [
        "Six force terms, two of them switched on by the driver's perception "
        "state, are summed into a longitudinal and a lateral demand.",
        "The longitudinal demand is then capped by the safety deceleration of the "
        "most restrictive leader, so the safety layer can only ever brake harder "
        "than the social terms ask.",
    ])
    return _save(fig, "sfm_02_composition.svg", png)


# =====================================================================
# 3. car-car: elliptic repulsion, near field, IDM bound
# =====================================================================
def fig_car_car(png):
    fig = plt.figure(figsize=(9.4, 4.1), facecolor=PAL["page"])
    _head(fig, "Vehicle-vehicle interaction: an elliptic field under a hard "
               "safety bound")
    axA = fig.add_axes([0.052, 0.30, 0.330, 0.52])
    cax = fig.add_axes([0.395, 0.375, 0.010, 0.30])
    axB = fig.add_axes([0.545, 0.30, 0.180, 0.52])
    axC = fig.add_axes([0.800, 0.30, 0.180, 0.52])

    def mag(dx, dy):
        """|F| the ego car at the origin receives from a car at (dx, dy)."""
        gx = np.maximum(np.abs(dx) - CAR_L, 0.05)
        gy = np.maximum(np.abs(dy) - CAR_W, 0.05)
        q = np.sqrt((gx / P.Bx_v) ** 2 + (gy / P.By_v) ** 2)
        m = P.A_v * np.exp(-q) + P.A_near * np.exp(-q / P.q_near)
        d = np.hypot(dx, dy) + 1e-9
        wgt = P.lam_v + (1 - P.lam_v) * 0.5 * (1 + dx / d)
        return np.minimum(m * wgt, P.F_cap_v)

    X, Y = np.meshgrid(np.linspace(-26, 26, 420), np.linspace(-9, 9, 260))
    Z = mag(X, Y)
    # imshow, not pcolormesh: a QuadMesh writes one SVG polygon per cell (20 MB
    # for this grid); imshow embeds a single raster and leaves the contours,
    # labels and axes vector, which is what an editable SVG needs.
    pcm = axA.imshow(Z, extent=(-26, 26, -9, 9), origin="lower", aspect="auto",
                     cmap=viz.CMAP_SEQ, interpolation="bilinear",
                     norm=LogNorm(vmin=0.02, vmax=P.F_cap_v), zorder=0)
    cs = axA.contour(X, Y, Z, levels=[0.05, 0.2, 0.8, 2.5], colors=[INK2],
                     linewidths=0.7, zorder=2)
    axA.clabel(cs, fmt="%.2g", fontsize=6.6, inline_spacing=2)
    _car(axA, 0, 0, C_CAR, z=5)
    _txt(axA, 0, 2.0, "ego", C_CAR, ha="center", fs=7.6)
    _car(axA, 13.0, 0.0, PAL["muted"], alpha=0.75, z=5)
    _car(axA, -13.0, 0.0, PAL["muted"], alpha=0.75, z=5)
    _txt(axA, 13.0, -7.2, "ahead:\nfull weight", INK2, ha="center", fs=7.0)
    _txt(axA, -13.0, -7.2, f"behind:\nweight $\\lambda_v = {P.lam_v}$", INK2,
         ha="center", fs=7.0)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("$|F^{cc}|$ (m s$^{-2}$)", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)
    _clean(axA, "longitudinal offset $\\Delta x$ (m)",
           "lateral offset $\\Delta y$ (m)")
    axA.grid(False)
    axA.set_aspect("equal")
    _title(axA, "(a)  the field one driver feels")

    g = np.linspace(0.05, 24, 500)
    axB.semilogy(g, P.A_v * np.exp(-g / P.Bx_v), color=C_CC, lw=1.9,
                 label=f"along the road, $B_x = {P.Bx_v:.1f}$ m")
    axB.semilogy(g, P.A_v * np.exp(-g / P.By_v), color=C_CC, lw=1.4, ls="--",
                 label=f"across it, $B_y = {P.By_v:.1f}$ m")
    axB.semilogy(g, P.A_near * np.exp(-g / (P.q_near * P.Bx_v)), color=CAT["orange"],
                 lw=1.5, label=f"near-field body, $A_{{near}} = {P.A_near:.0f}$")
    axB.axhline(P.A_c, color=C_CORR, lw=1.2, ls=":",
                label=f"corridor force $A_c = {P.A_c:.2f}$")
    axB.set_ylim(1e-2, 30)
    axB.set_xlim(0, 22)
    _clean(axB, "bumper-to-bumper gap (m)", "specific force (m s$^{-2}$)")
    axB.legend(fontsize=6.4, frameon=False, loc="upper right", handlelength=1.6)
    _title(axB, "(b)  three length scales")

    gap = np.linspace(3.0, 60, 500)
    v, dv, T = 30.0, 6.0, 1.5
    s_star = P.s0 + max(v * T + v * dv / (2 * np.sqrt(P.a_max * P.b_comf)), 0.0)
    a_idm = -P.a_max * (s_star / gap) ** 2
    soc = -mag(gap + CAR_L, 0.0)
    axC.plot(gap, soc, color=C_CC, lw=1.6, ls="--", label=r"$\Sigma_x$  (social)")
    axC.plot(gap, a_idm, color=C_IDM, lw=2.6, label=r"$a^{IDM}$  (safety)")
    axC.plot(gap, np.minimum(soc, a_idm), color=INK, lw=1.2, ls=(0, (1.6, 1.6)),
             label=r"$\min(\Sigma_x, a^{IDM})$")
    axC.axhline(0, color=PAL["axis"], lw=0.8)
    axC.set_ylim(-9, 0.8)
    axC.set_xlim(3, 60)
    _clean(axC, "bumper-to-bumper gap (m)",
           "longitudinal acceleration (m s$^{-2}$)")
    axC.legend(fontsize=6.6, frameon=False, loc="lower left")
    _txt(axC, 22, -1.3, "$v = 30$ m/s, closing at 6 m/s\n"
                        f"$T = {T}$ s, $s_0 = {P.s0:.1f}$ m", INK2, fs=6.9)
    _title(axC, "(c)  the safety bound")

    _caption(fig, [
        "(a) Equipotentials are elongated along the road ($B_x \\gg B_y$) and "
        "weighted by direction: a vehicle behind counts only $\\lambda_v$ as much "
        "as the same gap ahead.",
        "(b) The near-field body term outmuscles the corridor force at contact, so "
        "vehicles pack laterally without overlapping.  (c) Below ~30 m the safety "
        "bound binds; the social term is inactive there.",
    ])
    return _save(fig, "sfm_03_car_car.svg", png)


# =====================================================================
# 4. the emergency-vehicle point field
# =====================================================================
def fig_ev_field(png):
    fig = plt.figure(figsize=(9.4, 4.1), facecolor=PAL["page"])
    _head(fig, "The emergency-vehicle field: strongly forward-focused, "
               "range and amplitude fitted")
    axA = fig.add_axes([0.052, 0.30, 0.345, 0.52])
    cax = fig.add_axes([0.408, 0.375, 0.010, 0.30])
    axB = fig.add_axes([0.505, 0.315, 0.175, 0.47], projection="polar")
    axC = fig.add_axes([0.795, 0.30, 0.185, 0.52])

    def ev_mag(dx, dy, A=P.A_ev, B=P.B_ev):
        d = np.maximum(np.hypot(dx, dy), 0.5)
        wgt = P.lam_ev + (1 - P.lam_ev) * 0.5 * (1 + dx / d)
        return np.minimum(A * np.exp((P.r_ev - d) / B), P.F_cap_ev) * wgt

    X, Y = np.meshgrid(np.linspace(-40, 80, 480), np.linspace(-20, 20, 320))
    Z = ev_mag(X, Y)
    pcm = axA.imshow(Z, extent=(-40, 80, -20, 20), origin="lower", aspect="auto",
                     cmap=viz.CMAP_SEQ, interpolation="bilinear",
                     norm=LogNorm(vmin=0.05, vmax=P.F_cap_ev), zorder=0)
    cs = axA.contour(X, Y, Z, levels=[0.2, 0.6, 1.5, 3.0], colors=[INK2],
                     linewidths=0.7, zorder=2)
    axA.clabel(cs, fmt="%.2g", fontsize=6.6, inline_spacing=2)
    for xa, ya in ((26, 9), (26, -9), (52, 5), (52, -5)):
        d = np.hypot(xa, ya)
        _arrow(axA, (xa, ya), (xa + 9 * xa / d, ya + 9 * ya / d), PAL["page"],
               lw=1.4, ms=8)
    _car(axA, 0, 0, C_EV, L=5.4, W=2.0, z=6)
    _txt(axA, 0, 5.2, "EV, heading $+x$", C_EV, ha="center", fs=7.6)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("$|F^{EV}|$ (m s$^{-2}$)", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)
    _clean(axA, "distance along the road from the EV (m)", "lateral offset (m)")
    axA.grid(False)
    axA.set_aspect("equal")
    _title(axA, "(a)  the field around the EV")

    phi = np.linspace(0, 2 * np.pi, 400)
    axB.plot(phi, 1.0 + 0 * phi, color=MUTED, lw=1.0)
    axB.plot(phi, P.lam_ev + (1 - P.lam_ev) * 0.5 * (1 + np.cos(phi)), color=C_EV,
             lw=2.0)
    axB.set_theta_zero_location("E")
    axB.set_rticks([0.5, 1.0])
    axB.set_yticklabels([])
    axB.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axB.set_xticklabels(["ahead", "left", "behind", "right"], fontsize=7.0)
    axB.tick_params(pad=-2.0)
    axB.grid(color=PAL["grid"], lw=0.6)
    axB.set_facecolor(PAL["surface"])
    axB.spines["polar"].set_color(PAL["axis"])
    axB.set_ylim(0, 1.12)
    fig.text(0.505, 0.845, "(b)  angular weight", fontsize=8.8, fontweight="bold",
             color=INK)
    fig.text(0.505, 0.245,
             f"$\\lambda_{{ev}} + (1-\\lambda_{{ev}})\\frac{{1+\\cos\\varphi}}{{2}}$,"
             f"  $\\lambda_{{ev}} = {P.lam_ev}$\ngrey: isotropic reference",
             fontsize=7.6, color=INK2, va="top")

    d = np.linspace(0, 150, 500)
    for A, B, col, lw, lab in ((P.A_ev, P.B_ev, C_EV, 2.0, "fit"),
                               (4.5, 18.0, MUTED, 1.2, "prior")):
        axC.plot(d, np.minimum(A * np.exp((P.r_ev - d) / B), P.F_cap_ev),
                 color=col, lw=lw, label=f"{lab}: $A={A:.2f}$, $B={B:.1f}$ m")
    axC.axhline(P.F_cap_ev, color=INK2, lw=0.9, ls=":")
    _txt(axC, 6, P.F_cap_ev + 0.22, f"cap {P.F_cap_ev}", INK2, fs=7.0)
    axC.axvline(P.R_front, color=CAT["yellow"], lw=1.1, ls="--")
    _txt(axC, P.R_front - 5, 4.1, "detection\nrange $R_{front}$", CAT["yellow"],
         ha="right", fs=7.0)
    axC.set_xlim(0, 152)
    axC.set_ylim(0, 6.6)
    _clean(axC, "distance $d$ to the EV (m)",
           "$|F^{EV}|$ straight ahead (m s$^{-2}$)")
    axC.legend(fontsize=6.4, frameon=False, loc="upper right", handlelength=1.3)
    _title(axC, "(c)  radial decay")

    _caption(fig, [
        "$F^{EV} = A_{ev}\\,e^{(r-d)/B_{ev}}\\,\\hat{n}\\,"
        "[\\lambda_{ev} + (1-\\lambda_{ev})(1+\\cos\\varphi)/2]$, directed away "
        "from the EV, felt only by drivers who have noticed it and reacted.",
        "Estimation moved the amplitude up and the range down relative to the "
        "hand-set prior: stronger close in, weaker beyond ~60 m, where the "
        "corridor term takes over.",
    ])
    return _save(fig, "sfm_04_ev_field.svg", png)


# =====================================================================
# 5. the predicted-corridor term
# =====================================================================
def fig_corridor(png):
    fig = plt.figure(figsize=(9.4, 4.9), facecolor=PAL["page"])
    _head(fig, "The corridor term: repulsion from where the EV is going to be",
          "The dominant term of the EV response - removing it costs seven "
          "eighths of the modelled yielding.")
    axP = fig.add_axes([0.030, 0.585, 0.950, 0.235])
    axB = fig.add_axes([0.075, 0.215, 0.355, 0.275])
    axC = fig.add_axes([0.620, 0.215, 0.355, 0.275])

    x0, x1 = -40.0, 250.0
    _road(axP, x0, x1, y_lo=-4.0, y_hi=15.0)
    y_ev, x_ev, w_need, L = 1.5 * W_LANE, 0.0, 2.4, 200.0
    axP.add_patch(Rectangle((x_ev - P.L_back, y_ev - w_need), L + P.L_back,
                            2 * w_need, facecolor=C_CORR, alpha=0.13,
                            edgecolor="none", zorder=2))
    axP.add_patch(Rectangle((x_ev + L - P.end_ramp, y_ev - w_need), P.end_ramp,
                            2 * w_need, facecolor=C_CORR, alpha=0.17,
                            edgecolor="none", zorder=2.5))
    for sgn in (-1, 1):
        axP.plot([x_ev - P.L_back, x_ev + L], [y_ev + sgn * w_need] * 2,
                 color=C_CORR, lw=1.0, ls=(0, (4, 3)), zorder=3)
    _car(axP, x_ev, y_ev, C_EV, L=5.4, W=2.0, exag=CAR_EXAG)
    _dim(axP, (x_ev, 12.6), (x_ev + L, 12.6), C_CORR)
    _txt(axP, x_ev + L / 2, 14.4,
         r"$L = \mathrm{clip}(\max(v_{EV},\,0.6\,v^0_{EV})\cdot T_{pred},\ "
         r"120,\ 260)$ m", C_CORR, ha="center", fs=8.0)
    _dim(axP, (x_ev - P.L_back, -2.6), (x_ev, -2.6), MUTED)
    _txt(axP, x_ev - P.L_back - 4, -2.6, r"$L_{back}$", MUTED, ha="right", fs=7.6)
    _txt(axP, x_ev + L - P.end_ramp / 2, -2.6,
         f"the force fades over the last {P.end_ramp:.0f} m", C_CORR,
         ha="center", fs=7.6)
    axP.plot([x_ev + L - P.end_ramp] * 2, [-1.6, y_ev - w_need], color=C_CORR,
             lw=0.7, ls=":", zorder=3)
    _car(axP, 62.0, y_ev + 1.0, C_CAR, exag=CAR_EXAG)
    _arrow(axP, (62.0, y_ev - 0.3), (62.0, y_ev - 4.6), C_CORR)
    _txt(axP, 66.0, y_ev - 3.6, r"pushed to the side $\sigma$, chosen once",
         C_CORR, fs=7.6)
    _car(axP, 128.0, 2.5 * W_LANE, C_CAR, exag=CAR_EXAG, alpha=0.5)
    _txt(axP, 128.0, 2.5 * W_LANE + 2.6, "clear: the force has decayed", INK2,
         ha="center", fs=7.6)
    _title(axP, "(a)  the corridor is the region the EV is predicted to occupy, "
                "not the region it occupies")

    dl = np.linspace(0, 7.0, 600)
    over = np.maximum(dl - w_need, 0.0)
    for u, alpha in ((1.0, 1.0), (0.5, 0.62), (0.25, 0.38)):
        axB.plot(dl, P.A_c * np.exp(-over / P.B_c) * u, color=C_CORR,
                 lw=2.0 if u == 1 else 1.4, alpha=alpha, label=f"$u = {u}$")
    axB.axvline(w_need, color=INK2, lw=0.9, ls="--")
    _txt(axB, w_need - 0.12, P.A_c * 1.13, "$w_{need}$", INK2, ha="right", fs=7.4)
    axB.axhline(P.a_pin, color=C_LANE, lw=1.3, ls=":")
    _txt(axB, 0.12, P.a_pin + 0.13, f"lane barrier $a_{{pin}} = {P.a_pin:.2f}$",
         C_LANE, fs=7.4)
    d_star = w_need + P.B_c * np.log(P.A_c / P.a_pin)
    axB.plot([d_star], [P.a_pin], "o", ms=5.5, color=CAT["violet"], zorder=6)
    _txt(axB, d_star + 0.2, P.a_pin + 0.55,
         f"$d^* = {d_star:.2f}$ m: the vehicle\nstops here, force = barrier",
         CAT["violet"], fs=7.4)
    axB.set_ylim(0, P.A_c * 1.22)
    axB.set_xlim(0, 7)
    _clean(axB, "offset from the corridor axis $|d_{\\perp}|$ (m)",
           "$F^{corr}_{\\perp}$ (m s$^{-2}$)")
    axB.legend(fontsize=7.0, frameon=False, loc="upper right", ncol=3,
               columnspacing=0.9, handlelength=1.3)
    _title(axB, f"(b)  lateral push: flat inside $w_{{need}}$, decay length "
                f"$B_c = {P.B_c:.2f}$ m")

    v = np.linspace(0, 40, 400)
    v0 = 33.0
    for blocked, alpha, lab in ((1.0, 1.0, "fully inside the corridor"),
                                (0.5, 0.6, "half overlapping it"),
                                (0.0, 0.3, "clear of it")):
        axC.plot(v, -P.gamma_c * blocked * np.maximum(v - P.kappa_merge * v0, 0),
                 color=C_CORR, lw=1.8, alpha=alpha, label=lab)
    axC.axvline(P.kappa_merge * v0, color=INK2, lw=0.9, ls="--")
    _txt(axC, P.kappa_merge * v0 - 1.2, -0.9,
         f"target speed\n$\\kappa_{{merge}}v^0 = {P.kappa_merge * v0:.1f}$ m/s",
         INK2, ha="right", fs=7.4)
    axC.set_xlim(0, 40)
    _clean(axC, "own speed $v$ (m s$^{-1}$)", "$F^{corr}_{\\parallel}$ (m s$^{-2}$)")
    axC.legend(fontsize=7.0, frameon=False, loc="lower left")
    _title(axC, r"(c)  merge brake: pull over $and$ slow down")

    _caption(fig, [
        "The corridor is rebuilt every step around the EV's nose, so even a "
        "blocked EV projects one: drivers respond to where it is trying to go.",
        "Both terms are scaled by the receiving driver's urgency $u$, so the same "
        "geometry moves a driver about to be overtaken and barely touches one "
        "200 m ahead.",
    ])
    return _save(fig, "sfm_05_corridor.svg", png)


# =====================================================================
# 6. lane keeping as a periodic potential; depinning
# =====================================================================
def fig_depinning(png):
    fig = plt.figure(figsize=(9.4, 4.1), facecolor=PAL["page"])
    _head(fig, "Lane changing as depinning: escape from a periodic potential")
    axA = fig.add_axes([0.055, 0.30, 0.355, 0.52])
    axB = fig.add_axes([0.535, 0.30, 0.190, 0.52])
    axC = fig.add_axes([0.805, 0.30, 0.185, 0.52])

    a_pin, w = P.a_pin, W_LANE
    amp = a_pin * w / (2 * np.pi)
    y = np.linspace(-0.55, 2.05, 800) * w
    U = -amp * np.cos(2 * np.pi * (y - w / 2) / w)
    axA.plot(y / w, U, color=INK, lw=2.0, label="no EV:  $U(y)$", zorder=4)
    axA.plot(y / w, U - 0.35 * a_pin * (y - w / 2), color=C_CORR, lw=1.6,
             ls=(0, (5, 2.5)), zorder=4,
             label=r"$A_cu < a^{eff}_{pin}$:  still trapped")
    axA.plot(y / w, U - 1.05 * a_pin * (y - w / 2), color=C_CORR, lw=2.0, zorder=4,
             label=r"$A_cu > a^{eff}_{pin}$:  escapes")
    axA.plot([0.5], [-amp], "o", ms=6, color=C_CAR, zorder=6)
    axA.plot([0.5, 1.06], [-amp, -amp], color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)))
    _dim(axA, (1.02, -amp), (1.02, amp), MUTED)
    _txt(axA, 0.98, 0.05, r"barrier $\propto a^{eff}_{pin}$", INK2, ha="right",
         fs=7.4)
    axA.set_ylim(-2.4, 1.5)
    axA.set_xlim(-0.55, 2.05)
    axA.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axA.set_xticklabels(["marking", "lane\ncentre", "marking", "next\ncentre",
                         "marking"], fontsize=7.2)
    axA.set_yticks([])
    _clean(axA, "", "lane potential $U(y)$")
    axA.grid(False)
    for xg in (0.0, 1.0, 2.0):
        axA.axvline(xg, color=PAL["grid"], lw=0.8)
    axA.legend(fontsize=7.0, frameon=False, loc="lower left")
    _title(axA, r"(a)  $U(y) = -a_{pin}\frac{w}{2\pi}\cos(2\pi\frac{y-w/2}{w})$,"
                r" tilted by the corridor force")

    Ac = np.linspace(0, 6, 400)
    for u, ls in ((1.0, "-"), (0.6, "--"), (0.3, ":")):
        bound = Ac * u / (1 - P.urgency_pin_relief * u)
        axB.plot(Ac, bound, color=CAT["violet"], lw=1.7, ls=ls, label=f"$u = {u}$")
    axB.fill_between(Ac, 0, np.clip(Ac / (1 - P.urgency_pin_relief), 0, 2.6),
                     color=C_CORR, alpha=0.10, lw=0)
    _txt(axB, 5.85, 0.40, "corridor forms\n(vehicle depins)", C_CORR, ha="right",
         fs=7.2)
    _txt(axB, 0.25, 2.35, "vehicle holds\nits lane", INK2, fs=7.2, va="top")
    axB.plot([P.A_c], [P.a_pin], "*", ms=12, color=C_EV, zorder=6)
    _txt(axB, P.A_c + 0.25, P.a_pin - 0.16, "calibrated\npoint", C_EV, fs=7.2)
    axB.set_xlim(0, 6)
    axB.set_ylim(0, 2.6)
    _clean(axB, "corridor amplitude $A_c$ (m s$^{-2}$)",
           "lane pinning $a_{pin}$ (m s$^{-2}$)")
    axB.legend(fontsize=6.8, loc="upper right", ncol=3, columnspacing=0.8,
               handlelength=1.4, frameon=True, facecolor=PAL["page"],
               edgecolor="none", framealpha=0.92)
    _title(axB, r"(b)  escape iff $A_cu > a_{pin}(1-\varepsilon u)$")

    ac = np.linspace(0.6, 6.0, 500)
    for apin_v, col, lw in ((0.6, MUTED, 1.2), (P.a_pin, C_CORR, 2.0),
                            (2.0, INK2, 1.2)):
        dd = np.where(ac > apin_v, 2.4 + P.B_c * np.log(np.maximum(ac / apin_v, 1e-9)),
                      np.nan)
        axC.plot(ac, dd, color=col, lw=lw,
                 label=f"$a^{{eff}}_{{pin}} = {apin_v:.2f}$")
    axC.axhline(2.4, color=INK2, lw=0.9, ls="--")
    _txt(axC, 0.75, 2.62, "$w_{need}$", INK2, fs=7.4)
    axC.plot([P.A_c], [2.4 + P.B_c * np.log(P.A_c / P.a_pin)], "*", ms=12,
             color=C_EV, zorder=6)
    axC.set_xlim(0.6, 6)
    axC.set_ylim(2.0, 9.5)
    _clean(axC, "corridor amplitude $A_c$ (m s$^{-2}$)",
           "cleared half-width $d^{*}$ (m)")
    axC.legend(fontsize=6.6, loc="lower right", frameon=True,
               facecolor=PAL["page"], edgecolor="none", framealpha=0.92)
    _title(axC, r"(c)  $d^{*} = w_{need} + B_c\ln(A_c/a^{eff}_{pin})$")

    _caption(fig, [
        "Lane keeping is a potential with minima at the lane centres, so a lane "
        "change is an activated escape rather than a rule. The corridor force "
        "tilts that potential; once the tilt",
        "exceeds the barrier the minimum disappears and the vehicle slides across. "
        "The escape condition (b) and the cleared width (c) are predictions of the "
        "formulation, not fitted quantities.",
    ])
    return _save(fig, "sfm_06_depinning.svg", png)


# =====================================================================
# 7. perception state machine + urgency
# =====================================================================
def fig_perception(png):
    fig = plt.figure(figsize=(9.4, 4.3), facecolor=PAL["page"])
    _head(fig, "Perception and urgency: what switches the EV terms on, and how "
               "hard")
    ax = _blank(fig, [0.018, 0.24, 0.455, 0.60])
    axB = fig.add_axes([0.585, 0.30, 0.175, 0.50])
    axC = fig.add_axes([0.815, 0.30, 0.175, 0.50])

    rows = (("UNAWARE", 0.775), ("NOTICED", 0.535), ("YIELDING", 0.295),
            ("HOLD", 0.055))
    for name, y in rows:
        _box(ax, 0.115, y, 0.285, 0.135, name, _light(STATE_C[name], 0.22),
             STATE_C[name], fs=8.6, weight="bold")
    trans = [
        (0.775, 0.535, "the EV is within $R_{front} = 140$ m ahead or\n"
                       "$R_{rear} = 60$ m behind (5 % of drivers never react)"),
        (0.535, 0.295, "the reaction delay has elapsed (lognormal,\n"
                       "median 1 s); the escape side is chosen once"),
        (0.295, 0.055, "the EV has passed and is receding"),
    ]
    for y_from, y_to, text in trans:
        _arrow(ax, (0.258, y_from - 0.005), (0.258, y_to + 0.140), INK2, lw=1.2)
        _txt(ax, 0.425, (y_from + y_to + 0.135) / 2, text, INK2, fs=7.3,
             halo=False)
    # merge-back runs down the left margin, so it crosses no label
    ax.add_patch(FancyArrowPatch((0.110, 0.122), (0.110, 0.842),
                                 connectionstyle="bar,fraction=-0.10", color=INK2,
                                 lw=1.1, arrowstyle="-|>", mutation_scale=8,
                                 zorder=3))
    _txt(ax, 0.005, 0.48, "hold timer expires:\nmerge back", INK2, fs=7.3,
         rot=90, ha="center", halo=False)
    fig.text(0.018, 0.845, "(a)  awareness state machine", fontsize=8.8,
             fontweight="bold", color=INK, va="bottom")
    fig.text(0.018, 0.215,
             "$F^{EV}$ acts only in YIELDING and HOLD; $F^{corr}$ is scaled by "
             "the urgency $u$, which is zero outside YIELDING.", fontsize=7.7,
             color=INK2, va="top")

    t_arr = np.linspace(0, 45, 500)
    for jam, col, lw in ((0.0, C_CAR, 1.9), (0.5, CAT["orange"], 1.5),
                         (1.0, C_EV, 1.5)):
        T_eff = P.T_react * (1 + P.jam_anticipation * jam)
        axB.plot(t_arr, 1 / (1 + np.exp((t_arr - T_eff) / P.sigma_t)), color=col,
                 lw=lw)
        _txt(axB, T_eff, 1.10, f"jam {jam:.1f}", col, ha="center", fs=7.0)
    axB.set_xlim(0, 45)
    axB.set_ylim(-0.03, 1.22)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _clean(axB, "estimated EV time to arrival (s)", "urgency $u$")
    _txt(axB, 1.5, 0.30,
         f"$T_{{eff}} = T_{{react}}(1 + {P.jam_anticipation:g}\\,$jam$)$", INK2,
         fs=7.2)
    _title(axB, "(b)  drivers act on time")

    s = np.linspace(0, 240, 600)
    u_prox = 1.0 / (1 + np.exp((s - P.R_urgent) / P.sigma_s))
    v_own, v_ev = 2.5, 4.0
    T_eff = P.T_react * (1 + P.jam_anticipation * 1.0)
    u_time = 1 / (1 + np.exp((s / max(v_ev - v_own, 0.5) - T_eff) / P.sigma_t))
    axC.plot(s, u_time, color=C_EV, lw=1.5, ls="--", label="$u_{time}$")
    axC.plot(s, u_prox, color=CAT["violet"], lw=1.5, ls=(0, (1.5, 1.5)),
             label="$u_{prox}$")
    axC.plot(s, np.maximum(u_time, u_prox), color=INK, lw=2.0, alpha=0.85,
             label=r"$u = \max$")
    axC.axvline(P.R_urgent, color=INK2, lw=0.9, ls=":")
    _txt(axC, P.R_urgent - 6, 0.16, "$R_{urgent}$", INK2, ha="right", fs=7.4)
    axC.set_xlim(0, 240)
    axC.set_ylim(-0.03, 1.22)
    axC.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _clean(axC, "distance ahead of the EV $s$ (m)", "urgency $u$")
    axC.legend(fontsize=6.8, frameon=False, loc="upper right")
    _txt(axC, 138, 0.52, "queue 2.5 m/s,\nEV 4 m/s", INK2, fs=6.9)
    _title(axC, "(c)  a blocked EV still acts")

    _caption(fig, [
        "(b) Urgency is set by the estimated time until the EV arrives, not by "
        "distance, so drivers act about $T_{react}$ seconds before it reaches "
        "them - and earlier when hemmed in.",
        "(c) In a queue that channel alone deadlocks (a slow EV calms the traffic, "
        "which keeps the EV slow), so a jam-gated second channel keyed on distance "
        "holds urgency up.",
    ])
    return _save(fig, "sfm_07_perception.svg", png)


# =====================================================================
# 8. two-stage identification and the adequacy bar
# =====================================================================
def fig_identification(png):
    fig = plt.figure(figsize=(9.4, 5.2), facecolor=PAL["page"])
    _head(fig, "Identification: two nested stages, judged against a bar the data "
               "set for themselves")
    ax = _blank(fig, [0.018, 0.530, 0.964, 0.340])
    _box(ax, 0.0, 0.42, 0.205, 0.50,
         "highD\n60 drone recordings, 110 516 vehicles\n"
         "no emergency vehicle present", _light(C_CAR), C_CAR, fs=7.7)
    _box(ax, 0.250, 0.42, 0.250, 0.50,
         "stage 1: host block\n"
         r"$\tau,\,T_{hw},\,s_0,\,A_v,\,B_x,\,a_{max},\,b$" + "\n"
         r"$a_{pin},\,\zeta_{lat},\,v^{max}_{lat},\,a^{max}_{lat},\,\sigma_{off},$"
         + "\n" + r"$h_{lat},\,k_\rho,\,A_{pass},\,T_{frust},\,s_{veto}$",
         _light(C_LANE), C_LANE, fs=7.7)
    _box(ax, 0.545, 0.42, 0.205, 0.50,
         "dashcam ground truth\n930 s, 1033 tracks\nrecorded EV passages",
         _light(C_EV), C_EV, fs=7.7)
    _box(ax, 0.795, 0.42, 0.205, 0.50,
         "stage 2: EV block\n" + r"$A_{ev},\,B_{ev},\,A_c,\,B_c$" + "\n"
         "host block held fixed", _light(C_CORR), C_CORR, fs=7.7)
    for xa, xb in ((0.207, 0.248), (0.502, 0.543), (0.752, 0.793)):
        _arrow(ax, (xa, 0.67), (xb, 0.67), INK2, lw=1.2)
    ax.plot([0.375, 0.375, 0.897], [0.42, 0.30, 0.30], color=C_LANE, lw=1.2,
            zorder=3, solid_joinstyle="round")
    _arrow(ax, (0.897, 0.32), (0.897, 0.415), C_LANE, lw=1.2, ms=8)
    _txt(ax, 0.636, 0.30, "stage-1 estimates frozen", C_LANE, ha="center", fs=7.6)
    _txt(ax, 0.0, 0.17,
         "Why nested: estimated in one stage with an EV in the scenario, the host "
         "parameters produce zero discretionary lane changes over 206 vehicle-"
         "kilometres,\nagainst 0.28 per vehicle-kilometre measured - the lateral "
         "block had been identified from manoeuvres the EV itself induced.",
         INK2, fs=7.6, va="top", halo=False)

    axB = fig.add_axes([0.075, 0.225, 0.360, 0.245])
    names = ["fitted sites,\nfresh replications", "sites not used\nin estimation",
             "congestion,\nregime withheld", "EV block,\nout of sample"]
    vals, bars, prev = [1.50, 1.53, 2.48, 1.07], [0.49, 0.49, 0.49, 0.73], 3.51
    xp = np.arange(4)
    axB.bar(xp, vals, 0.55, color=[C_CAR, C_CAR, C_CAR, C_CORR], zorder=3)
    axB.bar(xp[2], prev, 0.55, facecolor="none", edgecolor=MUTED, lw=1.0,
            ls="--", zorder=2)
    _txt(axB, 2.0, prev + 0.10, f"uncalibrated {prev}", MUTED, ha="center", fs=6.9)
    for i, (v, b) in enumerate(zip(vals, bars)):
        axB.plot([i - 0.36, i + 0.36], [b, b], color=C_EV, lw=2.2, zorder=5)
        _txt(axB, i, v + 0.10, f"{v:.2f}", INK, ha="center", fs=7.4)
    _txt(axB, -0.45, 3.72, "red line = adequacy bar", C_EV, fs=7.0)
    axB.set_xticks(xp)
    axB.set_xticklabels(names, fontsize=7.0)
    axB.set_ylim(0, 4.1)
    _clean(axB, "", "distance to the data")
    _title(axB, "(b)  results: lower is closer to the measurements")

    axC = _blank(fig, [0.560, 0.225, 0.420, 0.245])
    for k in range(7):
        x = 0.02 + k * 0.072
        held = (k == 3)
        axC.add_patch(Rectangle((x, 0.79), 0.052, 0.15,
                                facecolor=C_EV if held else _light(C_CAR, 0.5),
                                edgecolor=C_EV if held else C_CAR, lw=1.0))
    _txt(axC, 0.55, 0.865, "88 free-flow carriageways", INK2, fs=7.5, halo=False)
    _arrow(axC, (0.262, 0.77), (0.262, 0.66), C_EV, lw=1.2)
    _txt(axC, 0.29, 0.70, "one held out", C_EV, fs=7.3, halo=False)
    _box(axC, 0.50, 0.60, 0.49, 0.14, "the other 87 pooled into a reference",
         _light(C_CAR), C_CAR, fs=7.4)
    _arrow(axC, (0.49, 0.655), (0.34, 0.605), C_CAR, lw=1.2)
    _txt(axC, 0.02, 0.47,
         "The held-out carriageway is scored against that pool by the same\n"
         "distance the objective minimises; the mean over all of them is the bar.",
         INK2, fs=7.4, va="top", halo=False)
    _txt(axC, 0.02, 0.20,
         "The bar is a statistic of the measurements alone, hence invariant\n"
         "under the estimation: an exogenous criterion of adequacy.", C_EV,
         fs=7.4, va="top", halo=False)
    fig.text(0.560, 0.483, "(c)  where the bar comes from", fontsize=8.8,
             fontweight="bold", color=INK, va="bottom")

    _caption(fig, [
        "Scoring below the bar would mean reproducing the pooled average of the "
        "measurements more closely than any single real carriageway does: the bar "
        "is a floor, not a target.",
        "The estimate transfers: the distance at sites never used in estimation "
        "equals the distance at the fitted sites, and the withheld congested "
        "regime still beats the uncalibrated model.",
    ])
    return _save(fig, "sfm_08_identification.svg", png)


# =====================================================================
# 0. nomenclature: every symbol the other figures and the equations use
# =====================================================================
def _sym_groups():
    """(group title, colour, [(symbol, meaning, value and where it came from)]).

    Values are read from the calibrated parameter object, so this table cannot
    drift away from the model the other figures draw.
    """
    return [
        ("State of one vehicle", C_DRIVE, [
            (r"$x,\ y$", "position along and across the road", "m"),
            (r"$v_x,\ v_y$", "speed along and across the road", "m s$^{-1}$"),
            (r"$a_x,\ a_y$", "the acceleration the model computes", "m s$^{-2}$"),
            (r"$v^0$", "desired speed of this driver", "per lane, measured"),
            (r"$\tau$", "how fast the driver returns to $v^0$",
             f"{P.tau:.2f} s, fitted"),
            (r"$\Delta t$", "integration step (semi-implicit Euler)",
             f"{P.dt:.2f} s, set"),
            (r"$u$", "urgency: how pressing the EV is", "0 to 1, computed"),
            (r"$g$", "gate: has this driver reacted yet", "0 or 1, computed"),
        ]),
        ("Road and corridor geometry", C_CORR, [
            (r"$w$", "lane width", f"{W_LANE:.1f} m (highD median 3.9)"),
            (r"$s$", "distance ahead of the EV's front bumper", "m"),
            (r"$d_{\perp}$", "signed offset from the corridor axis", "m"),
            (r"$w_{need}$", "clearance the EV needs, each side",
             f"{CAR_W + P.margin_c:.1f} m, from widths"),
            (r"$L$", "how far ahead the corridor reaches",
             f"{P.L_pred_min:.0f}-{P.L_pred_max:.0f} m, from $v_{{EV}}$"),
            (r"$L_{back}$", "corridor extent behind the EV nose",
             f"{P.L_back:.0f} m, set"),
            (r"$T_{pred}$", "how far ahead the EV path is predicted",
             f"{P.T_pred:.0f} s, set"),
            (r"$d^{*}$", "half-width the traffic clears", "m, predicted"),
            (r"$\rho,\ \rho_{ref}$", "local density, reference density",
             f"veh/km/lane, ref {P.rho_ref:.1f}"),
        ]),
        ("Vehicle-vehicle interaction and the safety layer", C_CC, [
            (r"$g_x,\ g_y$", "bumper-to-bumper gaps to a neighbour", "m"),
            (r"$q$", r"elliptic distance, $\sqrt{(g_x/B_x)^2+(g_y/B_y)^2}$",
             ""),
            (r"$A_v$", "strength of vehicle-vehicle repulsion",
             f"{P.A_v:.2f} m s$^{{-2}}$, fitted"),
            (r"$B_x,\ B_y$", "its decay length, along / across",
             f"{P.Bx_v:.2f} / {P.By_v:.2f} m"),
            (r"$A_{near},\ q_{near}$", "short-range body term",
             f"{P.A_near:.0f} m s$^{{-2}}$, {P.q_near:.2f}, set"),
            (r"$\lambda_v$", "weight of a neighbour behind",
             f"{P.lam_v:.2f} (1 = as ahead)"),
            (r"$a^{IDM}$", "deceleration the safety layer asks",
             "m s$^{-2}$, computed"),
            (r"$T_{hw},\ s_0$", "time headway, standstill gap",
             f"{P.T_hw_lo:.2f}-{P.T_hw_hi:.2f} s, {P.s0:.2f} m"),
            (r"$a_{max},\ b$", "comfortable accel. / decel.",
             f"{P.a_max:.2f} / {P.b_comf:.2f} m s$^{{-2}}$"),
        ]),
        ("Emergency-vehicle terms (stage 2)", C_EV, [
            (r"$A_{ev},\ B_{ev}$", "strength, range of the EV field",
             f"{P.A_ev:.2f} m s$^{{-2}}$, {P.B_ev:.1f} m"),
            (r"$\lambda_{ev}$", "forward focus of that field",
             f"{P.lam_ev:.2f} (small = focused)"),
            (r"$r_{ev},\ d$", "EV body radius, distance to the EV",
             f"{P.r_ev:.0f} m, m"),
            (r"$\varphi$", "angle off the EV heading", "rad"),
            (r"$A_c,\ B_c$", "strength, decay of the corridor push",
             f"{P.A_c:.2f} m s$^{{-2}}$, {P.B_c:.2f} m"),
            (r"$\sigma$", "escape side, chosen once", "+1 left / -1 right"),
            (r"$\gamma_c,\ \kappa_{merge}$", "merge-brake gain, target speed",
             f"{P.gamma_c:.2f} s$^{{-1}}$, {P.kappa_merge:.2f}$v^0$"),
            (r"$f(s)$", "fade at the far end of the corridor",
             f"over {P.end_ramp:.0f} m, set"),
        ]),
        ("Lane keeping and lane changing", C_LANE, [
            (r"$U(y)$", "lane potential, minima at centres", "m$^2$ s$^{-2}$"),
            (r"$a_{pin}$", "how firmly the lane is held",
             f"{P.a_pin:.2f} m s$^{{-2}}$, fitted"),
            (r"$a^{eff}_{pin}$",
             r"the barrier, $a_{pin}(\rho/\rho_{ref})^{k_\rho}(1-\varepsilon u)$",
             ""),
            (r"$k_\rho$", "density stiffening of that barrier",
             f"{P.k_rho:.2f}, fitted"),
            (r"$\varepsilon$", "how much urgency relaxes it",
             f"{P.urgency_pin_relief:.2f}, set"),
            (r"$\zeta_{lat}$", "damping of the sideways motion",
             f"{P.zeta_lat:.2f}, fitted"),
            (r"$v^{max}_{lat},\ a^{max}_{lat}$", "sideways comfort limits",
             f"{P.v_lat_max:.2f} m s$^{{-1}}$, {P.a_lat_max:.2f} m s$^{{-2}}$"),
            (r"$\sigma_{off},\ h_{lat}$", "driver-to-driver spread",
             f"{P.sigma_off:.2f} m, {P.het_lat:.2f}, fitted"),
            (r"$A_{pass}$", "strength of the overtaking incentive",
             f"{P.A_pass:.2f}, fitted"),
        ]),
        ("Perception and urgency", CAT["yellow"], [
            (r"$R_{front},\ R_{rear}$", "distance at which the EV is seen",
             f"{P.R_front:.0f} / {P.R_rear:.0f} m, set"),
            (r"$T_{react}$", "how early a driver starts to act",
             f"{P.T_react:.0f} s, set"),
            (r"$\sigma_t$", "spread of that timing across drivers",
             f"{P.sigma_t:.1f} s, set"),
            (r"$T_{eff}$", "that horizon, stretched when hemmed in",
             f"$T_{{react}}(1+{P.jam_anticipation:g}\\,$jam$)$"),
            (r"$R_{urgent},\ \sigma_s$", "range of the distance channel",
             f"{P.R_urgent:.0f} m, {P.sigma_s:.0f} m, set"),
            (r"$t_{arr}$", "estimated time until the EV arrives", "s, computed"),
            (r"$p_{noncomply}$", "drivers who never react", f"{100 * P.p_noncomply:.0f} %, set"),
        ]),
    ]


def fig_symbols(png):
    fig = plt.figure(figsize=(9.4, 6.9), facecolor=PAL["page"])
    _head(fig, "What every symbol means",
          "Everything the equations and the other seven figures use, with the "
          "value it takes in the calibrated model and where that value came "
          "from.", y=0.978)
    ax = _blank(fig, [0.018, 0.125, 0.964, 0.790])

    groups = _sym_groups()
    col_x = (0.0, 0.515)                 # left / right column origins
    col_w = 0.465
    row_h = 0.029
    y = [0.955, 0.955]
    col = 0
    for gi, (title, colour, rows) in enumerate(groups):
        if gi == 3:                      # second column starts at group 4
            col = 1
        x0 = col_x[col]
        y[col] -= 0.014
        ax.add_patch(Rectangle((x0, y[col] - 0.006), col_w, 0.026,
                               facecolor=_light(colour, 0.22), edgecolor="none"))
        ax.text(x0 + 0.008, y[col] + 0.007, title, fontsize=8.4, color=INK,
                fontweight="bold", va="center")
        y[col] -= 0.036
        for sym, meaning, value in rows:
            # the meaning column runs from x0+0.098 to the value column; at
            # 7.5 pt on a 4.4 in column that is ~40 characters, and an
            # over-long meaning silently overprints the value
            n_m = len(meaning) - 3 * meaning.count("$")
            n_v = len(value) - 3 * value.count("$")
            if n_m > 40 or n_v > 26:
                print(f"  ! symbol row too wide ({n_m}/{n_v}): {meaning[:40]}")
            ax.text(x0 + 0.008, y[col], sym, fontsize=8.0, color=colour,
                    va="center")
            ax.text(x0 + 0.098, y[col], meaning, fontsize=7.5, color=INK,
                    va="center")
            ax.text(x0 + col_w - 0.004, y[col], value, fontsize=7.2,
                    color=INK2, va="center", ha="right")
            y[col] -= row_h
        y[col] -= 0.014

    _caption(fig, [
        "'set' = fixed from the literature or from geometry; 'fitted' = "
        "estimated from highD motorway trajectories with no emergency vehicle "
        "present (stage 1); 'stage 2' = estimated from recorded EV passages "
        "with the stage-1 values held fixed; 'computed' = a quantity the "
        "simulation produces at run time rather than a parameter.",
    ])
    return _save(fig, "sfm_00_symbols.svg", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true",
                    help="also write a raster preview next to each SVG")
    a = ap.parse_args()
    outs = [fig_symbols(a.png), fig_overview(a.png), fig_composition(a.png),
            fig_car_car(a.png),
            fig_ev_field(a.png), fig_corridor(a.png), fig_depinning(a.png),
            fig_perception(a.png), fig_identification(a.png)]
    log_run("paper_figs", metrics=dict(n_figures=len(outs)),
            outputs=[os.path.relpath(o, ROOT) for o in outs],
            note="explanatory SVG graphics of the social force model -> images/")
    print(f"\n{len(outs)} SVG figures in images/")


if __name__ == "__main__":
    main()
