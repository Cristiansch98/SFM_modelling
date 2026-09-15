"""Visualisation: animations, force-field maps, space-time and phase diagrams.

Style follows the dataviz reference palette: categorical slots in fixed order,
one-hue sequential ramps for magnitude, blue<->red diverging for signed
displacement. Analysis figures render on the light surface; the animations
deliberately commit to a dark cinematic look.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle

from .state import UNAWARE, NOTICED, YIELDING, HOLD

# ---------------------------------------------------------------- palette
LIGHT = dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
             muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
DARK = dict(surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7",
            muted="#898781", grid="#2c2c2a", axis="#383835")

CAT_LIGHT = dict(blue="#2a78d6", aqua="#1baf7a", yellow="#eda100", red="#e34948",
                 orange="#eb6834", violet="#4a3aa7")
CAT_DARK = dict(blue="#3987e5", aqua="#199e70", yellow="#c98500", red="#e66767",
                orange="#d95926", violet="#9085e9")

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2b1b0", "#e34948", "#8f1d1d"])

STATE_FILL_DARK = {UNAWARE: "#5d5c58", NOTICED: "#c98500",
                   YIELDING: "#d95926", HOLD: "#199e70"}
EV_DARK = "#e66767"


def _style_ax(ax, pal=LIGHT):
    ax.set_facecolor(pal["surface"])
    for s in ax.spines.values():
        s.set_color(pal["axis"]); s.set_linewidth(0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=pal["muted"], labelsize=8.5, length=3)
    ax.xaxis.label.set_color(pal["ink2"]); ax.yaxis.label.set_color(pal["ink2"])
    ax.title.set_color(pal["ink"])
    ax.grid(True, color=pal["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _fig(w, h, pal=LIGHT):
    fig = plt.figure(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor(pal["page"])
    return fig


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path, flush=True)


# ================================================================ animation
def _draw_road(ax, road, x0, x1, pal):
    ax.add_patch(Rectangle((x0, road.y_min), x1 - x0, road.y_max - road.y_min,
                           fc="#232325", ec="none", zorder=0))
    for sh_lo, sh_hi in ((road.y_min, 0.0), (road.width_lanes, road.y_max)):
        if sh_hi > sh_lo:
            ax.add_patch(Rectangle((x0, sh_lo), x1 - x0, sh_hi - sh_lo,
                                   fc="#2b2b2e", ec="none", zorder=0.5))
    for k in range(1, road.n_lanes):
        ax.plot([x0, x1], [k * road.lane_width] * 2, ls=(0, (7, 7)),
                color="#e1e0d9", lw=1.0, alpha=0.55, zorder=1)
    for yb in (0.0, road.width_lanes):
        ax.plot([x0, x1], [yb, yb], color="#e1e0d9", lw=1.4, alpha=0.8, zorder=1)


def _car_patch(x, y, L, W, fc, ec="none", lw=0.0, alpha=1.0, z=3):
    return FancyBboxPatch((x - L / 2, y - W / 2), L, W,
                          boxstyle="round,pad=0,rounding_size=0.5",
                          mutation_aspect=0.5, fc=fc, ec=ec, lw=lw,
                          alpha=alpha, zorder=z)


def animate(hist, path, t_end=None, frame_dt=0.25, fps=16, span=(90, 200),
            title=""):
    """Dark cinematic top-down animation following the EV."""
    pal = DARK
    road, ev = hist.road, hist.ev
    t_end = t_end if t_end is not None else float(hist.t[-1])
    frames = np.searchsorted(hist.t, np.arange(hist.t[0], t_end, frame_dt))
    frames = np.unique(np.clip(frames, 0, hist.n_frames - 1))

    fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
    fig.patch.set_facecolor(pal["page"])
    gs = fig.add_gridspec(2, 2, height_ratios=[2.9, 1.15], width_ratios=[1, 1],
                          hspace=0.34, wspace=0.22,
                          left=0.055, right=0.985, top=0.9, bottom=0.11)
    ax = fig.add_subplot(gs[0, :])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for a in (ax_v, ax_c):
        _style_ax(a, pal)

    # precomputed traces for the insets
    kmh = 3.6
    ev_v = hist.vx[:, ev] * kmh
    others = np.ones(hist.x.shape[1], bool); others[ev] = False
    bg_v = hist.vx[:, others].mean(axis=1) * kmh
    from .metrics import _clearance_trace
    clr = _clearance_trace(hist)

    ax_v.plot(hist.t, ev_v, color=EV_DARK, lw=2.0, label="EV")
    ax_v.plot(hist.t, bg_v, color=pal["muted"], lw=1.6, ls="--", label="traffic mean")
    ax_v.set_ylabel("speed (km/h)"); ax_v.set_xlabel("t (s)")
    ax_v.legend(loc="lower right", fontsize=8, frameon=False,
                labelcolor=pal["ink2"])
    ax_c.plot(hist.t, clr, color=CAT_DARK["blue"], lw=2.0)
    ax_c.set_ylabel("clear corridor ahead (m)"); ax_c.set_xlabel("t (s)")
    cur_v = ax_v.axvline(hist.t[0], color=pal["ink2"], lw=0.9, alpha=0.7)
    cur_c = ax_c.axvline(hist.t[0], color=pal["ink2"], lw=0.9, alpha=0.7)

    legend_handles = [
        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc=STATE_FILL_DARK[UNAWARE], mec="none", label="unaware"),
        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc=STATE_FILL_DARK[NOTICED], mec="none", label="noticed"),
        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc=STATE_FILL_DARK[YIELDING], mec="none", label="yielding"),
        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc=STATE_FILL_DARK[HOLD], mec="none", label="hold"),
        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc=EV_DARK, mec="none", label="EV"),
    ]

    p = hist.params
    w_c = 0.5 * hist.W[ev] + p.margin_c

    def draw(fi):
        k = frames[fi]
        ax.clear()
        ax.set_facecolor(pal["page"])
        x_ev = hist.x[k, ev]
        x0, x1 = x_ev - span[0], x_ev + span[1]
        _draw_road(ax, road, x0, x1, pal)

        # corridor band (predicted path)
        L_c = min(240.0, p.L_pred_max)
        ax.add_patch(Rectangle((x_ev, hist.y_corr - w_c), L_c, 2 * w_c,
                               fc=EV_DARK, alpha=0.13, ec="none", zorder=1.5))
        ax.plot([x_ev, x_ev + L_c], [hist.y_corr] * 2, color=EV_DARK, lw=1.0,
                ls=(0, (5, 6)), alpha=0.55, zorder=1.6)

        vis = (hist.x[k] > x0 - 8) & (hist.x[k] < x1 + 8)
        for i in np.flatnonzero(vis):
            if i == ev:
                continue
            fc = STATE_FILL_DARK[int(hist.aware[k, i])]
            ax.add_patch(_car_patch(hist.x[k, i], hist.y[k, i], hist.L[i],
                                    hist.W[i], fc))
        # EV with flashing beacon + glow
        blink = (fi % 2 == 0)
        glow = _car_patch(hist.x[k, ev], hist.y[k, ev], hist.L[ev] + 2.6,
                          hist.W[ev] + 2.6, EV_DARK, alpha=0.28 if blink else 0.16,
                          z=2.5)
        ax.add_patch(glow)
        ax.add_patch(_car_patch(hist.x[k, ev], hist.y[k, ev], hist.L[ev],
                                hist.W[ev], EV_DARK, z=4))
        bx = hist.x[k, ev] + (0.9 if blink else -0.9)
        ax.add_patch(Rectangle((bx - 0.55, hist.y[k, ev] - 0.38), 1.1, 0.76,
                               fc="#ffffff" if blink else "#9ec5f4", ec="none",
                               zorder=5))

        ax.set_xlim(x0, x1)
        ax.set_ylim(road.y_min - 1.6, road.y_max + 1.6)
        ax.set_aspect(2.6)
        ax.set_yticks([]); ax.tick_params(colors=pal["muted"], labelsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{title}    t = {hist.t[k]:5.1f} s    EV {ev_v[k]:4.0f} km/h    "
                     f"clear {min(clr[k], 240):3.0f} m",
                     color=pal["ink"], fontsize=11, family="monospace", loc="left")
        ax.legend(handles=legend_handles, loc="upper right", ncol=5, fontsize=8,
                  frameon=False, labelcolor=pal["ink2"], bbox_to_anchor=(1.0, 1.16))
        cur_v.set_xdata([hist.t[k]]); cur_c.set_xdata([hist.t[k]])
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path, flush=True)


# ============================================================ force field
def field_probe(xs, ys, x_ev, y_ev, y_corr, p, heading=(1.0, 0.0), W_car=1.8,
                W_ev=2.2):
    """Analytic EV + corridor field on a grid, as felt by a fully aware
    compliant driver (u = 1). Returns (fx, fy) per unit mass."""
    X, Y = np.meshgrid(xs, ys)
    ddx, ddy = X - x_ev, Y - y_ev
    d = np.maximum(np.hypot(ddx, ddy), 0.5)
    nx, ny = ddx / d, ddy / d
    cphi = nx * heading[0] + ny * heading[1]
    wgt = p.lam_ev + (1 - p.lam_ev) * 0.5 * (1 + cphi)
    mag = np.minimum(p.A_ev * np.exp((p.r_ev - d) / p.B_ev), p.F_cap_ev) * wgt
    fx, fy = mag * nx, mag * ny

    s, dlat = X - x_ev, Y - y_corr
    L = p.L_pred_max
    w_need = 0.5 * (W_ev + W_car) + p.margin_c
    over = np.maximum(np.abs(dlat) - w_need, 0.0)
    fade = np.clip((L - s) / p.end_ramp, 0.0, 1.0)
    inc = (s > -p.L_back) & (s < L)
    f_lat = np.where(inc, p.A_c * np.exp(-over / p.B_c) * fade * np.sign(dlat), 0.0)
    fy = fy + f_lat
    return fx, fy


def plot_force_field(p, path, x_ev=0.0, y_ev=5.25, y_corr=5.25, road=None):
    pal = LIGHT
    xs = np.linspace(x_ev - 70, x_ev + 150, 221)
    ys = np.linspace(road.y_min, road.y_max, 90) if road else np.linspace(-2, 11.3, 90)
    fx, fy = field_probe(xs, ys, x_ev, y_ev, y_corr, p)
    mag = np.hypot(fx, fy)

    fig = _fig(11.5, 4.4, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    pc = ax.pcolormesh(xs, ys, mag, cmap=CMAP_SEQ, shading="auto",
                       vmin=0, vmax=min(p.F_cap_ev, p.A_c) + 1.0)
    step = 8
    Xq, Yq = np.meshgrid(xs[::step], ys[::step])
    ax.quiver(Xq, Yq, fx[::step, ::step], fy[::step, ::step],
              color="#0b0b0b", alpha=0.55, width=0.0022, scale=90)
    if road:
        for k in range(1, road.n_lanes):
            ax.axhline(k * road.lane_width, color="#ffffff", lw=1.0, ls=(0, (6, 6)), alpha=0.8)
        for yb in (0.0, road.width_lanes):
            ax.axhline(yb, color="#ffffff", lw=1.4, alpha=0.9)
    ax.plot([x_ev, x_ev + p.L_pred_max], [y_corr] * 2, color=CAT_LIGHT["red"],
            lw=1.6, ls="--", label="predicted corridor")
    ax.plot([x_ev], [y_ev], marker=(3, 0, -90), ms=15, color=CAT_LIGHT["red"],
            mec="white", mew=1.0, ls="none", label="EV")
    cb = fig.colorbar(pc, ax=ax, pad=0.012)
    cb.set_label("|F| (m/s$^2$)", color=pal["ink2"], fontsize=9)
    cb.ax.tick_params(colors=pal["muted"], labelsize=8)
    cb.outline.set_visible(False)
    ax.set_xlabel("x - x$_{EV}$ (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Repulsion field of the emergency vehicle: anisotropic point source "
                 f"(A={p.A_ev:.1f}, B={p.B_ev:.0f} m, $\\lambda$={p.lam_ev:.2f}) "
                 f"+ predicted-path corridor (A$_c$={p.A_c:.1f}, B$_c$={p.B_c:.1f} m)",
                 fontsize=10.5, loc="left", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=pal["ink2"])
    ax.set_ylim(ys[0], ys[-1])
    _save(fig, path)


# ============================================================ space-time
def plot_spacetime(hist, path):
    pal = LIGHT
    ev = hist.ev
    fig = _fig(10.5, 5.2, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    v_ref = np.median(hist.v0[np.arange(hist.v0.size) != ev])
    segs, cols = [], []
    for i in range(hist.x.shape[1]):
        if i == ev:
            continue
        pts = np.column_stack([hist.t, hist.x[:, i]])
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))       # (K-1, 2, 2)
        cong = np.clip(1.0 - hist.vx[:, i] / v_ref, 0.0, 1.0)
        cols.append(cong[:-1])
    lc = LineCollection(np.concatenate(segs), cmap=CMAP_SEQ,
                        norm=plt.Normalize(0, 1), lw=0.75, alpha=0.85)
    lc.set_array(np.concatenate(cols))
    ax.add_collection(lc)
    ax.plot(hist.t, hist.x[:, ev], color=CAT_LIGHT["red"], lw=2.4, label="EV",
            zorder=5)
    ax.set_xlim(hist.t[0], hist.t[-1])
    ax.set_ylim(hist.x[:, ev].min() - 40, np.percentile(hist.x[-1], 98))
    cb = fig.colorbar(lc, ax=ax, pad=0.012)
    cb.set_label("congestion  1 - v/v$_0$", color=pal["ink2"], fontsize=9)
    cb.ax.tick_params(colors=pal["muted"], labelsize=8)
    cb.outline.set_visible(False)
    ax.set_xlabel("t (s)"); ax.set_ylabel("x (m)")
    ax.set_title("Space-time diagram: the EV (red) cuts through traffic "
                 "with minimal disruption (line shade = congestion)",
                 fontsize=10.5, loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=pal["ink2"])
    _save(fig, path)


# ============================================================ bow wave
def plot_bowwave(hist, path, t_frac=(0.25, 0.9)):
    """Mean lateral distance from the corridor line vs EV-relative position,
    aggregated over the quasi-steady window (and over several histories if a
    list is given): the yield field is a traveling wave in the EV comoving
    frame."""
    pal = LIGHT
    hists = hist if isinstance(hist, (list, tuple)) else [hist]
    hist = hists[0]
    bins = np.arange(-220, 221, 10)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    s_all, d_all = [], []
    mins = []                       # per-history bin minima ('closest vehicle')
    for h in hists:
        ev = h.ev
        k0, k1 = int(t_frac[0] * h.n_frames), int(t_frac[1] * h.n_frames)
        ss, dd = [], []
        for k in range(k0, k1):
            s = h.x[k] - h.x[k, ev]
            d = np.abs(h.y[k] - h.y_corr)
            m = (np.abs(s) < 220) & (np.arange(s.size) != ev)
            ss.append(s[m]); dd.append(d[m])
        ss, dd = np.concatenate(ss), np.concatenate(dd)
        s_all.append(ss); d_all.append(dd)
        mn_h = np.full(ctr.size, np.nan)
        for b in range(ctr.size):
            m = (ss >= bins[b]) & (ss < bins[b + 1])
            if m.sum() > 4:
                mn_h[b] = dd[m].min()
        mins.append(mn_h)
    ev = hist.ev
    k0, k1 = int(t_frac[0] * hist.n_frames), int(t_frac[1] * hist.n_frames)
    s_all = np.concatenate(s_all); d_all = np.concatenate(d_all)

    med = np.full(ctr.size, np.nan)
    q25 = np.full(ctr.size, np.nan); q75 = np.full(ctr.size, np.nan)
    for b in range(ctr.size):
        m = (s_all >= bins[b]) & (s_all < bins[b + 1])
        if m.sum() > 4:
            med[b] = np.median(d_all[m])
            q25[b], q75[b] = np.percentile(d_all[m], [25, 75])
    with np.errstate(invalid="ignore"):
        mn = np.nanmean(np.stack(mins), axis=0)   # seed-averaged closest vehicle

    p = hist.params
    w_need = 0.5 * (hist.W[ev] + 1.8) + p.margin_c
    fig = _fig(10.0, 4.3, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    ax.fill_between(ctr, q25, q75, color=CAT_LIGHT["blue"], alpha=0.18, lw=0)
    ax.plot(ctr, med, color=CAT_LIGHT["blue"], lw=2.2, label="median |d$_\\perp$|")
    ax.plot(ctr, mn, color=CAT_LIGHT["orange"], lw=1.6, ls="--",
            label="closest vehicle")
    ax.axhline(w_need, color=CAT_LIGHT["red"], lw=1.2, ls=":",
               label=f"required half-width ({w_need:.1f} m)")
    ax.axvline(0, color=pal["ink2"], lw=1.0)
    ax.annotate("EV", (0, ax.get_ylim()[1] * 0.02), xytext=(4, 4),
                textcoords="offset points", color=pal["ink2"], fontsize=9)
    d_star = w_need + p.B_c * np.log(max(p.A_c / max(p.a_pin * (1 - p.urgency_pin_relief), 1e-3), 1.001))
    ax.axhline(d_star, color=CAT_LIGHT["violet"], lw=1.2, ls="--",
               label=f"depinning prediction d* = {d_star:.1f} m")
    ax.set_xlabel("distance ahead of EV, s (m)")
    ax.set_ylabel("|lateral offset from corridor| (m)")
    ax.set_title("Bow wave of yielding in the EV comoving frame "
                 f"(t = {hist.t[k0]:.0f}-{hist.t[k1-1]:.0f} s aggregated)",
                 fontsize=10.5, loc="left")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False,
              labelcolor=pal["ink2"], ncol=1)
    _save(fig, path)


# ============================================================ phase diagram
def plot_phase(dens, Ac, ratio, ttc, p, path):
    pal = LIGHT
    fig = _fig(9.2, 5.0, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    vmin = max(0.55, np.floor(ratio.min() * 20) / 20 - 0.05)
    pc = ax.pcolormesh(dens, Ac, ratio, cmap=CMAP_SEQ, vmin=vmin, vmax=1.0,
                       shading="auto")
    cs = ax.contour(dens, Ac, ratio, levels=[0.85, 0.95], colors=["#ffffff"],
                    linewidths=[1.1, 1.8], linestyles=["--", "-"])
    ax.clabel(cs, fmt={0.85: "85%", 0.95: "95%"}, fontsize=8.5, colors="#ffffff")
    unsafe = ttc < 1.2
    if unsafe.any():
        ax.contourf(dens, Ac, unsafe.astype(float), levels=[0.5, 1.5],
                    colors=[CAT_LIGHT["red"]], alpha=0.18, hatches=["////"])
    for val, lab in ((p.a_pin, "depinning threshold  $A_c = a_{pin}$"),
                     (p.a_pin * (1 - p.urgency_pin_relief),
                      "with urgency relief")):
        ax.axhline(val, color="#ffb0e6", lw=1.4,
                   ls="--" if "relief" in lab else "-")
        ax.annotate(lab, (dens.max() * 0.995, val), ha="right", va="bottom",
                    fontsize=9, color="#ffffff",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0b0b0b",
                              ec="none", alpha=0.55))
    cb = fig.colorbar(pc, ax=ax, pad=0.012)
    cb.set_label("EV speed ratio  $\\bar v_{EV}/v^0_{EV}$", color=pal["ink2"],
                 fontsize=9)
    cb.ax.tick_params(colors=pal["muted"], labelsize=8)
    cb.outline.set_visible(False)
    ax.set_xlabel("traffic density (veh/km/lane)")
    ax.set_ylabel("corridor force strength A$_c$ (m/s$^2$)")
    ax.set_title("Corridor-formation phase diagram: EV progress vs density and "
                 "force strength (hatched red: min TTC < 1.2 s)",
                 fontsize=10.5, loc="left")
    _save(fig, path)


# ============================================================ comparisons
def plot_compare(metrics_pairs, path):
    """Grouped bars: EV mean speed, baseline vs force model, per scenario.
    Optional per-entry 'base_err'/'model_err' (km/h) draw error bars."""
    pal = LIGHT
    labels = [m["scenario"] for m in metrics_pairs]
    base = [m["baseline"]["ev_mean_speed"] * 3.6 for m in metrics_pairs]
    mod = [m["model"]["ev_mean_speed"] * 3.6 for m in metrics_pairs]
    des = [m["v0_ev"] * 3.6 for m in metrics_pairs]
    base_err = [m.get("base_err") for m in metrics_pairs]
    mod_err = [m.get("model_err") for m in metrics_pairs]
    has_err = all(e is not None for e in base_err + mod_err)

    fig = _fig(8.4, 4.0, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    xpos = np.arange(len(labels), dtype=float)
    wd = 0.34
    ekw = dict(capsize=4, ecolor=LIGHT["ink"], error_kw=dict(lw=1.3)) if has_err else {}
    ax.bar(xpos - wd / 2 - 0.01, base, wd, color=pal["axis"], label="no yielding",
           yerr=base_err if has_err else None, **ekw)
    ax.bar(xpos + wd / 2 + 0.01, mod, wd, color=CAT_LIGHT["blue"], label="force model",
           yerr=mod_err if has_err else None, **ekw)
    for i, (b, mo, d) in enumerate(zip(base, mod, des)):
        ax.plot([i - 0.5 * wd - wd / 2, i + 0.5 * wd + wd / 2], [d, d],
                color=CAT_LIGHT["red"], lw=1.4, ls=":")
        ob = base_err[i] + 0.6 if has_err else 0.0
        om = mod_err[i] + 0.6 if has_err else 0.0
        ax.annotate(f"{mo:.0f}", (i + wd / 2 + 0.01, mo + om), ha="center",
                    va="bottom", fontsize=9, color=pal["ink"])
        ax.annotate(f"{b:.0f}", (i - wd / 2 - 0.01, b + ob), ha="center",
                    va="bottom", fontsize=9, color=pal["ink2"])
        ax.annotate(f"+{(mo / b - 1) * 100:.0f}%", (i + wd / 2 + 0.01, mo * 0.5),
                    ha="center", fontsize=9.5, color="#ffffff", weight="bold")
    ax.plot([], [], color=CAT_LIGHT["red"], lw=1.4, ls=":", label="EV desired speed")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("EV mean speed (km/h)")
    ax.set_title("Emergency-vehicle progress with and without the repulsion model",
                 fontsize=10.5, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=pal["ink2"])
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def plot_reactions(m_model, path, target=(50, 150)):
    pal = LIGHT
    fig = _fig(8.0, 3.8, pal)
    ax = fig.add_subplot(111)
    _style_ax(ax, pal)
    r = m_model["react_dists"]
    ax.axvspan(target[0], target[1], color=CAT_LIGHT["aqua"], alpha=0.14,
               label="empirical onset band [18, 20]")
    ax.hist(r, bins=np.arange(0, 260, 12), color=CAT_LIGHT["blue"],
            edgecolor=pal["surface"], lw=1.2)
    ax.axvline(np.mean(r), color=CAT_LIGHT["red"], lw=1.6,
               label=f"mean {np.mean(r):.0f} m")
    ax.set_xlabel("distance ahead of EV at yield onset (m)")
    ax.set_ylabel("drivers")
    ax.set_title("When do drivers start to move? (perception + urgency gating)",
                 fontsize=10.5, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=pal["ink2"])
    _save(fig, path)


def plot_calibration(calib, path):
    pal = LIGHT
    evals = calib["evals"]
    names = [k for k in evals[0] if k != "loss"]
    losses = np.array([e["loss"] for e in evals])
    best = calib["theta"]
    fig = _fig(11.5, 6.2, pal)
    axs = fig.subplots(2, 3).ravel()
    cap = np.percentile(losses, 92)
    for ax, nm in zip(axs, names):
        _style_ax(ax, pal)
        xv = np.array([e[nm] for e in evals])
        ax.scatter(xv, np.minimum(losses, cap), s=14, color=CAT_LIGHT["blue"],
                   alpha=0.55, edgecolors="none")
        ax.axvline(best[nm], color=CAT_LIGHT["red"], lw=1.6)
        ax.annotate(f"{best[nm]:.2f}", (best[nm], cap * 0.95), fontsize=9,
                    color=CAT_LIGHT["red"], ha="left")
        ax.set_xlabel(nm); ax.set_ylabel("loss")
    fig.suptitle("Calibration landscape: all evaluations (LHS + Nelder-Mead); "
                 f"red = optimum, loss {calib['loss']:.2f} "
                 f"(defaults {calib['loss_default']:.2f})",
                 color=pal["ink"], fontsize=11, x=0.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
