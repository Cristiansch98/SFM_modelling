"""Real-time parameter lab: the endless version of out/anim_*.gif.

Same dark top-down look as the recorded animations, but the world never ends and
the eight parameters that decide the outcome sit on live sliders. Nothing is
re-implemented: the sliders mutate the single Params object that emv.forces and
emv.perception read every step, so a drag shows up in the traffic within one
integration step.

    python experiments/live_lab.py                      # free-flow motorway
    python experiments/live_lab.py --scenario jam       # rescue-lane formation
    python experiments/live_lab.py --preset game --fps 30
    python experiments/live_lab.py --seed 5 --density 40

Sliders (model): A_c, B_c, a_pin, A_ev, B_ev, T_react, R_front, p_noncomply.
Sliders (world): traffic density, EV desired speed, time scale.
Keys: [space] pause  [r] reset  [1/2/3] preset  [h/j] scenario  [g] record GIF
      [ and ] time scale  [q] quit.

Endless: cars that leave the band around the EV re-enter at the opposite end
with a fresh desired speed, reaction delay and compliance draw, so p_noncomply
and the delay knobs flow into the fleet as it recycles. The density slider adds
and removes cars outside the visible span, so the run continues while you retune.

Rendering: measured on this machine, a full matplotlib redraw of this figure
costs ~150 ms (Agg raster ~55 ms + TkAgg canvas blit ~48 ms + ~0.25 ms per text
character), i.e. 6 fps - so the lab never does one while running. Static art is
cached as a background image; each frame restores it, draws ~10 animated artists
(one PolyCollection for all cars) and pushes only the changed regions to Tk
(one axes = 3.6 ms, whole canvas = 48 ms). Text is the most expensive artist
here, so the status line refreshes at 4 Hz and the readout block at 2 Hz.
"""
import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import fields as dc_fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from emv.params import Params, tuned_params
from emv.road import Road
from emv.runlog import RunLogger
from emv.scenarios import game_params
from emv.scenarios import _assemble, _spawn_traffic      # scenario builders reused verbatim
from emv.state import VehState, UNAWARE, NOTICED, YIELDING, HOLD, STATE_NAMES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
FIELDS = [f.name for f in dc_fields(VehState)]

# ------------------------------------------------------------------ scenarios
SCEN = {
    "highway": dict(label="free-flow motorway", lane_speeds=(24.5, 27.5, 30.0),
                    density=22.0, ev_v0=36.0, ev_v=30.0, corridor="ev_lane",
                    shoulders=(2.0, 0.8)),
    "jam": dict(label="stop-and-go jam", lane_speeds=(2.5, 2.5, 2.5),
                density=72.0, ev_v0=9.0, ev_v=6.0, corridor="boundary",
                shoulders=(2.5, 1.0)),
}
PRESETS = {"default": Params, "tuned": tuned_params, "game": game_params}

# slider spec: (attribute, label, lo, hi, fmt). Ranges cover every preset.
PHYS_SLIDERS = [
    ("A_c",         "A_c   corridor push [m/s2]",    0.0,   8.0, "%.2f"),
    ("B_c",         "B_c   corridor width [m]",      0.3,   3.0, "%.2f"),
    ("a_pin",       "a_pin  lane discipline [m/s2]", 0.0,   4.0, "%.2f"),
    ("A_ev",        "A_ev  EV repulsion [m/s2]",     0.0,   8.0, "%.2f"),
    ("B_ev",        "B_ev  siren range [m]",         5.0,  45.0, "%.1f"),
    ("T_react",     "T_react  anticipation [s]",     1.0,  20.0, "%.1f"),
    ("R_front",     "R_front  detection [m]",       20.0, 320.0, "%.0f"),
    ("p_noncomply", "p_noncomply  never yield [-]",  0.0,  0.60, "%.2f"),
]
WORLD_SLIDERS = [
    ("density", "traffic density [veh/km/lane]",  6.0, 85.0, "%.0f"),
    ("ev_v0",   "EV desired speed [km/h]",       20.0, 190.0, "%.0f"),
    ("scale",   "time scale [x real time]",       0.1,  3.0, "%.2f"),
]

# figure + panel geometry (figure fractions). The road strip has a fixed data
# aspect, so the cell height ratio is chosen to leave little dead space around it.
FIGSIZE = (16.2, 6.6)
ROAD_ASPECT, HEIGHT_RATIOS = 3.0, (1.7, 1.2)
PANEL = 0.245
PX, PW = 0.775, 0.200
DY, SH = 0.0525, 0.019          # slider row pitch / track height
BTN_DY, BTN_H = 0.054, 0.038
Y_HEAD, Y_TOP = 0.968, 0.925    # first section header / first slider track
GAP_HEAD, DROP_HEAD = 0.033, 0.043    # track->next header, header->its track
TRACE_S = 45.0                      # seconds of history in the insets
DASH_PERIOD = 14.0                  # m, lane-marking period (7 m on, 7 m off)
# Text costs ~0.25 ms per character to rasterise here, so the three text blocks
# refresh on staggered schedules (frame % period == phase) and never land on the
# same frame. Each is blitted only on its own frames, so skipping a frame does
# not blank it - the region simply keeps what was last pushed to screen.
TEXT_SCHED = dict(title=(6, 0), hud_pin=(12, 3), hud=(12, 9))


# ======================================================== VehState array surgery
def _subset(st: VehState, keep) -> VehState:
    return VehState(**{f: np.asarray(getattr(st, f))[keep].copy() for f in FIELDS})


def _concat(st: VehState, rows: dict) -> VehState:
    return VehState(**{f: np.concatenate([np.asarray(getattr(st, f)), rows[f]])
                       for f in FIELDS})


# ================================================================ endless world
class EndlessWorld:
    """A scenario that never runs out: a fixed band around the EV, with cars
    recycled from one end to the other (same idea as the UE bridge's endless
    mode, emv/ue/server.py::_recycle)."""

    BACK, FRONT = 260.0, 620.0      # m of live world behind / ahead of the EV

    def __init__(self, name: str, seed: int, p: Params, density=None):
        self.name, self.seed, self.p = name, int(seed), p
        self.cfg = SCEN[name]
        self.density = float(self.cfg["density"] if density is None else density)
        self.rng = np.random.default_rng(seed + 91)
        self.build()

    # ---------------------------------------------------------------- geometry
    @property
    def spacing(self) -> float:
        return 1000.0 / max(self.density, 1.0)

    @property
    def band(self) -> float:
        return self.BACK + self.FRONT

    def build(self):
        cfg, p = self.cfg, self.p
        rng = np.random.default_rng(self.seed)
        road = Road(n_lanes=3, lane_width=3.5, shoulder_right=cfg["shoulders"][0],
                    shoulder_left=cfg["shoulders"][1], length=self.band)
        y_corr = (float(road.lane_center(1)) if cfg["corridor"] == "ev_lane"
                  else (road.n_lanes - 1) * road.lane_width)
        sp = self.spacing
        xs, ys, _, v0s = _spawn_traffic(rng, road, -self.BACK, self.FRONT, sp,
                                        list(cfg["lane_speeds"]),
                                        jitter_x=min(0.18 * sp, 2.4), p=p)
        keep = ~((np.abs(xs) < 30.0) & (np.abs(ys - y_corr) < 3.5))   # clear the EV spawn
        ev_kw = dict(x=0.0, y=y_corr, v_init=cfg["ev_v"], v0=cfg["ev_v0"],
                     y_corr=y_corr)
        self.road = road
        self.sim = _assemble(rng, road, p, xs[keep], ys[keep], v0s[keep], ev_kw,
                             self.seed, True, f"live_{self.name}")

    # ------------------------------------------------------------------- spawns
    def _lane_of(self, y):
        return np.clip(np.round(np.asarray(y) / self.road.lane_width - 0.5),
                       0, self.road.n_lanes - 1).astype(int)

    def _edges(self, ex: float, head: bool) -> np.ndarray:
        """End of each lane's chain at the head (or tail) of the band. Only cars
        *inside* the band count: a lane whose leaders have already run out the
        front must not drag the spawn point along with them (runaway drift)."""
        st = self.sim.st
        lane = self._lane_of(st.y)
        inband = (~st.is_ev) & (st.x <= ex + self.FRONT) & (st.x >= ex - self.BACK)
        edge = np.empty(self.road.n_lanes)
        for k in range(self.road.n_lanes):
            sel = inband & (lane == k)
            if head:
                edge[k] = max(ex + 0.75 * self.FRONT,
                              float(st.x[sel].max()) if sel.any() else -np.inf)
            else:
                edge[k] = min(ex - 0.75 * self.BACK,
                              float(st.x[sel].min()) if sel.any() else np.inf)
        return edge

    def _new_rows(self, m: int, ex: float, head: bool = True) -> dict:
        """m fresh cars entering at the head (or tail) of the band, sparsest lane
        first. Driver draws use the *current* Params, which is how p_noncomply
        and the reaction-delay knobs reach traffic as it recycles."""
        p, rng, road = self.p, self.rng, self.road
        edge = self._edges(ex, head)
        xs, ys, v0s = np.empty(m), np.empty(m), np.empty(m)
        for j in range(m):
            k = int(np.argmin(edge) if head else np.argmax(edge))
            step = self.spacing * float(rng.uniform(0.85, 1.15))
            edge[k] += step if head else -step
            xs[j] = edge[k]
            ys[j] = road.lane_center(k) + float(rng.uniform(-0.25, 0.25))
            v0s[j] = max(float(rng.normal(self.cfg["lane_speeds"][k], 1.2)), 2.0)
        z, one = np.zeros(m), np.ones(m)
        return dict(
            x=xs, y=ys, vx=v0s * 0.97, vy=z.copy(),
            L=one * 4.5, W=one * 1.8, v0=v0s, tau=one * p.tau,
            T_hw=rng.uniform(1.1, 1.8, m), s0=one * p.s0,
            amax=one * p.a_max, bcomf=one * p.b_comf,
            is_ev=np.zeros(m, bool), comply=rng.random(m) >= p.p_noncomply,
            aware=np.full(m, UNAWARE, np.int8), side=np.zeros(m, np.int8),
            t_react=np.full(m, np.inf), hold_until=np.full(m, np.inf),
            delay=np.exp(rng.normal(np.log(p.delay_med), p.delay_sig, m)),
            u=z.copy(), y_init=ys.copy())

    def recycle(self):
        """Cars leave through one end of the band and re-enter through the other:
        overtaken by the EV -> back to the head; outran a blocked EV -> back to
        the tail. Both happen (a jammed EV is slower than lane 2)."""
        st = self.sim.st
        ex = float(st.x[st.ev])
        for head, gone in (
                (True, np.flatnonzero((~st.is_ev) & (st.x < ex - self.BACK))),
                (False, np.flatnonzero((~st.is_ev) & (st.x > ex + self.FRONT + 120.0)))):
            if gone.size == 0:
                continue
            rows = self._new_rows(gone.size, ex, head=head)
            for f in FIELDS:
                np.asarray(getattr(st, f))[gone] = rows[f]

    # ------------------------------------------------------------------ density
    def set_density(self, d: float):
        """Add/remove cars *outside the visible span* so the run continues."""
        self.density = float(d)
        st = self.sim.st
        target = int(round(self.density * self.road.n_lanes * self.band / 1000.0))
        ex = float(st.x[st.ev])
        delta = target - (st.n - 1)
        if delta > 0:
            self.sim.st = _concat(st, self._new_rows(delta, ex, head=True))
        elif delta < 0:
            dx = st.x - ex
            cand = np.flatnonzero((~st.is_ev) & ((dx > 260.0) | (dx < -210.0)))
            if cand.size == 0:
                return                      # nothing off-screen to take away yet
            drop = cand[np.argsort(-np.abs(dx[cand]))][:-delta]   # farthest first
            keep = np.ones(st.n, bool)
            keep[drop] = False
            self.sim.st = _subset(st, keep)

    def step(self):
        self.sim.step()
        self.recycle()


# =================================================================== rendering
def _car_polys(x, y, L, W, cx=0.55, cy=0.21):
    """(M, 8, 2) chamfered rectangles - the rounded look of viz._car_patch at a
    fraction of the cost (one PolyCollection instead of M FancyBboxPatches)."""
    hl, hw = 0.5 * np.asarray(L), 0.5 * np.asarray(W)
    dx = np.stack([-hl + cx, hl - cx, hl, hl, hl - cx, -hl + cx, -hl, -hl], -1)
    dy = np.stack([-hw, -hw, -hw + cy, hw - cy, hw, hw, hw - cy, -hw + cy], -1)
    return np.stack([np.asarray(x)[:, None] + dx, np.asarray(y)[:, None] + dy], -1)


class _Scene:
    """The animation figure: everything static is built once and cached as a
    background bitmap, every frame only updates and redraws animated artists.
    Coordinates are EV-relative, so axis limits (and hence ticks, text layout and
    font lookups) never change."""

    def __init__(self, road: Road, p: Params, span=(90.0, 200.0)):
        from emv import viz                     # NOTE: viz forces Agg on import ...
        import matplotlib
        for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)      # ... so switch back, after
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.colors import to_rgba
        from matplotlib.font_manager import FontProperties, findfont
        from matplotlib.patches import Rectangle

        # A concrete font file skips matplotlib's per-character family/fallback
        # resolution, which is ~25 % of the cost of drawing a text artist here.
        self.mono = FontProperties(fname=findfont(FontProperties(family="monospace")))
        self.viz, self.plt, self.road, self.p, self.span = viz, plt, road, p, span
        pal = self.pal = viz.DARK
        self.state_rgba = np.array([to_rgba(viz.STATE_FILL_DARK[s])
                                    for s in (UNAWARE, NOTICED, YIELDING, HOLD)])
        self.ev_rgba = to_rgba(viz.EV_DARK)
        self.tr = {k: deque() for k in ("t", "ev", "bg", "clr")}

        self.fig = fig = plt.figure(figsize=FIGSIZE, dpi=100)
        try:
            fig.canvas.manager.set_window_title("EMV live parameter lab")
        except Exception:
            pass
        fig.patch.set_facecolor(pal["page"])
        gs = fig.add_gridspec(2, 2, height_ratios=list(HEIGHT_RATIOS), hspace=0.42,
                              wspace=0.22, left=0.055, right=0.985 - PANEL,
                              top=0.90, bottom=0.115)
        ax = self.ax = fig.add_subplot(gs[0, :])
        self.ax_v = fig.add_subplot(gs[1, 0])
        self.ax_c = fig.add_subplot(gs[1, 1])
        for a in (self.ax_v, self.ax_c):
            viz._style_ax(a, pal)

        # ------------------------------------------------- static road (EV frame)
        ax.set_facecolor(pal["page"])
        self.dashes = LineCollection([], colors="#e1e0d9", linewidths=1.0,
                                     alpha=0.55, zorder=1)
        ax.add_collection(self.dashes)
        self._road_static = []
        self.rebuild_road(road)

        # ------------------------------------------------------ animated artists
        self.band = Rectangle((0, 0), 1, 1, fc=viz.EV_DARK, alpha=0.13, ec="none",
                              zorder=1.5)
        ax.add_patch(self.band)
        self.band_line, = ax.plot([], [], color=viz.EV_DARK, lw=1.0,
                                  ls=(0, (5, 6)), alpha=0.55, zorder=1.6)
        self.cars = PolyCollection([], ec="none", zorder=3)
        ax.add_collection(self.cars)
        self.ev = PolyCollection([], ec="none", zorder=4)
        ax.add_collection(self.ev)
        self.beacon = Rectangle((0, 0), 1.1, 0.76, fc="#ffffff", ec="none", zorder=5)
        ax.add_patch(self.beacon)

        ax.set_xlim(-span[0], span[1])         # y limits: set by rebuild_road()
        ax.set_aspect(ROAD_ASPECT)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-50.0, span[1] + 1.0, 50.0))
        ax.set_xlabel("distance from EV (m)", color=pal["ink2"], fontsize=8.5)
        ax.tick_params(colors=pal["muted"], labelsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        self.title = ax.set_title("", color=pal["ink"], loc="left",
                                  fontproperties=self.mono, fontsize=10.5)
        ax.legend(handles=[plt.Line2D([], [], marker="s", ls="none", ms=8, mec="none",
                                      mfc=viz.STATE_FILL_DARK[s], label=STATE_NAMES[s])
                           for s in (UNAWARE, NOTICED, YIELDING, HOLD)]
                          + [plt.Line2D([], [], marker="s", ls="none", ms=8,
                                        mec="none", mfc=viz.EV_DARK, label="EV")],
                  loc="upper right", ncol=5, fontsize=8, frameon=False,
                  labelcolor=pal["ink2"], bbox_to_anchor=(1.0, 1.30))
        # 1.30: above the status line (which sits at ~1.15 - the axes box is
        # short here because the data aspect is fixed) and still inside the
        # figure; draw()'s title strip is sized to cover both.

        # --------------------------------------------------------------- insets
        self.l_ev, = self.ax_v.plot([], [], color=viz.EV_DARK, lw=2.0, label="EV")
        self.l_bg, = self.ax_v.plot([], [], color=pal["muted"], lw=1.6, ls="--",
                                    label="traffic mean")
        self.ax_v.legend(loc="lower right", fontsize=8, frameon=False,
                         labelcolor=pal["ink2"])
        self.l_clr, = self.ax_c.plot([], [], color=viz.CAT_DARK["blue"], lw=2.0)
        self.ax_v.set_ylabel("speed (km/h)")
        self.ax_c.set_ylabel("clear corridor ahead (m)")
        for a, ylim in ((self.ax_v, (0, 175)), (self.ax_c, (0, 260))):
            a.set_xlim(-TRACE_S, 0.0)          # fixed: no tick work per frame
            a.set_ylim(*ylim)
            a.set_xlabel("seconds ago")

        self.road_art = [self.dashes, self.band, self.band_line, self.cars,
                         self.ev, self.beacon]
        self.inset_art = [(self.l_ev, self.ax_v), (self.l_bg, self.ax_v),
                          (self.l_clr, self.ax_c)]
        for a in self.road_art + [self.title] + [x[0] for x in self.inset_art]:
            a.set_animated(True)
        fig.show()

    def rebuild_road(self, road: Road):
        """(Re)draw the static road for `road`. Called on scenario switches: the
        two scenarios differ in shoulder width, so the painted band and the y
        limits have to follow. Caller must trigger a full canvas draw afterwards
        so the blit background picks it up."""
        ax, viz, pal = self.ax, self.viz, self.pal
        for a in self._road_static:
            a.remove()
        x0, x1 = -self.span[0], self.span[1]
        before_p, before_l = list(ax.patches), list(ax.lines)
        viz._draw_road(ax, road, x0 - DASH_PERIOD, x1 + DASH_PERIOD, pal)
        new_l = [a for a in ax.lines if a not in before_l]
        for line in new_l[:road.n_lanes - 1]:
            line.remove()                      # dashed lane markings: scrolled below
        self._road_static = ([a for a in ax.patches if a not in before_p]
                             + new_l[road.n_lanes - 1:])
        seg = np.array([[[s, 0.0], [s + 0.5 * DASH_PERIOD, 0.0]]
                        for s in np.arange(x0 - DASH_PERIOD, x1 + DASH_PERIOD,
                                           DASH_PERIOD)])
        self.dash_tpl = np.concatenate(
            [seg + [0.0, k * road.lane_width] for k in range(1, road.n_lanes)])
        self.dashes.set_segments(self.dash_tpl)
        ax.set_ylim(road.y_min - 1.6, road.y_max + 1.6)
        self.road = road

    # ------------------------------------------------------------------- update
    def update(self, st: VehState, t: float, y_corr: float):
        p, e = self.p, st.ev
        x_ev, y_ev, v_ev = float(st.x[e]), float(st.y[e]), float(st.vx[e])
        rel = st.x - x_ev
        others = ~st.is_ev

        near = others & (np.abs(rel) < 300.0)
        bg = 3.6 * float(st.vx[near].mean()) if near.any() else 0.0
        w_need = 0.5 * (st.W[e] + st.W) + p.margin_c
        gap = rel - 0.5 * (st.L[e] + st.L)
        blocking = others & (np.abs(st.y - y_corr) < w_need) & (gap > 0.0)
        clr = min(float(gap[blocking].min()), 250.0) if blocking.any() else 250.0

        for k, v in (("t", t), ("ev", 3.6 * v_ev), ("bg", bg), ("clr", clr)):
            self.tr[k].append(v)
        while self.tr["t"] and self.tr["t"][-1] - self.tr["t"][0] > TRACE_S:
            for d in self.tr.values():
                d.popleft()

        self.dashes.set_segments(self.dash_tpl - [x_ev % DASH_PERIOD, 0.0])

        L_c = min(240.0, p.L_pred_max)
        w_c = 0.5 * st.W[e] + p.margin_c
        self.band.set_bounds(0.0, y_corr - w_c, L_c, 2.0 * w_c)
        self.band_line.set_data([0.0, L_c], [y_corr, y_corr])

        vis = others & (rel > -self.span[0] - 8.0) & (rel < self.span[1] + 8.0)
        self.cars.set_verts(_car_polys(rel[vis], st.y[vis], st.L[vis], st.W[vis]))
        self.cars.set_facecolors(self.state_rgba[st.aware[vis]])

        blink = int(t * 3.0) % 2 == 0
        self.ev.set_verts(_car_polys(np.zeros(2), np.full(2, y_ev),
                                     [st.L[e] + 2.6, st.L[e]],
                                     [st.W[e] + 2.6, st.W[e]]))
        self.ev.set_facecolors([self.ev_rgba[:3] + (0.28 if blink else 0.16,),
                                self.ev_rgba])
        self.beacon.set_bounds((0.9 if blink else -0.9) - 0.55, y_ev - 0.38,
                               1.1, 0.76)
        self.beacon.set_facecolor("#ffffff" if blink else "#9ec5f4")

        rel_t = np.asarray(self.tr["t"]) - t
        self.l_ev.set_data(rel_t, self.tr["ev"])
        self.l_bg.set_data(rel_t, self.tr["bg"])
        self.l_clr.set_data(rel_t, self.tr["clr"])
        n_yield = int(((st.aware == YIELDING) | (st.aware == HOLD))[others].sum())
        return 3.6 * v_ev, clr, n_yield

    def set_status(self, text):
        self.title.set_text(text)

    def draw(self, with_title: bool, with_insets: bool = True):
        """Draw the animated artists into the restored buffer; return the screen
        regions that need pushing. Artists that are skipped (title, insets) must
        also have their region skipped - the region then keeps whatever was last
        pushed to the screen, which is why they can run on slower schedules."""
        from matplotlib.transforms import Bbox
        for a in self.road_art:
            self.ax.draw_artist(a)
        regions = [self.ax.bbox]
        if with_insets:                        # traces move slowly: half rate
            for a, ax in self.inset_art:
                ax.draw_artist(a)
            regions += [self.ax_v.bbox, self.ax_c.bbox]
        if with_title:
            self.ax.draw_artist(self.title)
            b = self.ax.bbox
            # tall enough to cover title + legend (both sit above the shrunk,
            # aspect-constrained axes box); the legend is static, so re-pushing
            # its pixels is free of charge and keeps the strip robust
            regions.append(Bbox.from_extents(b.x0, b.y1 + 1.0, b.x1,
                                             min(b.y1 + 90.0, self.fig.bbox.y1)))
        return regions

    def clear_traces(self):
        for d in self.tr.values():
            d.clear()


class _Blit:
    """Background cache + region blitter. A full canvas blit costs ~48 ms here,
    a single axes 3.6 ms, so only changed regions are pushed."""

    def __init__(self, fig):
        self.fig, self.canvas = fig, fig.canvas
        self.bg, self.stale = None, True
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, _evt=None):
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self.stale = True              # animated artists are missing from the bg

    def restore(self) -> bool:
        """True once a background exists (else a full draw is issued first)."""
        if self.bg is None:
            self.canvas.draw()
        if self.bg is None:
            return False
        self.canvas.restore_region(self.bg)
        return True

    def push(self, regions):
        for r in regions:
            self.canvas.blit(r)


# =================================================================== GIF record
class _Rec:
    """Frame grabber with one fixed palette (same trick as
    experiments/make_survey_media.py::_Gif: no colour flicker, low memory)."""

    def __init__(self, path, fps, crop_w, stride=2, cap=320):
        from PIL import Image
        self.Image, self.path, self.fps = Image, path, fps
        self.crop_w, self.stride, self.cap = crop_w, stride, cap
        self.pal, self.frames, self.stamps = None, [], []

    def grab(self, fig):
        # Slice the canvas buffer directly (crop to the animation half and
        # decimate): building a full-size PIL image and resampling it costs
        # ~67 ms per frame here, this costs ~10 ms.
        self.stamps.append(time.perf_counter())
        s = self.stride
        im = self.Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[::s, :self.crop_w:s, :3])
        if self.pal is None:
            self.pal = im.quantize(colors=255, method=self.Image.MEDIANCUT,
                                   dither=self.Image.Dither.NONE)
        self.frames.append(im.quantize(palette=self.pal,
                                       dither=self.Image.Dither.NONE))
        return len(self.frames) < self.cap

    def save(self):
        if not self.frames:
            return None
        # Play back at the rate the frames were actually captured (grabbing +
        # forcing every text block costs time), so the GIF runs at real speed.
        fps = self.fps
        if len(self.stamps) > 4:
            dt = float(np.median(np.diff(self.stamps)))
            fps = float(np.clip(1.0 / max(dt, 1e-3), 4.0, 25.0))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.frames[0].save(self.path, save_all=True,
                            append_images=self.frames[1:],
                            duration=int(round(1000.0 / fps)), loop=0)
        print(f"wrote {self.path} ({os.path.getsize(self.path)/1e6:.1f} MB, "
              f"{len(self.frames)} frames, {fps:.1f} fps)", flush=True)
        return self.path


# ======================================================================= the lab
class LiveLab:
    def __init__(self, scenario="highway", preset="tuned", seed=1, fps=20.0,
                 density=None, span=(90.0, 200.0), rec_max=320):
        self.p = PRESETS[preset]()
        self.preset, self.seed, self.fps = preset, int(seed), float(fps)
        self.scenario, self.rec_max, self.span = scenario, rec_max, span
        self.world = EndlessWorld(scenario, seed, self.p, density)
        self.paused, self.scale, self._acc = False, 1.0, 0.0
        self.peak_contacts, self._loop_dt, self._last_dens = 0, 1.0 / fps, 0.0
        self.rec, self.outputs = None, []
        self.scene = _Scene(self.world.road, self.p, span=span)
        self._dirty = set()                  # widget axes needing a re-blit
        self._build_ui()
        self.blit = _Blit(self.scene.fig)

    # ---------------------------------------------------------------------- UI
    def _rect_bbox(self, rect):
        """Figure-fraction rect -> display Bbox (recomputed, so resize-safe)."""
        from matplotlib.transforms import Bbox
        w, h = self.fig.canvas.get_width_height()
        x, y, dw, dh = rect
        return Bbox.from_bounds(x * w, y * h, dw * w, dh * h)

    def _build_ui(self):
        import matplotlib
        from matplotlib.widgets import Button, Slider
        for km in ("grid", "grid_minor", "yscale", "xscale", "save", "pan",
                   "zoom", "home", "fullscreen", "back", "forward", "copy"):
            matplotlib.rcParams[f"keymap.{km}"] = []      # free the keyboard
        self.plt, self.fig = self.scene.plt, self.scene.fig
        pal, viz = self.scene.pal, self.scene.viz
        self.rects = {}                      # widget key -> figure-fraction rect

        def head(y, text):
            self.fig.text(PX, y, text, color=pal["muted"], fontsize=8.5,
                          family="monospace", weight="bold")

        def slider(key, y, label, lo, hi, val, fmt):
            ax = self.fig.add_axes([PX, y, PW, SH])
            ax.patch.set_alpha(0.0)
            s = Slider(ax, label, lo, hi, valinit=float(np.clip(val, lo, hi)),
                       valfmt=fmt, color=viz.CAT_DARK["blue"],
                       track_color="#2c2c2a", initcolor="none",
                       handle_style=dict(facecolor="#e1e0d9", edgecolor="none",
                                         size=9))
            s.label.set_position((0.0, 1.45))
            s.label.set_horizontalalignment("left")
            s.label.set_color(pal["ink2"])
            s.label.set_fontsize(8)
            s.valtext.set_position((1.0, 1.45))
            s.valtext.set_horizontalalignment("right")
            s.valtext.set_color(pal["ink"])
            s.valtext.set_fontsize(8)
            s.valtext.set_family("monospace")
            s.drawon = False                 # no full redraw per drag event
            for a in (s.poly, s._handle, s.valtext):
                a.set_animated(True)
            self.rects[key] = (PX - 0.004, y - 0.006, PW + 0.008, SH + 0.032)
            return s

        head(Y_HEAD, "MODEL PARAMETERS")
        self.sl = {}
        y = Y_TOP
        for attr, label, lo, hi, fmt in PHYS_SLIDERS:
            s = slider(attr, y, label, lo, hi, getattr(self.p, attr), fmt)
            s.on_changed(lambda v, a=attr: self._set_param(a, v))
            self.sl[attr] = s
            y -= DY

        y += DY - GAP_HEAD
        head(y, "WORLD")
        y -= DROP_HEAD
        init = dict(density=self.world.density,
                    ev_v0=self.world.cfg["ev_v0"] * 3.6, scale=1.0)
        for key, label, lo, hi, fmt in WORLD_SLIDERS:
            s = slider(key, y, label, lo, hi, init[key], fmt)
            s.on_changed(getattr(self, f"_set_{key}"))
            self.sl[key] = s
            y -= DY

        # ------------------------------------------------------------- buttons
        y += DY - GAP_HEAD
        head(y, "PRESET / SCENARIO")
        y_btn = y - 0.052
        self.btn = {}

        def button(col, row, label, cb):
            rect = (PX + col * 0.069, y_btn - row * BTN_DY, 0.062, BTN_H)
            ax = self.fig.add_axes(rect)
            b = Button(ax, label, color="#232325", hovercolor=viz.CAT_DARK["blue"])
            b.label.set_color(pal["ink2"])
            b.label.set_fontsize(8)
            b.drawon = False                 # hover repaint goes through _dirty
            b.on_clicked(lambda _e: cb())
            self.rects[f"btn_{label}"] = rect
            self.btn[label] = b

        for c, name in enumerate(PRESETS):
            button(c, 0, name, lambda n=name: self.apply_preset(n))
        button(0, 1, "highway", lambda: self.set_scenario("highway"))
        button(1, 1, "jam", lambda: self.set_scenario("jam"))
        button(2, 1, "reset", self.reset)
        button(0, 2, "pause", self.toggle_pause)
        button(1, 2, "rec gif", self.toggle_record)
        button(2, 2, "quit", lambda: self.plt.close(self.fig))
        self._btn_fc = {k: b.ax.get_facecolor() for k, b in self.btn.items()}

        # ----------------------------------------------------------- HUD block
        y1 = y_btn - 2 * BTN_DY - 0.020
        y2 = y1 - 0.058
        mono = self.scene.mono
        self.hud_pin = self.fig.text(PX, y1, "", color=pal["ink"], fontsize=7.6,
                                     fontproperties=mono, va="top", linespacing=1.6)
        self.hud = self.fig.text(PX, y2, "", color=pal["ink2"], fontsize=7.6,
                                 fontproperties=mono, va="top", linespacing=1.6)
        for t in (self.hud_pin, self.hud):
            t.set_animated(True)
            t.set_fontsize(7.6)
        self.rects["hud_pin"] = (PX - 0.004, y1 - 0.052, PW + 0.024, 0.058)
        self.rects["hud"] = (PX - 0.004, max(y2 - 0.080, 0.0), PW + 0.024, 0.086)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # --------------------------------------------------------------- callbacks
    def _on_motion(self, _evt):
        """Buttons repaint themselves on hover; with drawon=False we blit them."""
        for k, b in self.btn.items():
            fc = b.ax.get_facecolor()
            if fc != self._btn_fc[k]:
                self._btn_fc[k] = fc
                self._dirty.add(f"btn_{k}")

    def _set_param(self, attr, val):
        setattr(self.p, attr, float(val))
        self._dirty.add(attr)
        if attr == "p_noncomply":              # redraw the coin for calm drivers
            st = self.world.sim.st
            idle = (st.aware == UNAWARE) & (~st.is_ev)
            st.comply[idle] = self.world.rng.random(int(idle.sum())) >= float(val)

    def _set_density(self, val):
        self.world.set_density(float(val))
        self._dirty.add("density")

    def _set_ev_v0(self, val):
        st = self.world.sim.st
        st.v0[st.ev] = float(val) / 3.6
        self._dirty.add("ev_v0")

    def _set_scale(self, val):
        self.scale = float(val)
        self._dirty.add("scale")

    def apply_preset(self, name):
        src = PRESETS[name]()
        for f in dc_fields(Params):
            setattr(self.p, f.name, getattr(src, f.name))
        self.preset = name
        for attr, _, lo, hi, _fmt in PHYS_SLIDERS:      # slider -> p, clipped
            self.sl[attr].set_val(float(np.clip(getattr(self.p, attr), lo, hi)))

    def set_scenario(self, name):
        self.scenario = name
        self.world = EndlessWorld(name, self.seed, self.p, SCEN[name]["density"])
        self.sl["density"].set_val(SCEN[name]["density"])
        self.sl["ev_v0"].set_val(SCEN[name]["ev_v0"] * 3.6)
        self._restart()

    def reset(self):
        self.world = EndlessWorld(self.scenario, self.seed, self.p,
                                  self.sl["density"].val)
        self._set_ev_v0(self.sl["ev_v0"].val)
        self._restart()

    def _restart(self):
        """Shared tail of reset/scenario switch: the new world brings a new Road
        (the scenarios differ in shoulder width), so the static art is redrawn
        and the blit background re-cached from it."""
        self.peak_contacts = 0
        self.scene.clear_traces()
        self.scene.rebuild_road(self.world.road)
        self.fig.canvas.draw()               # -> draw_event -> fresh background

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_record(self):
        if self.rec is None:
            w, _h = self.fig.canvas.get_width_height()
            self.rec = _Rec(os.path.join(OUT, "anim_live_lab.gif"),
                            fps=min(self.fps, 16.0),
                            crop_w=int(w * (1.0 - PANEL - 0.005)), cap=self.rec_max)
            print(f"[rec] recording up to {self.rec_max} frames -> "
                  f"out/anim_live_lab.gif", flush=True)
        else:
            path, self.rec = self.rec.save(), None
            if path:
                self.outputs.append(path)

    def _on_key(self, ev):
        k = ev.key
        if k == " ":
            self.toggle_pause()
        elif k == "r":
            self.reset()
        elif k in ("1", "2", "3"):
            self.apply_preset(list(PRESETS)[int(k) - 1])
        elif k in ("h", "j"):
            self.set_scenario("highway" if k == "h" else "jam")
        elif k == "g":
            self.toggle_record()
        elif k in ("[", "]"):
            f = 0.8 if k == "[" else 1.25
            self.sl["scale"].set_val(float(np.clip(self.scale * f, 0.1, 3.0)))

    # ------------------------------------------------------------------ readout
    def _near_stats(self):
        st = self.world.sim.st
        e = st.ev
        sel = np.flatnonzero((np.abs(st.x - st.x[e]) < 300.0) & ~st.is_ev)
        counts = [int((st.aware[sel] == s).sum())
                  for s in (UNAWARE, NOTICED, YIELDING, HOLD)]
        near = sel[np.abs(st.x[sel] - st.x[e]) < 150.0]
        contacts, ttc = 0, np.inf
        if near.size > 1:
            x, y, vx = st.x[near], st.y[near], st.vx[near]
            dx = x[None, :] - x[:, None]
            dy = np.abs(y[None, :] - y[:, None])
            half_L = 0.5 * (st.L[near][None, :] + st.L[near][:, None])
            half_W = 0.5 * (st.W[near][None, :] + st.W[near][:, None])
            gap = dx - half_L
            ahead = (dx > 0.0) & (dy < half_W - 0.05)      # metrics._pairwise_safety
            contacts = int((ahead & (gap < 0.0)).sum())
            dv = vx[:, None] - vx[None, :]
            closing = ahead & (dv > 0.3) & (gap > 0.0)
            if closing.any():
                ttc = float((gap[closing] / dv[closing]).min())
        self.peak_contacts = max(self.peak_contacts, contacts)
        v_bg = float(st.vx[sel].mean()) if sel.size else 0.0
        return counts, contacts, ttc, v_bg, int(st.n - 1)

    def _update_pin(self):
        """Depinning verdict: the escape condition A_c*u > a_pin*(1-relief*u) and
        the analytic cleared half-width d* (docs/FRAMEWORK.md sec. 4), evaluated
        at full urgency."""
        p, st = self.p, self.world.sim.st
        a_pin_eff = p.a_pin * (1.0 - p.urgency_pin_relief)
        w_need = 0.5 * (st.W[st.ev] + 1.8) + p.margin_c
        if p.A_c > a_pin_eff and p.A_c > 0.0:
            d_star = (w_need + p.B_c * math.log(p.A_c / a_pin_eff)
                      if a_pin_eff > 0.0 else float("inf"))
            self.hud_pin.set_text(f"A_c {p.A_c:4.2f} > pin {a_pin_eff:4.2f} ESCAPES\n"
                                  f"d* {d_star:4.2f} m  need {w_need:4.2f} m")
            self.hud_pin.set_color(self.scene.viz.CAT_DARK["aqua"])
        else:
            self.hud_pin.set_text(f"A_c {p.A_c:4.2f} < pin {a_pin_eff:4.2f} PINNED\n"
                                  f"lane cannot be opened")
            self.hud_pin.set_color(self.scene.viz.CAT_DARK["red"])

    def _update_block(self):
        st = self.world.sim.st
        e = st.ev
        counts, contacts, ttc, v_bg, n = self._near_stats()
        self.hud.set_text(
            f"EV {3.6*st.vx[e]:5.1f} {100*st.vx[e]/max(st.v0[e],1e-6):3.0f}%"
            f"  bg {3.6*v_bg:4.0f} km/h\n"
            f"U{counts[0]:3d} N{counts[1]:3d} Y{counts[2]:3d} H{counts[3]:3d}"
            f"  n {n:3d}\n"
            f"hit {contacts:2d}/{self.peak_contacts:2d} TTC {ttc:4.1f}s"
            f" {1.0/max(self._loop_dt,1e-3):4.1f}fps")

    def _status(self, kmh, clr, n_yield):
        """The GIF's status line, plus what the controls are currently set to."""
        return (f"t ={self.world.sim.t:6.1f} s  EV {kmh:4.0f} km/h  "
                f"clear {clr:3.0f} m  yield {n_yield:3d}   "
                f"[{self.preset} | {self.scenario} {self.world.density:.0f}"
                f" | x{self.scale:.2f}"
                + ("  PAUSED" if self.paused else "")
                + ("  REC" if self.rec is not None else "") + "]")

    # --------------------------------------------------------------- main loop
    def _draw_widgets(self, keys):
        """Redraw the animated parts of the given widgets; return their regions."""
        out = []
        for k in keys:
            if k in self.sl:
                s = self.sl[k]
                for a in (s.poly, getattr(s, "_handle", None), s.valtext):
                    if a is not None:
                        s.ax.draw_artist(a)
            elif k.startswith("btn_"):
                self.fig.draw_artist(self.btn[k[4:]].ax)
            out.append(self._rect_bbox(self.rects[k]))
        return out

    def run(self, duration: float | None = None) -> dict:
        plt = self.plt
        canvas = self.fig.canvas
        period = 1.0 / self.fps
        t_start = last = time.perf_counter()
        frames, sim_steps = 0, 0
        canvas.draw()
        while plt.fignum_exists(self.fig.number):
            t0 = time.perf_counter()
            if duration is not None and t0 - t_start > duration:
                break
            dt_wall = min(t0 - last, 0.5)
            last = t0
            if not self.paused:
                self._acc += dt_wall * self.scale
                n_steps = int(self._acc / self.p.dt)
                if n_steps > 12:                        # never spiral
                    n_steps, self._acc = 12, 0.0
                else:
                    self._acc -= n_steps * self.p.dt
                for _ in range(n_steps):
                    self.world.step()
                sim_steps += n_steps
            if t0 - self._last_dens > 0.5:              # re-assert the density knob
                self.world.set_density(self.sl["density"].val)
                self._last_dens = t0

            # ---- render: restore cached background, draw only what moves ----
            self.scene.p = self.p
            kmh, clr, n_yield = self.scene.update(
                self.world.sim.st, self.world.sim.t, self.world.sim.y_corr)
            recording = self.rec is not None
            fresh = self.blit.stale          # background just re-cached: redraw all
            # while recording, the status line must be in every captured frame
            # (the HUD block is outside the crop, so it keeps its own schedule)
            due = {k: fresh or (recording and k == "title") or frames % per == ph
                   for k, (per, ph) in TEXT_SCHED.items()}
            if due["title"]:
                self.scene.set_status(self._status(kmh, clr, n_yield))
            if due["hud_pin"]:
                self._update_pin()
            if due["hud"]:
                self._update_block()
            if self.blit.restore():
                regions = self.scene.draw(with_title=due["title"],
                                          with_insets=fresh or recording
                                          or frames % 2 == 0)
                for key, art in (("hud_pin", self.hud_pin), ("hud", self.hud)):
                    if due[key]:
                        self.fig.draw_artist(art)
                        regions.append(self._rect_bbox(self.rects[key]))
                regions += self._draw_widgets(
                    set(self.sl) | {f"btn_{k}" for k in self.btn} if fresh
                    else self._dirty)
                self._dirty.clear()
                self.blit.stale = False
                if recording and frames and not self.rec.grab(self.fig):
                    self.toggle_record()
                self.blit.push(regions)
            canvas.flush_events()
            frames += 1
            self._loop_dt = 0.85 * self._loop_dt + 0.15 * (time.perf_counter() - t0)
            # Pace to the frame period with short sleeps (flush_events costs
            # ~9 ms per call here, so it stays out of the wait - one flush per
            # frame is enough to keep the widgets responsive).
            while True:
                rem = t0 + period - time.perf_counter()
                if rem <= 0.002:
                    break
                time.sleep(min(rem - 0.001, 0.008))

        if self.rec is not None:
            self.toggle_record()
        return dict(frames=frames, sim_steps=sim_steps,
                    sim_time=round(self.world.sim.t, 2),
                    n_veh=int(self.world.sim.st.n),
                    render_fps=round(1.0 / max(self._loop_dt, 1e-3), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="highway", choices=list(SCEN))
    ap.add_argument("--preset", default="tuned", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=None,
                    help="veh/km/lane (default: scenario value)")
    ap.add_argument("--rec-max", type=int, default=320,
                    help="max frames per GIF recording")
    ap.add_argument("--record", action="store_true",
                    help="start recording out/anim_live_lab.gif immediately")
    ap.add_argument("--duration", type=float, default=None,
                    help="close the window after N s (unattended demo/test)")
    ap.add_argument("--self-test", action="store_true",
                    help="headless: build, step, retune, resize - no window")
    args = ap.parse_args()

    if args.self_test:
        p = PRESETS[args.preset]()
        w = EndlessWorld(args.scenario, args.seed, p, args.density)
        n0 = w.sim.st.n
        for _ in range(400):
            w.step()
        p.A_c, p.a_pin = 0.5, 3.0                 # live retune mid-run
        for _ in range(200):
            w.step()
        for d in (60.0, 12.0):
            w.set_density(d)
            for _ in range(100):
                w.step()
        st = w.sim.st
        rel = st.x - st.x[st.ev]
        print(f"[self-test] ok: n {n0} -> {st.n}, t={w.sim.t:.1f}s, "
              f"band=[{rel.min():.0f},{rel.max():.0f}] m", flush=True)
        sys.exit(0)

    with RunLogger("live_lab", note=f"{args.scenario} preset={args.preset} "
                                    f"seed={args.seed}") as rl:
        lab = LiveLab(scenario=args.scenario, preset=args.preset, seed=args.seed,
                      fps=args.fps, density=args.density, rec_max=args.rec_max)
        if args.record:
            lab.toggle_record()
        print("[lab] window open - drag the sliders while it runs "
              "(space=pause, r=reset, g=record, q=quit)", flush=True)
        m = lab.run(duration=args.duration)
        rl.finish(metrics=m, outputs=lab.outputs)
        print(f"[lab] closed: {m}", flush=True)
