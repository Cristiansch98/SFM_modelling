"""Real-time top-down 2D window for the UE game session.

Same look as `out/anim_surrogate_sumo.gif` (emv.viz.animate), but driven by the
live brain state instead of a recorded History: the camera follows the EV that
the human is driving in Unreal, NPCs are tinted by perception state, and the two
insets scroll a rolling window of EV/traffic speed and corridor clearance.

The viewer runs in a **separate process** by default. Matplotlib in the brain's
own process holds the GIL long enough to wreck real-time pacing - measured at
16.7 Hz it cost ~1 %, but at 33 Hz it dropped the brain to 72 % of real time and
at 50 Hz to 66 %. Out of process, the brain keeps its full tick budget and only
pays a few microseconds per step to copy the state into shared memory.

`--view-inproc` keeps the old single-process behaviour (handy when debugging the
viewer itself, since exceptions surface directly).

NOTE: this renderer rebuilds the scene every frame (ax.clear + fresh patches +
fresh legend), which costs ~250 ms/frame - fine for a 12 fps observer window,
too slow for an interactive one. experiments/live_lab.py keeps the same visual
language but updates persistent artists instead; see _Scene there.
"""
import multiprocessing as mp
import threading
import time
from collections import deque

import numpy as np

from ..state import UNAWARE, NOTICED, YIELDING, HOLD, STATE_NAMES

TRACE_S = 60.0                          # seconds of history in the insets
# scalar slots in the shared 'scal' array
S_T, S_LIVE, S_STOP, S_YCORR = 0, 1, 2, 3


# =========================================================== shared state
def _alloc(n: int) -> dict:
    """Shared blocks the brain writes and the viewer reads. lock=False: a torn
    read costs at worst one slightly stale car, which is invisible at 12 fps."""
    return dict(
        n=n,
        x=mp.Array("d", n, lock=False),
        y=mp.Array("d", n, lock=False),
        vx=mp.Array("d", n, lock=False),
        aware=mp.Array("b", n, lock=False),
        scal=mp.Array("d", 8, lock=False),
    )


def _publish(sh: dict, bridge) -> None:
    """Called from the serve loop each step. Must stay cheap."""
    st = bridge.sim.st
    n = sh["n"]
    np.frombuffer(sh["x"], dtype=np.float64)[:n] = st.x[:n]
    np.frombuffer(sh["y"], dtype=np.float64)[:n] = st.y[:n]
    np.frombuffer(sh["vx"], dtype=np.float64)[:n] = st.vx[:n]
    np.frombuffer(sh["aware"], dtype=np.int8)[:n] = st.aware[:n]
    sc = np.frombuffer(sh["scal"], dtype=np.float64)
    sc[S_T] = bridge.sim.t
    sc[S_LIVE] = 1.0 if getattr(bridge, "live", False) else 0.0
    sc[S_YCORR] = bridge.sim.y_corr


def _static_of(bridge) -> dict:
    """Everything the viewer needs that never changes (picklable)."""
    st, p, road = bridge.sim.st, bridge.p, bridge.sim.road
    return dict(
        n=int(st.n), ev=int(st.ev),
        L=st.L.tolist(), W=st.W.tolist(),
        n_lanes=road.n_lanes, lane_width=road.lane_width,
        shoulder_right=road.shoulder_right, shoulder_left=road.shoulder_left,
        margin_c=p.margin_c, L_pred_max=p.L_pred_max,
    )


# ================================================================ drawing
class LiveView:
    """Renders one frame from a plain state dict - no bridge reference, so the
    same code serves the in-process and subprocess modes."""

    def __init__(self, static: dict, fps=12.0, span=(90.0, 200.0)):
        self.s = static
        self.fps = float(fps)
        self.span = span
        self.tr = {k: deque() for k in ("t", "ev", "bg", "clr")}
        self._frame = 0
        self.L = np.asarray(static["L"])
        self.W = np.asarray(static["W"])
        self.ev = static["ev"]

    # ------------------------------------------------------------------ setup
    def build(self):
        from .. import viz                  # NOTE: viz forces Agg on import ...
        import matplotlib
        for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)   # ... so switch back, after
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from ..road import Road

        self.viz, self.plt, self.Rectangle = viz, plt, Rectangle
        self.pal = viz.DARK
        s = self.s
        self.road = Road(n_lanes=s["n_lanes"], lane_width=s["lane_width"],
                         shoulder_right=s["shoulder_right"],
                         shoulder_left=s["shoulder_left"])
        pal = self.pal

        self.fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
        try:
            self.fig.canvas.manager.set_window_title(
                "EMV brain - live 2D view (NPCs = force model, EV = you in UE)")
        except Exception:
            pass
        self.fig.patch.set_facecolor(pal["page"])
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.9, 1.15],
                                   hspace=0.34, wspace=0.22,
                                   left=0.055, right=0.985, top=0.9, bottom=0.11)
        self.ax = self.fig.add_subplot(gs[0, :])
        self.ax_v = self.fig.add_subplot(gs[1, 0])
        self.ax_c = self.fig.add_subplot(gs[1, 1])
        for a in (self.ax_v, self.ax_c):
            viz._style_ax(a, pal)
        self.ax_v.set_ylabel("speed (km/h)"); self.ax_v.set_xlabel("t (s)")
        self.ax_c.set_ylabel("clear corridor ahead (m)"); self.ax_c.set_xlabel("t (s)")
        self.l_ev, = self.ax_v.plot([], [], color=viz.EV_DARK, lw=2.0, label="EV (you)")
        self.l_bg, = self.ax_v.plot([], [], color=pal["muted"], lw=1.6, ls="--",
                                    label="traffic mean")
        self.ax_v.legend(loc="lower right", fontsize=8, frameon=False,
                         labelcolor=pal["ink2"])
        self.l_clr, = self.ax_c.plot([], [], color=viz.CAT_DARK["blue"], lw=2.0)
        self.legend_handles = [
            plt.Line2D([], [], marker="s", ls="none", ms=8, mec="none",
                       mfc=viz.STATE_FILL_DARK[st], label=STATE_NAMES[st])
            for st in (UNAWARE, NOTICED, YIELDING, HOLD)
        ] + [plt.Line2D([], [], marker="s", ls="none", ms=8, mec="none",
                        mfc=viz.EV_DARK, label="EV (you)")]
        self.fig.show()

    # ------------------------------------------------------------------- draw
    def draw(self, x, y, vx, aware, t, live, y_corr):
        viz, plt, Rectangle = self.viz, self.plt, self.Rectangle
        pal, e = self.pal, self.ev
        x_ev, v_ev = float(x[e]), float(vx[e])
        others = np.ones(x.size, bool); others[e] = False

        near = others & (np.abs(x - x_ev) < 300.0)
        bg = 3.6 * float(vx[near].mean()) if near.any() else 0.0
        w_need = 0.5 * (self.W[e] + self.W) + self.s["margin_c"]
        dx = x - x_ev - 0.5 * (self.L[e] + self.L)
        blocking = (np.abs(y - y_corr) < w_need) & (dx > 0.0) & others
        clr = min(float(dx[blocking].min()), 250.0) if blocking.any() else 250.0

        for k, v in (("t", t), ("ev", 3.6 * v_ev), ("bg", bg), ("clr", clr)):
            self.tr[k].append(v)
        while self.tr["t"] and self.tr["t"][-1] - self.tr["t"][0] > TRACE_S:
            for d in self.tr.values():
                d.popleft()

        ax = self.ax
        ax.clear()
        ax.set_facecolor(pal["page"])
        x0, x1 = x_ev - self.span[0], x_ev + self.span[1]
        viz._draw_road(ax, self.road, x0, x1, pal)

        L_c = min(240.0, self.s["L_pred_max"])
        w_c = 0.5 * self.W[e] + self.s["margin_c"]
        ax.add_patch(Rectangle((x_ev, y_corr - w_c), L_c, 2 * w_c,
                               fc=viz.EV_DARK, alpha=0.13, ec="none", zorder=1.5))
        ax.plot([x_ev, x_ev + L_c], [y_corr] * 2, color=viz.EV_DARK, lw=1.0,
                ls=(0, (5, 6)), alpha=0.55, zorder=1.6)

        vis = np.flatnonzero((x > x0 - 8) & (x < x1 + 8) & others)
        for i in vis:
            ax.add_patch(viz._car_patch(x[i], y[i], self.L[i], self.W[i],
                                        viz.STATE_FILL_DARK[int(aware[i])]))
        blink = (self._frame % 2 == 0)
        ax.add_patch(viz._car_patch(x_ev, y[e], self.L[e] + 2.6, self.W[e] + 2.6,
                                    viz.EV_DARK, alpha=0.28 if blink else 0.16, z=2.5))
        ax.add_patch(viz._car_patch(x_ev, y[e], self.L[e], self.W[e], viz.EV_DARK, z=4))
        bx = x_ev + (0.9 if blink else -0.9)
        ax.add_patch(Rectangle((bx - 0.55, y[e] - 0.38), 1.1, 0.76,
                               fc="#ffffff" if blink else "#9ec5f4", ec="none",
                               zorder=5))

        ax.set_xlim(x0, x1)
        ax.set_ylim(self.road.y_min - 1.6, self.road.y_max + 1.6)
        ax.set_aspect(2.6)
        ax.set_yticks([]); ax.tick_params(colors=pal["muted"], labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        n_yield = int(((aware == YIELDING) | (aware == HOLD))[others].sum())
        link = "connected" if live else "waiting for UE"
        ax.set_title(f"t = {t:6.1f} s    EV {3.6 * v_ev:4.0f} km/h    "
                     f"clear {clr:3.0f} m    yielding {n_yield:3d}    [{link}]",
                     color=pal["ink"], fontsize=11, family="monospace", loc="left")
        ax.legend(handles=self.legend_handles, loc="upper right", ncol=5,
                  fontsize=8, frameon=False, labelcolor=pal["ink2"],
                  bbox_to_anchor=(1.0, 1.16))

        tt = list(self.tr["t"])
        self.l_ev.set_data(tt, list(self.tr["ev"]))
        self.l_bg.set_data(tt, list(self.tr["bg"]))
        self.l_clr.set_data(tt, list(self.tr["clr"]))
        if tt:
            self.ax_v.set_xlim(tt[0], max(tt[-1], tt[0] + 5.0))
            self.ax_c.set_xlim(tt[0], max(tt[-1], tt[0] + 5.0))
            self.ax_v.set_ylim(0, max(140.0, max(self.tr["ev"]) * 1.1))
            self.ax_c.set_ylim(0, 260)
        self._frame += 1


# ====================================================== viewer subprocess
def _viewer_main(sh: dict, static: dict, fps: float) -> None:
    view = LiveView(static, fps=fps)
    view.build()
    plt = view.plt
    n = static["n"]
    xb = np.frombuffer(sh["x"], dtype=np.float64)
    yb = np.frombuffer(sh["y"], dtype=np.float64)
    vb = np.frombuffer(sh["vx"], dtype=np.float64)
    ab = np.frombuffer(sh["aware"], dtype=np.int8)
    sc = np.frombuffer(sh["scal"], dtype=np.float64)
    period = 1.0 / fps
    try:
        while sc[S_STOP] == 0.0:
            if not plt.fignum_exists(view.fig.number):
                break
            view.draw(xb[:n].copy(), yb[:n].copy(), vb[:n].copy(), ab[:n].copy(),
                      float(sc[S_T]), sc[S_LIVE] > 0.5, float(sc[S_YCORR]))
            plt.pause(period)
    except (KeyboardInterrupt, Exception):
        pass


# ================================================================= drivers
def serve_with_view(bridge, duration=None, fps=12.0, on_listening=None,
                    in_process=False) -> dict:
    """Serve the UE session with the live 2D window up.

    Default: the brain owns the main thread and the viewer is a child process.
    `in_process=True` restores the old layout (viewer on the main thread, brain
    on a worker) - simpler to debug, but it costs real-time pacing.
    """
    if in_process:
        return _serve_view_inproc(bridge, duration, fps, on_listening)

    bridge._build()                       # need st.n / road before allocating
    sh = _alloc(int(bridge.sim.st.n))
    _publish(sh, bridge)
    proc = mp.Process(target=_viewer_main, args=(sh, _static_of(bridge), fps),
                      name="emv-liveview", daemon=True)
    proc.start()
    print(f"[ue] live 2D view in pid {proc.pid} (out of process)", flush=True)

    sc = np.frombuffer(sh["scal"], dtype=np.float64)
    try:
        return bridge.serve(duration=duration, on_listening=on_listening,
                            on_step=lambda b: _publish(sh, b))
    finally:
        sc[S_STOP] = 1.0
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()


def _serve_view_inproc(bridge, duration, fps, on_listening) -> dict:
    out = {}

    def worker():
        try:
            out["result"] = bridge.serve(duration=duration,
                                         on_listening=on_listening)
        except BaseException as exc:                  # surface, don't swallow
            out["error"] = exc
        finally:
            out["done"] = True

    th = threading.Thread(target=worker, name="ue-brain", daemon=True)
    th.start()
    while bridge.sim is None and not out.get("done"):
        time.sleep(0.05)

    view = LiveView(_static_of(bridge), fps=fps)
    view.build()
    plt = view.plt
    period = 1.0 / fps
    try:
        while not out.get("done"):
            if not plt.fignum_exists(view.fig.number):
                # Closing the window must NOT take the brain down with it.
                print("[ue] live view closed - brain keeps serving headless "
                      "(Ctrl-C here to stop)", flush=True)
                break
            st = bridge.sim.st
            view.draw(st.x.copy(), st.y.copy(), st.vx.copy(), st.aware.copy(),
                      float(bridge.sim.t), getattr(bridge, "live", False),
                      float(bridge.sim.y_corr))
            plt.pause(period)
        while not out.get("done"):          # headless tail of the session
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[ue] interrupted", flush=True)

    th.join(timeout=5.0)
    if "error" in out:
        raise out["error"]
    return out.get("result", {})
