"""Simulation driver and trajectory recording."""
from dataclasses import dataclass, field
import numpy as np

from . import dynamics
from . import forces
from .state import VehState
from .road import Road, build_corridor
from .perception import Perception
from .lanechange import LaneChange
from .params import Params


@dataclass
class History:
    """Recorded trajectories (K frames x N vehicles) plus static metadata."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    ax: np.ndarray
    aware: np.ndarray
    u: np.ndarray
    side: np.ndarray
    # static
    L: np.ndarray
    W: np.ndarray
    v0: np.ndarray
    ev: int
    y_corr: float
    road: Road
    params: Params
    name: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return self.t.size

    def save(self, path):
        np.savez_compressed(
            path, t=self.t, x=self.x, y=self.y, vx=self.vx, vy=self.vy,
            ax=self.ax, aware=self.aware, u=self.u, side=self.side,
            L=self.L, W=self.W, v0=self.v0,
            ev=self.ev, y_corr=self.y_corr, name=self.name,
            road=np.array([self.road.n_lanes, self.road.lane_width,
                           self.road.shoulder_right, self.road.shoulder_left,
                           self.road.length]))


class Sim:
    """Owns the state, perception and corridor; runs the fixed-step loop."""

    def __init__(self, st: VehState, road: Road, p: Params, y_corr: float,
                 seed: int = 0, yielding: bool = True, name: str = ""):
        self.st, self.road, self.p = st, road, p
        self.y_corr = y_corr
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.per = Perception(st, p, road, self.rng, enabled=yielding)
        # Discretionary lane changing (2026-08-06). Inert unless params.A_pass or
        # A_keep_right is nonzero, so no existing scenario changes behaviour and
        # no RNG is drawn here (the constructor takes no generator on purpose).
        self.lc = LaneChange(st, p, road)
        self.t = 0.0

    def _corridor(self):
        i = self.st.ev
        return build_corridor(self.st.x[i] + 0.5 * self.st.L[i], self.st.vx[i],
                              self.y_corr, self.st.v0[i], self.p)

    def step(self, diag: bool = False):
        corr = self._corridor()
        self.per.update(self.st, corr, self.t)
        if self.lc.enabled:
            self.lc.update(self.st, self.t,
                           rho_fac=forces.pin_density_factor(self.st, self.road,
                                                             self.p))
        out = dynamics.step(self.st, self.road, self.p, corr, self.p.dt, diag=diag)
        self.t += self.p.dt
        return out

    def snapshot_forces(self):
        """Per-term force diagnostic at the current state (for viz)."""
        corr = self._corridor()
        self.per.update(self.st, corr, self.t)
        from . import forces
        _, _, dg = forces.total(self.st, self.road, self.p, corr, diag=True)
        return corr, dg

    def run(self, T: float, rec_dt: float = 0.1,
            stop_when_ev_x: float | None = None, on_diag=None) -> History:
        """Fixed-step loop, optionally recording per-term force diagnostics.

        `on_diag(dg, st, dt)` is called after every step with the per-term
        diagnostic dict from `forces.total(diag=True)`. Opt-in because it forces
        the diagnostic branch and adds a bookkeeping pass per step; with
        `on_diag=None` this is the loop of record, unchanged
        (`emv/terms.py:budget_run` is the intended caller).
        """
        st, p = self.st, self.p
        every = max(1, int(round(rec_dt / p.dt)))
        n_steps = int(round(T / p.dt))
        rec = {k: [] for k in ("t", "x", "y", "vx", "vy", "ax", "aware", "u", "side")}

        ax = np.zeros(st.n)
        for k in range(n_steps):
            if k % every == 0:
                rec["t"].append(self.t)
                rec["x"].append(st.x.copy());   rec["y"].append(st.y.copy())
                rec["vx"].append(st.vx.copy()); rec["vy"].append(st.vy.copy())
                rec["ax"].append(ax.copy())
                rec["aware"].append(st.aware.copy())
                rec["u"].append(st.u.astype(np.float32))
                rec["side"].append(st.side.copy())
            if on_diag is None:
                ax, _, _ = self.step()
            else:
                ax, _, dg = self.step(diag=True)
                on_diag(dg, st, p.dt)
            if stop_when_ev_x is not None and st.x[st.ev] > stop_when_ev_x:
                break

        return History(
            t=np.array(rec["t"]),
            x=np.array(rec["x"]), y=np.array(rec["y"]),
            vx=np.array(rec["vx"]), vy=np.array(rec["vy"]), ax=np.array(rec["ax"]),
            aware=np.array(rec["aware"]), u=np.array(rec["u"]),
            side=np.array(rec["side"]),
            L=st.L.copy(), W=st.W.copy(), v0=st.v0.copy(), ev=st.ev,
            y_corr=self.y_corr, road=self.road, params=p, name=self.name)
