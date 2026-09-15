"""Vehicle state arrays (structure-of-arrays for vectorised force evaluation)."""
from dataclasses import dataclass, field
import numpy as np

# awareness state machine (emv.perception)
UNAWARE, NOTICED, YIELDING, HOLD = 0, 1, 2, 3
STATE_NAMES = {UNAWARE: "unaware", NOTICED: "noticed", YIELDING: "yielding", HOLD: "hold"}


@dataclass
class VehState:
    # kinematics
    x: np.ndarray            # m
    y: np.ndarray            # m
    vx: np.ndarray           # m/s
    vy: np.ndarray           # m/s
    # geometry
    L: np.ndarray            # m, vehicle length
    W: np.ndarray            # m, vehicle width
    # driver parameters (heterogeneous)
    v0: np.ndarray           # m/s, desired speed
    tau: np.ndarray          # s, drive-force relaxation time
    T_hw: np.ndarray         # s, IDM time headway
    s0: np.ndarray           # m, IDM standstill gap
    amax: np.ndarray         # m/s^2
    bcomf: np.ndarray        # m/s^2
    # roles / awareness
    is_ev: np.ndarray        # bool
    comply: np.ndarray       # bool, driver will yield when aware
    aware: np.ndarray        # int8, UNAWARE/NOTICED/YIELDING/HOLD
    side: np.ndarray         # int8, chosen escape side (-1 right, +1 left, 0 none)
    t_react: np.ndarray      # s, time at which NOTICED -> YIELDING
    hold_until: np.ndarray   # s, time at which HOLD -> UNAWARE
    delay: np.ndarray        # s, sampled per-driver reaction delay
    u: np.ndarray            # -, urgency in [0, 1] (computed each step)
    y_init: np.ndarray = field(default=None)  # m, pre-EV lateral position
    y_pref: np.ndarray = field(default=None)  # m, per-driver lateral offset from
    #                                          the lane centre (params.sigma_off;
    #                                          zeros = pin to the centre exactly)
    # Optional per-driver lateral dynamics. None => every driver uses the global
    # Params value, which is the behaviour of record. Populated only by
    # scenarios.make_highd_like when params.het_lat > 0, so that driver-to-driver
    # variety in manoeuvre execution can be represented (see PROJECT_LOG sec. 5).
    a_pin: np.ndarray = field(default=None)      # m/s^2
    zeta_lat: np.ndarray = field(default=None)   # -
    v_lat_max: np.ndarray = field(default=None)  # m/s
    a_lat_max: np.ndarray = field(default=None)  # m/s^2
    # Discretionary lane changing (emv/lanechange.py, added 2026-08-06). Only
    # touched when params.A_pass or A_keep_right is nonzero, so the default
    # all-inert values keep every published result unchanged.
    lc_target: np.ndarray = field(default=None)    # int, committed lane (-1 none)
    lc_until: np.ndarray = field(default=None)     # s, commitment timeout
    lc_free_at: np.ndarray = field(default=None)   # s, earliest next commitment

    def __post_init__(self):
        if self.y_init is None:
            self.y_init = self.y.copy()
        if self.y_pref is None:
            self.y_pref = np.zeros_like(self.y)
        if self.lc_target is None:
            self.lc_target = np.full(self.x.size, -1, dtype=int)
        if self.lc_until is None:
            self.lc_until = np.zeros_like(self.y)
        if self.lc_free_at is None:
            self.lc_free_at = np.zeros_like(self.y)

    @property
    def n(self) -> int:
        return self.x.size

    @property
    def ev(self) -> int:
        return int(np.argmax(self.is_ev))

    def ev_heading(self) -> np.ndarray:
        """EV motion direction (falls back to +x when nearly stopped)."""
        i = self.ev
        v = np.array([self.vx[i], self.vy[i]])
        s = np.hypot(*v)
        return v / s if s > 1.0 else np.array([1.0, 0.0])


def blank_state(n: int) -> dict:
    """Array template used by scenario builders."""
    f = lambda v=0.0: np.full(n, v, dtype=float)
    return dict(
        x=f(), y=f(), vx=f(), vy=f(),
        L=f(4.5), W=f(1.8),
        v0=f(28.0), tau=f(0.55), T_hw=f(1.4), s0=f(2.0), amax=f(2.6), bcomf=f(3.0),
        is_ev=np.zeros(n, dtype=bool), comply=np.ones(n, dtype=bool),
        aware=np.zeros(n, dtype=np.int8), side=np.zeros(n, dtype=np.int8),
        t_react=f(np.inf), hold_until=f(np.inf), delay=f(1.0), u=f(0.0),
        y_pref=f(0.0),      # zeros = pin to the lane centre (params.sigma_off)
    )
