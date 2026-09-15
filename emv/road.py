"""Road geometry and the EV's predicted-path corridor.

The demonstration scenarios use a straight multi-lane carriageway along +x
with right-hand traffic: lane 0 (rightmost) at low y, lane centres at
y_k = (k + 1/2) * lane_width. Shoulders extend the drivable band beyond the
marked lanes (used for squeezing during rescue-lane formation).

Corridor: the EV's route projected T_pred seconds forward (gap 1 of the
literature review). On a straight road this is the axis-aligned segment
[x_ev - L_back, x_ev + L_pred] at lateral position y_c; the general
polyline case is discussed in docs/FRAMEWORK.md sec. 3.3.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Road:
    n_lanes: int = 3
    lane_width: float = 3.5       # m
    shoulder_right: float = 2.0   # m, drivable hard shoulder at low y
    shoulder_left: float = 0.8    # m, drivable strip at high y
    length: float = 2200.0        # m (visual/scenario extent; x is unbounded)

    @property
    def width_lanes(self) -> float:
        return self.n_lanes * self.lane_width

    @property
    def y_min(self) -> float:
        return -self.shoulder_right

    @property
    def y_max(self) -> float:
        return self.width_lanes + self.shoulder_left

    def lane_center(self, k) -> np.ndarray | float:
        return (np.asarray(k) + 0.5) * self.lane_width

    def nearest_lane_center(self, y):
        k = np.clip(np.round(np.asarray(y) / self.lane_width - 0.5), 0, self.n_lanes - 1)
        return (k + 0.5) * self.lane_width


@dataclass
class Corridor:
    """EV predicted-path corridor, rebuilt every step around the EV nose."""
    x0: float          # m, EV front-bumper x
    y_c: float         # m, lateral position of the predicted path
    L: float           # m, forward extent (v_pred * T_pred, clamped)
    L_back: float      # m, backward extent behind the EV nose

    def rel(self, x, y):
        """Return (s, d): longitudinal distance ahead of the EV nose and
        signed perpendicular offset from the corridor line."""
        return np.asarray(x) - self.x0, np.asarray(y) - self.y_c


def build_corridor(x_ev: float, v_ev: float, y_c: float, v0_ev: float, p) -> Corridor:
    """Corridor length reflects *intent*: even a blocked EV projects a
    corridor, because drivers respond to where it is trying to go [20]."""
    v_pred = max(v_ev, p.v_app_floor * v0_ev)
    L = float(np.clip(v_pred * p.T_pred, p.L_pred_min, p.L_pred_max))
    return Corridor(x0=x_ev, y_c=y_c, L=L, L_back=p.L_back)
