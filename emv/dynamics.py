"""Time stepping: force composition, dynamic constraints, semi-implicit Euler.

Semi-implicit (symplectic) Euler - update velocity first, then position with
the *new* velocity - is stable for the stiff short-range potentials at
dt = 0.05 s and is the scheme recommended for direct force integration
(docs/FRAMEWORK.md sec. 5; cf. stability discussion in [10]).

Constraint projection (a poor man's non-holonomic model): the raw force is
clamped component-wise in the road frame -
  longitudinal a in [-b_emerg, amax * boost], v_x >= 0 (no reversing)
  lateral      |a| <= a_lat_max, |v_y| <= v_lat_max (steering-rate proxy)
- which keeps trajectories within vehicle-dynamics envelopes without a full
bicycle model [6].
"""
import numpy as np

from . import forces
from .state import VehState, YIELDING
from .road import Road, Corridor


def step(st: VehState, road: Road, p, corr: Corridor, dt: float, diag: bool = False):
    ax, ay, dg = forces.total(st, road, p, corr, diag=diag)

    # --- constraint projection (road frame) ---------------------------
    boost = np.where(st.aware == YIELDING, p.aware_amax_boost, 1.0)
    amax = np.where(st.is_ev, p.ev_amax, st.amax * boost)
    ax = np.clip(ax, -p.b_emerg, amax)
    a_lat_cap = p.a_lat_max if st.a_lat_max is None else st.a_lat_max
    ay = np.clip(ay, -a_lat_cap, a_lat_cap)

    # --- semi-implicit Euler -------------------------------------------
    st.vx += ax * dt
    st.vy += ay * dt
    np.maximum(st.vx, 0.0, out=st.vx)
    v_lat_cap = p.v_lat_max if st.v_lat_max is None else st.v_lat_max
    np.clip(st.vy, -v_lat_cap, v_lat_cap, out=st.vy)
    st.x += st.vx * dt
    st.y += st.vy * dt

    # hard backstop at the drivable-band edges
    lo = road.y_min + 0.5 * st.W + 0.05
    hi = road.y_max - 0.5 * st.W - 0.05
    below, above = st.y < lo, st.y > hi
    st.y[below], st.y[above] = lo[below], hi[above]
    st.vy[below | above] = 0.0
    return ax, ay, dg
