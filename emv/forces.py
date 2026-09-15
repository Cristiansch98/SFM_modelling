"""Force terms of the EV-yielding social-force model.

All forces are specific forces (accelerations, m = 1), returned as (N,)
component arrays (ax, ay). Reference numbers [n] refer to
literature_review_ev_repulsion_models.md; section numbers refer to
docs/FRAMEWORK.md.

Composition (dynamics.py):
    ax = ax_drive + min( ax_sfm + ax_evfield + ax_corridor ,  a_IDM )
    ay = ay_drive + ay_lane + ay_incentive + ay_sfm + ay_evfield + ay_corridor
The `min` makes the IDM car-following demand an *upper bound* on the
longitudinal interaction acceleration: safety always wins over the social
terms and mild braking terms do not stack (sec. 2.6).

`ay_incentive` (`lane_incentive`, added 2026-08-06) is the only *discretionary*
term: it makes a vehicle change lane with no emergency vehicle anywhere in the
scenario. It is off by default (`A_pass = A_keep_right = 0`), which is the
behaviour of record for papers 1-6.
"""
import numpy as np

from .state import VehState, YIELDING, HOLD
from .road import Road, Corridor

BIG = 1e9  # 'no constraint' sentinel for the IDM bound


# ----------------------------------------------------------------------
# 1. driving force [2]
# ----------------------------------------------------------------------
def drive(st: VehState):
    """F_drive = (v_desired - v)/tau, desired velocity = v0 * x_hat."""
    ax = (st.v0 - st.vx) / st.tau
    ay = -st.vy / st.tau
    return ax, ay


# ----------------------------------------------------------------------
# 2. lane keeping: washboard potential + road-edge walls (sec. 4)
# ----------------------------------------------------------------------
def lane_index(st: VehState, road: Road):
    """Lane each vehicle currently occupies (0 = rightmost). Same expression as
    `metrics.behaviour_pools`, so model-side lane bookkeeping is consistent."""
    return np.clip(np.round(st.y / road.lane_width - 0.5), 0,
                   road.n_lanes - 1).astype(int)


def local_density(st: VehState, road: Road, p, lane=None):
    """Per-vehicle local density in veh/km/lane (own lane, +/- p.R_rho).

    Counted in the vehicle's own lane rather than across the carriageway because
    lane discipline is a lane-local phenomenon: a crowded rightmost lane does not
    stiffen a driver alone in the leftmost one. The window is symmetric, so a
    vehicle counts itself - which makes the estimate ~1/(2 R_rho) at minimum and
    keeps it strictly positive, as a power-law modulation requires.
    """
    lane = lane_index(st, road) if lane is None else lane
    same = lane[None, :] == lane[:, None]
    near = np.abs(st.x[None, :] - st.x[:, None]) <= p.R_rho
    n = (same & near & ~st.is_ev[None, :]).sum(axis=1)
    return n / (2.0 * p.R_rho / 1000.0)


def pin_density_factor(st: VehState, road: Road, p, rho=None):
    """(rho/rho_ref)^k_rho - the density stiffening of lane discipline.

    Returns exactly 1.0 when k_rho == 0, which is the behaviour of record: the
    factor is then not even computed, so `local_density`'s O(N^2) pass costs
    nothing in runs that do not use it.
    """
    if p.k_rho == 0.0:
        return 1.0
    rho = local_density(st, road, p) if rho is None else rho
    return (np.maximum(rho, 1e-6) / p.rho_ref) ** p.k_rho


def lane_keep(st: VehState, road: Road, p, corr: Corridor, rho_fac=None):
    """Periodic lane potential U(y) = -a_pin (w/2pi) cos(2pi (y - w/2)/w):
    minima at lane centres, maxima at lane markings. External lateral force
    must exceed a_pin to depin a car from its lane. Urgency relaxes lane
    discipline (panic modulation, [3]). The EV instead tracks the corridor
    line with a critically damped spring [10].

    Each driver's minimum is shifted by `st.y_pref` (params.sigma_off), because
    real drivers hold a personal offset from the centre rather than the centre
    itself - 0.28 m on average in highD. y_pref is all-zeros unless a scenario
    draws it, which keeps this identical to the published behaviour.

    `rho_fac` is the density stiffening (see `pin_density_factor`); it scales the
    pinning amplitude only, so the potential keeps its shape and the analytic
    depinning results carry over with `a_pin_eff` in place of `a_pin`. The
    supplementary damping deliberately uses the *unstiffened* a_pin so that
    `zeta_lat` keeps meaning "damping ratio at the reference density" and does
    not silently co-vary with k_rho during the fit.
    """
    w = road.lane_width
    a_pin_base = p.a_pin if st.a_pin is None else st.a_pin
    zeta_v = p.zeta_lat if st.zeta_lat is None else st.zeta_lat
    a_pin_v = a_pin_base if rho_fac is None else a_pin_base * rho_fac
    a_pin_eff = a_pin_v * (1.0 - p.urgency_pin_relief * st.u)
    inside = (st.y >= 0.0) & (st.y <= road.width_lanes)
    ay = np.where(inside, -a_pin_eff * np.sin(
        2.0 * np.pi * (st.y - st.y_pref - 0.5 * w) / w), 0.0)

    # road-edge walls (drivable band includes shoulders)
    edge_lo = (st.y - 0.5 * st.W) - road.y_min
    edge_hi = road.y_max - (st.y + 0.5 * st.W)
    ay += p.a_wall * (np.exp(-np.maximum(edge_lo, 0.0) / p.B_wall)
                      - np.exp(-np.maximum(edge_hi, 0.0) / p.B_wall))

    # supplementary lateral damping to reach damping ratio zeta_lat
    # (linearised pinning frequency omega^2 = 2 pi a_pin / w; drive() already
    #  contributes 1/tau of damping)
    omega = np.sqrt(2.0 * np.pi * a_pin_base / w)
    c_extra = np.maximum(2.0 * zeta_v * omega - 1.0 / st.tau, 0.0)
    ay += -c_extra * st.vy

    # EV: critically damped spring to the corridor line (no washboard),
    # weaving around the nearest straggler that blocks the line [18]
    i = st.ev
    om = p.ev_omega_lat
    y_t = _ev_lat_target(st, p, corr)
    ay_ev = om * om * (y_t - st.y[i]) - 2.0 * p.ev_zeta_lat * om * st.vy[i]
    ay[i] = ay_ev + p.a_wall * (np.exp(-max(edge_lo[i], 0.0) / p.B_wall)
                                - np.exp(-max(edge_hi[i], 0.0) / p.B_wall))
    return ay


def _ev_lat_target(st: VehState, p, corr: Corridor) -> float:
    """EV lateral target: the corridor line, offset to clear the nearest
    vehicle still straddling the line (real EV drivers thread around
    stragglers rather than waiting behind them [18])."""
    i = st.ev
    dx = st.x - st.x[i]
    half_w = 0.5 * (st.W + st.W[i])
    blk = ((dx > 0.0) & (dx < p.ev_weave_lookahead) & ~st.is_ev
           & (np.abs(st.y - corr.y_c) < half_w + p.ev_weave_margin))
    if not blk.any():
        return corr.y_c
    j = np.flatnonzero(blk)[np.argmin(dx[blk])]
    esc = 1.0 if corr.y_c >= st.y[j] else -1.0
    y_t = st.y[j] + esc * (half_w[j] + p.ev_weave_margin)
    return float(np.clip(y_t, corr.y_c - p.ev_weave_max, corr.y_c + p.ev_weave_max))


# ----------------------------------------------------------------------
# 3. car-car interaction: elliptic social force + IDM bound [2, 4, 11, 16]
# ----------------------------------------------------------------------
def car_car(st: VehState, p, want_leader: bool = False):
    """Returns (ax_sfm, ay_sfm, a_idm) - and (.., leader) if `want_leader`.

    * Elliptic anisotropic exponential repulsion between all pairs
      (elongated equipotentials ~ vehicle footprint, [11, 16]), including a
      short-range 'body' term that outmuscles the corridor force so cars
      compress laterally without overlapping [3].
    * a_idm: most restrictive IDM interaction demand from any vehicle ahead
      with (smooth) lateral overlap - the longitudinal safety layer [4].
    * leader (optional): index of that most restrictive leader, or -1 if the
      vehicle is unconstrained. Returned rather than recomputed so that
      `lane_incentive` is frustrated by *the same* vehicle the safety layer is
      constrained by - the incentive and the brake can never disagree about who
      is in front. `want_leader=False` keeps the original 3-tuple, so every
      existing caller is untouched.
    """
    n = st.n
    dx = st.x[None, :] - st.x[:, None]     # dx[i, j] = x_j - x_i
    dy = st.y[None, :] - st.y[:, None]
    act = (np.abs(dx) < p.pair_cutoff) & ~np.eye(n, dtype=bool)

    half_L = 0.5 * (st.L[None, :] + st.L[:, None])
    half_W = 0.5 * (st.W[None, :] + st.W[:, None])

    # --- elliptic social repulsion -----------------------------------
    gx = np.maximum(np.abs(dx) - half_L, 0.05)     # bumper gaps, floored
    gy = np.maximum(np.abs(dy) - half_W, 0.05)
    q = np.sqrt((gx / p.Bx_v) ** 2 + (gy / p.By_v) ** 2)
    mag = p.A_v * np.exp(-q) + p.A_near * np.exp(-q / p.q_near)

    # anisotropy of the *receiver*: stimuli ahead of i count fully [2]
    d = np.sqrt(dx * dx + dy * dy) + 1e-9
    cphi = dx / d                                   # heading ~ +x on a highway
    wgt = p.lam_v + (1.0 - p.lam_v) * 0.5 * (1.0 + cphi)

    # direction: gradient of the elliptic metric, pointing away from j
    ux = -np.sign(dx) * gx / p.Bx_v ** 2
    uy = -np.sign(dy) * gy / p.By_v ** 2
    un = np.sqrt(ux * ux + uy * uy) + 1e-12
    scale = np.where(act, mag * wgt, 0.0) / un
    fx = (scale * ux).sum(axis=1)
    fy = (scale * uy).sum(axis=1)

    # cap the summed social force; EV receives it discounted (assertive)
    fn = np.hypot(fx, fy)
    shrink = np.minimum(1.0, p.F_cap_v / np.maximum(fn, 1e-9))
    fx, fy = fx * shrink, fy * shrink
    fx[st.ev] *= p.ev_sfm_receive
    fy[st.ev] *= p.ev_sfm_receive

    # --- IDM interaction bound ----------------------------------------
    ahead = act & (dx > 0.1)
    lat_gap = np.maximum(np.abs(dy) - half_W, 0.0)
    margin = np.where(st.is_ev, p.ev_overlap_margin, p.idm_overlap_margin)[:, None]
    overlap = np.clip((margin - lat_gap) / margin, 0.0, 1.0)
    cand = ahead & (overlap > 0.0)

    gap = dx - half_L                                # bumper-to-bumper
    dv = st.vx[:, None] - st.vx[None, :]             # closing speed > 0
    s_star = (st.s0[:, None]
              + np.maximum(st.vx[:, None] * st.T_hw[:, None]
                           + st.vx[:, None] * dv
                           / (2.0 * np.sqrt(st.amax * st.bcomf))[:, None], 0.0))
    a_pair = -st.amax[:, None] * (s_star / np.maximum(gap, 0.3)) ** 2 * overlap
    a_pair = np.where(cand, a_pair, BIG)
    a_idm = a_pair.min(axis=1)                       # most restrictive leader
    if not want_leader:
        return fx, fy, a_idm
    leader = np.where(a_idm < BIG, a_pair.argmin(axis=1), -1)
    return fx, fy, a_idm, leader


# ----------------------------------------------------------------------
# 3b. discretionary lane changing: overtake incentive + keep-right pressure
#     (added 2026-08-06; sec. 4b of docs/FRAMEWORK.md)
# ----------------------------------------------------------------------
def frustration(st: VehState, p, leader):
    """How much a vehicle's own leader is costing it, in [0, 1].

    Two factors, both required. The speed factor asks whether the leader is
    slower than this driver *wants* to go; the proximity factor asks whether the
    leader is close enough for that to matter yet. Without the second, a car
    200 m behind a slow truck would already be pulling out; without the first,
    a car following a leader at its own desired speed would.
    """
    has = leader >= 0
    j = np.where(has, leader, 0)
    dv = np.clip((st.v0 - st.vx[j]) / np.maximum(st.v0, 1e-6), 0.0, 1.0)
    gap = np.maximum(st.x[j] - st.x - 0.5 * (st.L[j] + st.L), 0.0)
    thw = gap / np.maximum(st.vx, 0.1)
    prox = np.clip((p.T_frust - thw) / max(p.T_frust, 1e-6), 0.0, 1.0)
    return np.where(has & ~st.is_ev, dv * prox, 0.0)


def rear_pressure(st: VehState, p, follower):
    """How much the vehicle *behind* is being held up by this one, in [0, 1].

    The reciprocal of `frustration`, and the reason it exists rather than a blunt
    "keep right when the right lane is free": with the blunt rule every
    unimpeded car in the middle and left lanes commits to the right on every
    cooldown expiry, arrives, finds itself slower than it wants, and pulls out
    again - measured as ~660 manoeuvres per 3 runs, a lane-keeping offset of
    0.42 m against a real 0.27, and 7-11 s median manoeuvres. Real drivers move
    over when somebody faster is behind them, which is what this measures.

    It also makes the framework's central claim structural rather than
    stipulated: an ordinary faster follower and an emergency vehicle push the
    car in front sideways through the *same* mechanism, differing only in the
    strength and range of that push. The EV terms are then an extension of a
    term that ordinary traffic already needs, not a bolted-on special case.
    """
    has = follower >= 0
    j = np.where(has, follower, 0)
    dv = np.clip((st.v0[j] - st.vx) / np.maximum(st.v0[j], 1e-6), 0.0, 1.0)
    gap = np.maximum(st.x - st.x[j] - 0.5 * (st.L[j] + st.L), 0.0)
    thw = gap / np.maximum(st.vx[j], 0.1)
    prox = np.clip((p.T_frust - thw) / max(p.T_frust, 1e-6), 0.0, 1.0)
    return np.where(has & ~st.is_ev, dv * prox, 0.0)


def lane_incentive(st: VehState, road: Road, p, lane=None):
    """Sustained lateral push toward the lane a driver has committed to.

    Pure function of state: whether a driver is committed, and to which lane, is
    decided by `emv/lanechange.py:LaneChange` (which owns `st.lc_target`) exactly
    as `Perception` decides whether a driver yields and `corridor_field` then
    supplies the push. Amplitude is constant while committed - `A_pass` when
    moving left to overtake, `A_keep_right` when returning right - so the
    *duration* and *peak lateral speed* of the manoeuvre are set by the lateral
    block (a_pin, zeta_lat, v_lat_max, a_lat_max) and not by these amplitudes.
    That separation is what lets stage 1 fit the rate and the kinematics against
    different observables without the two fighting each other.

    The force releases itself the moment the marking is crossed (`lane` reaches
    `lc_target`, so the sign term is zero): the driver is then at the potential
    *maximum* with the washboard slope pushing them on into the new lane, which
    reproduces the steer-out/settle-in shape of a real manoeuvre.

    Returns zeros (cheaply) when both amplitudes are off - the default, and the
    behaviour of record for papers 1-6.
    """
    if p.A_pass == 0.0 and p.A_keep_right == 0.0:
        return np.zeros(st.n)
    lane = lane_index(st, road) if lane is None else lane
    d = np.sign(st.lc_target - lane) * (st.lc_target >= 0)
    return np.where(d > 0, p.A_pass, np.where(d < 0, -p.A_keep_right, 0.0))


# ----------------------------------------------------------------------
# 4. EV point repulsion: anisotropic exponential field (user term 2)
# ----------------------------------------------------------------------
def ev_field(st: VehState, p):
    """F = A exp((r - d)/B) * n_hat * (lam + (1 - lam)(1 + cos phi)/2)
    with n_hat from EV to car and phi the angle between the EV heading and
    n_hat: the field is strongly forward-focused [2, 12, 16]. Only drivers
    in the YIELDING/HOLD states feel it (perception gating, gap 2).
    """
    i = st.ev
    ddx = st.x - st.x[i]
    ddy = st.y - st.y[i]
    d = np.hypot(ddx, ddy)
    d = np.maximum(d, 0.5)
    nx, ny = ddx / d, ddy / d

    ex, ey = st.ev_heading()
    cphi = nx * ex + ny * ey
    wgt = p.lam_ev + (1.0 - p.lam_ev) * 0.5 * (1.0 + cphi)
    mag = np.minimum(p.A_ev * np.exp((p.r_ev - d) / p.B_ev), p.F_cap_ev)

    gate = (((st.aware == YIELDING) | (st.aware == HOLD)) & ~st.is_ev).astype(float)
    f = mag * wgt * gate
    return f * nx, f * ny


# ----------------------------------------------------------------------
# 5. corridor repulsion from the EV's predicted path (user term 3; gap 1)
# ----------------------------------------------------------------------
def corridor_field(st: VehState, p, corr: Corridor):
    """Repulsion from the *predicted path* rather than the EV position [8].

    Lateral term: pushes cars out of the corridor toward their chosen escape
    side; saturates at A_c inside the required clearance and decays with
    length B_c beyond it. Modulated by urgency u (timing, [20]) and a smooth
    fade at the far corridor end.

    Longitudinal term: 'pull over AND slow down' - cars still blocking the
    corridor shed speed toward kappa_merge * v0 so they can drop into gaps
    in the adjacent lane instead of pacing the EV.
    """
    s, dlat = corr.rel(st.x, st.y)
    in_corr = (s > -corr.L_back) & (s < corr.L) & ~st.is_ev

    w_need = 0.5 * (st.W[st.ev] + st.W) + p.margin_c
    over = np.maximum(np.abs(dlat) - w_need, 0.0)
    fade = np.clip((corr.L - s) / p.end_ramp, 0.0, 1.0)
    gate = np.where(in_corr, st.u * fade, 0.0)

    side = np.where(st.side != 0, st.side, np.where(dlat >= 0.0, 1.0, -1.0))
    f_lat = p.A_c * np.exp(-over / p.B_c) * gate * side

    blocked = np.clip((w_need - np.abs(dlat)) / w_need, 0.0, 1.0)
    f_long = -p.gamma_c * gate * blocked * np.maximum(st.vx - p.kappa_merge * st.v0, 0.0)
    return f_long, f_lat


# ----------------------------------------------------------------------
# total
# ----------------------------------------------------------------------
def total(st: VehState, road: Road, p, corr: Corridor, diag: bool = False):
    lane = lane_index(st, road)
    rho_fac = pin_density_factor(st, road, p)
    ax_d, ay_d = drive(st)
    ay_lane = lane_keep(st, road, p, corr,
                        rho_fac=None if p.k_rho == 0.0 else rho_fac)
    fx_s, fy_s, a_idm = car_car(st, p)
    ay_p = lane_incentive(st, road, p, lane=lane)
    fx_e, fy_e = ev_field(st, p)
    fx_c, fy_c = corridor_field(st, p, corr)

    social_x = fx_s + fx_e + fx_c
    ax = ax_d + (np.minimum(social_x, a_idm) if p.idm_bound else social_x)
    ay = ay_d + ay_lane + ay_p + fy_s + fy_e + fy_c

    if diag:
        z = np.zeros_like(ay_lane)
        return ax, ay, dict(drive=(ax_d, ay_d), lane=(z, ay_lane),
                            sfm=(fx_s, fy_s), lc=(z.copy(), ay_p),
                            ev=(fx_e, fy_e), corr=(fx_c, fy_c),
                            idm=a_idm)
    return ax, ay, None
