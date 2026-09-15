"""Siren perception and the driver awareness state machine (gap 2).

Field models in the literature are geometric; detection distance and driver
reaction latency are rarely modelled although timing dominates outcomes [20].
Here the repulsion fields are *gated* by a per-driver state machine:

    UNAWARE --(EV within detection radius)--> NOTICED
    NOTICED --(reaction delay elapsed)-----> YIELDING   (escape side chosen)
    YIELDING --(EV passed by s_passed)-----> HOLD       (keep position)
    HOLD    --(hold timer elapsed)---------> UNAWARE    (merge back)

Detection is anisotropic: a car ahead of the EV sees flashing lights in the
mirror and hears the siren early (R_front); a car behind the EV notices late
(R_rear). Reaction delays are lognormal; a fraction p_noncomply never yields
(distracted drivers, [18]).

Urgency u in [0, 1] scales the corridor force by the estimated EV time of
arrival:  u = sigmoid((T_react - t_arr)/sigma_t). Drivers act when the EV is
~T_react seconds behind them - not sooner, not later [20].
"""
import numpy as np

from .state import VehState, UNAWARE, NOTICED, YIELDING, HOLD
from .road import Road, Corridor


class Perception:
    def __init__(self, st: VehState, p, road: Road, rng: np.random.Generator,
                 enabled: bool = True, sample: bool = True):
        self.p = p
        self.road = road
        self.rng = rng
        self.enabled = enabled
        if sample:                      # SUMO bridge samples per-arrival instead
            n = st.n
            st.delay[:] = np.exp(rng.normal(np.log(p.delay_med), p.delay_sig, n))
            st.comply[:] = rng.random(n) >= p.p_noncomply
            st.comply[st.ev] = True

    # ------------------------------------------------------------------
    def update(self, st: VehState, corr: Corridor, t: float):
        p, road = self.p, self.road
        i_ev = st.ev
        if not self.enabled:
            st.u[:] = 0.0
            return

        s, dlat = corr.rel(st.x, st.y)
        d_ev = np.hypot(st.x - st.x[i_ev], st.y - st.y[i_ev])
        ahead = s >= 0.0

        # --- UNAWARE -> NOTICED: anisotropic detection ------------------
        R = np.where(ahead, p.R_front, p.R_rear)
        trig = (st.aware == UNAWARE) & (d_ev < R) & st.comply & ~st.is_ev
        st.t_react[trig] = t + st.delay[trig]
        st.aware[trig] = NOTICED

        # --- NOTICED -> YIELDING: reaction delay elapsed -----------------
        act = (st.aware == NOTICED) & (t >= st.t_react)
        if act.any():
            st.aware[act] = YIELDING
            st.side[act] = self._choose_side(st, dlat, act)

        # --- YIELDING -> HOLD: EV has passed -----------------------------
        # 'passed' needs the EV to actually recede: a car behind the EV nose
        # (s < -s_passed) is only done yielding if the EV is pulling away.
        # For a moving EV this is the old rule (whoever it passed is slower);
        # it keeps a *parked* EV from ending yield episodes on arrival.
        recede = st.vx[i_ev] - st.vx > p.v_passed
        passed = (st.aware == YIELDING) & (s < -p.s_passed) & recede
        if passed.any():
            st.aware[passed] = HOLD
            st.hold_until[passed] = t + self.rng.uniform(p.T_hold_lo, p.T_hold_hi,
                                                         int(passed.sum()))

        # --- HOLD -> UNAWARE: merge back ----------------------------------
        release = (st.aware == HOLD) & (t >= st.hold_until)
        st.aware[release] = UNAWARE
        st.side[release] = 0

        # --- urgency: estimated EV time of arrival ----------------------
        # Anticipation horizon stretches as own manoeuvrability shrinks:
        # a crawling car needs far more lead time to shuffle aside than a
        # free-flowing car needs to change lanes (Rettungsgasse pre-forming).
        v_app = max(st.vx[i_ev], p.v_app_floor * st.v0[i_ev])
        t_arr = np.where(ahead, s / np.maximum(v_app - st.vx, 0.5), 0.0)
        jam = np.clip(1.0 - st.vx / p.v_manoeuvre_ref, 0.0, 1.0)
        T_eff = p.T_react * (1.0 + p.jam_anticipation * jam)
        u_time = 1.0 / (1.0 + np.exp(np.clip((t_arr - T_eff) / p.sigma_t, -40, 40)))
        # proximity channel (jam-gated): a blocked EV still projects urgency,
        # otherwise slow EV -> calm traffic -> blocked EV is self-sustaining
        gate = np.maximum(jam, p.prox_gate_floor)
        u_prox = gate / (1.0 + np.exp(np.clip((s - p.R_urgent) / p.sigma_s, -40, 40)))
        u = np.maximum(u_time, u_prox)
        st.u[:] = np.where(st.aware == YIELDING, u, 0.0)
        st.u[st.aware == HOLD] = 0.0
        st.u[i_ev] = 0.0

    # ------------------------------------------------------------------
    def _choose_side(self, st: VehState, dlat, mask):
        """Escape side with hysteresis: keep it until the yield episode ends.
        Cars clearly off the corridor line move further to their own side;
        cars near the line pick the side with more drivable room, with a
        right-hand bias (pull-right convention)."""
        p, road = self.p, self.road
        room_left = (road.y_max - 0.5 * st.W) - st.y
        room_right = (st.y - 0.5 * st.W) - road.y_min + p.side_right_bias
        by_room = np.where(room_left > room_right, 1, -1)
        by_pos = np.sign(dlat)
        side = np.where(np.abs(dlat) >= 0.35, by_pos, by_room).astype(np.int8)
        side[side == 0] = -1
        return side[mask]
