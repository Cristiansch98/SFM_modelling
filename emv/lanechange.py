"""Discretionary lane-change decisions (added 2026-08-06).

Why this module exists
----------------------
With no emergency vehicle in the scenario the model made **zero** lane changes
over 206 veh-km, against a measured highD rate of 0.2758 per veh-km in free
flow and 0.2026 in dense traffic: every lane
change it produced was EV-induced. That made the "host-traffic" calibration
circular - the lateral block could only be identified *because* an EV was
present. This module supplies the missing incentive, so that ordinary German
motorway traffic can be calibrated with the emergency vehicle absent, and the EV
terms can then be fitted on top of a frozen normal model (two-stage calibration).

Structure mirrors `emv/perception.py`: a small manager owns the per-vehicle
decision state and is stepped by `Sim` before the forces are evaluated, so the
force term itself (`forces.lane_incentive`) stays a pure function of state. The
division of labour is the same one the EV side already uses - `Perception`
decides *whether* a driver acts and `corridor_field` supplies the push.

The decision rule is the depinning condition
--------------------------------------------
A driver commits to a lane change when the incentive can lift them out of their
lane's potential well:

    A_pass * frust        > a_pin_eff      (overtake a slower leader)
    A_keep_right * (1-frust) > a_pin_eff   (return to the right)

with `a_pin_eff = a_pin * (rho/rho_ref)^k_rho * (1 - urgency_pin_relief * u)`.
That is the *same* inequality the EV corridor force must satisfy
(`A_c * u > a_pin_eff`, docs/FRAMEWORK.md sec. 4), so the framework carries one
depinning condition with two things that can drive it - which is what makes the
lane-change rate predictable from the parameters rather than merely observed.

Why the decision is latched
---------------------------
The first implementation applied the incentive only while the vehicle sat near
its lane centre, on the theory that the washboard slope would carry it the rest
of the way. **Measured, and it does not**: with the calibrated lateral block
(a_pin 0.224, zeta_lat 2.42 - weak pinning, heavily overdamped) the residual
slope moves a car sideways at ~0.17 m/s, so a manoeuvre took 24-38 s against a
real 2.28 s. A committed driver steers continuously until the manoeuvre is done,
so the target lane is latched and the force is sustained until the marking is
crossed. Duration then remains a property of the lateral block (the force is
constant while committed), which is what keeps the stage-1 blocks separable.
"""
import numpy as np

from .road import Road
from .state import VehState, YIELDING, HOLD


class LaneChange:
    """Owns `st.lc_target` (committed lane, -1 = none) and `st.lc_free_at`."""

    def __init__(self, st: VehState, p, road: Road):
        self.p, self.road = p, road
        self.enabled = bool(p.A_pass > 0.0 or p.A_keep_right > 0.0)
        #: counters, for the term-influence study (never used by the dynamics)
        self.stats = dict(commit=0, arrive=0, abort=0, timeout=0)

    # ------------------------------------------------------------------
    def update(self, st: VehState, t: float, rho_fac=None):
        if not self.enabled:
            return
        p, road = self.p, self.road
        lane = np.clip(np.round(st.y / road.lane_width - 0.5), 0,
                       road.n_lanes - 1).astype(int)

        # ---- release committed manoeuvres --------------------------------
        act = st.lc_target >= 0
        if act.any():
            arrived = act & (lane == st.lc_target)
            timeout = act & ~arrived & (t > st.lc_until)
            blocked = act & ~arrived & ~timeout & self._blocked(st, lane,
                                                               st.lc_target)
            done = arrived | timeout | blocked
            self.stats["arrive"] += int(arrived.sum())
            self.stats["timeout"] += int(timeout.sum())
            self.stats["abort"] += int(blocked.sum())
            st.lc_target[done] = -1
            st.lc_free_at[done] = t + p.T_lc_cool

        # ---- commit new manoeuvres ---------------------------------------
        from .forces import frustration, rear_pressure   # local: avoids a cycle
        # The commit threshold. `a_commit=None` uses a_pin, which is the pure
        # depinning reading; setting it separates *how often* a manoeuvre is
        # started from *how* it is executed, because a_pin governs both and they
        # pull in opposite directions (see params.a_commit for the measurements).
        a_pin_v = p.a_pin if st.a_pin is None else st.a_pin
        if p.a_commit is not None:
            # keep any per-driver dispersion of a_pin, rescaled to the threshold
            a_pin_v = a_pin_v * (p.a_commit / max(p.a_pin, 1e-9))
        if rho_fac is not None:
            a_pin_v = a_pin_v * rho_fac
        a_pin_eff = a_pin_v * (1.0 - p.urgency_pin_relief * st.u)

        y_home = (lane + 0.5) * road.lane_width + st.y_pref
        ready = ((st.lc_target < 0) & (t >= st.lc_free_at) & ~st.is_ev
                 & (np.abs(st.y - y_home) < p.lc_gate_frac * road.lane_width)
                 # a driver already busy giving way to the EV does not also
                 # start a discretionary manoeuvre; existing commitments finish
                 & (st.aware != YIELDING) & (st.aware != HOLD))
        if not ready.any():
            return
        # Both incentives are *comparative*, as in MOBIL: what matters is not
        # only what the current lane costs but whether the target lane costs
        # less. Moving left is driven by one's own frustration, moving right by
        # the pressure of a faster follower (see forces.rear_pressure for the
        # measurements that ruled out the blunt "move right when it is free").
        frust = frustration(st, p, self._leader(st, lane, lane))
        f_left = frustration(st, p, self._leader(st, lane, lane + 1))
        f_right = frustration(st, p, self._leader(st, lane, lane - 1))
        rear = rear_pressure(st, p, self._follower(st, lane, lane))
        want_left = p.A_pass * frust * (1.0 - f_left)
        want_right = p.A_keep_right * rear * (1.0 - f_right)
        # ties go to overtaking: a frustrated driver in the middle lane pulls out
        # rather than tucking in, which is what the keep-right rule prescribes
        go_left = ready & (want_left > a_pin_eff) & (lane < road.n_lanes - 1)
        go_right = (ready & ~go_left & (want_right > a_pin_eff) & (lane > 0))
        for mask, d in ((go_left, +1), (go_right, -1)):
            if not mask.any():
                continue
            tgt = lane + d
            ok = mask & ~self._blocked(st, lane, tgt)
            st.lc_target[ok] = tgt[ok]
            st.lc_until[ok] = t + p.T_lc_max
            self.stats["commit"] += int(ok.sum())

    # ------------------------------------------------------------------
    def _leader(self, st: VehState, lane, tgt):
        """Nearest vehicle ahead in lane `tgt[i]` as seen by vehicle i, or -1.

        With `tgt = lane` this is the same relation highD's `precedingId` column
        encodes and `metrics.headway_pools` measures. With `tgt = lane +/- 1` it
        is the *prospective* leader used for the comparative incentive.
        """
        return self._nearest(st, lane, tgt, ahead=True)

    def _follower(self, st: VehState, lane, tgt):
        """Nearest vehicle behind in lane `tgt[i]`, or -1."""
        return self._nearest(st, lane, tgt, ahead=False)

    @staticmethod
    def _nearest(st: VehState, lane, tgt, ahead: bool):
        dx = st.x[None, :] - st.x[:, None]
        signed = dx if ahead else -dx
        cand = (signed > 0.0) & (lane[None, :] == tgt[:, None])
        dist = np.where(cand, signed, np.inf)
        j = dist.argmin(axis=1)
        return np.where(np.isfinite(dist[np.arange(st.n), j]), j, -1)

    def _blocked(self, st: VehState, lane, tgt):
        """True where the target lane is not safe to move into.

        Two criteria, because a distance alone is not enough. Both were forced by
        measurement, in two rounds:

        * A static +/- s_veto window alone **produced collisions**: highD's
          measured per-lane speeds differ by up to 17 m/s (24 / 34 / 41 m/s
          right-to-left), so a follower 21 m back in the target lane is 1.2 s
          away, not comfortably behind.
        * Adding a closing-time criterion at the driver's own headway
          (`T_hw`, 1.1-1.8 s) **still produced collisions** - 3 contact
          frame-pairs in free flow, all between 17-19 m trucks, against 0 with the
          incentive switched off. The reason is that a lane change *takes* about
          2.3 s, so a follower accepted as "1.85 s away" arrives in the middle of
          the manoeuvre. The threshold must therefore cover the headway **plus
          the time to cross the lane**, `w / v_lat_max`, both of which are
          already per-driver quantities - so no new parameter is introduced and
          gap acceptance stays tied to the same limits the manoeuvre obeys.

        This is not a substitute for the safety layer: the elliptic repulsion and
        the IDM bound still govern what happens during the manoeuvre. It stops a
        driver *committing* to a slot that is occupied or about to be. It is
        needed because at a 13 m/s closing speed a following truck physically
        cannot brake out of the conflict once it has started.
        """
        p, road = self.p, self.road
        in_tgt = lane[None, :] == tgt[:, None]
        in_tgt &= ~np.eye(st.n, dtype=bool) & ~st.is_ev[None, :]
        dx = st.x[None, :] - st.x[:, None]          # dx[i, j] = x_j - x_i
        occupied = (in_tgt & (np.abs(dx) <= p.s_veto)).any(axis=1)
        # a vehicle behind me in the target lane, closing: the time until it
        # reaches my longitudinal position must cover my headway AND my crossing
        v_lat = p.v_lat_max if st.v_lat_max is None else st.v_lat_max
        need = st.T_hw + road.lane_width / np.maximum(v_lat, 0.1)
        closing = st.vx[None, :] - st.vx[:, None]
        behind = in_tgt & (dx < 0.0)
        t_arr = np.where(behind & (closing > 0.1), -dx / np.maximum(closing, 0.1),
                         np.inf)
        too_soon = (t_arr < need[:, None]).any(axis=1)
        outside = (tgt < 0) | (tgt >= road.n_lanes)
        return occupied | too_soon | outside
