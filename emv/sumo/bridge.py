"""Hybrid coupling of the repulsion model to SUMO via TraCI (gap 4).

SUMO keeps what it is good at - car-following, lane discipline, insertion -
and the force model supplies the EV-yielding behaviour on top:

  longitudinal:  F_x < 0  ->  slowDown(v + F_x * H, H)  (speedMode stays at
                 default 31, so SUMO's safe-speed logic still binds: safety
                 always wins, mirroring the min() composition of the
                 standalone model)
  lateral:       F_y -> target offset y_t = lane_centre + clamp(F_y / k_map)
                 tracked with a rate-limited proportional step (the discrete,
                 overshoot-free equivalent of a critically damped spring
                 [10]), issued via changeSublane each step; a lateral-gap
                 check vetoes pushes into an occupied sublane.

The *same* emv.forces.ev_field / corridor_field / emv.perception.Perception
code computes the fields - single source of truth with the standalone model.

Modes: "none" (no yielding), "bluelight" (SUMO's native rescue-lane device,
the baseline of [17]), "force" (this model), "rule" (scripted
Rettungsgasse baseline: fixed trigger distance and delay, lane-keyed
edge-hugging, scripted slowdown, 100 % compliance - the behaviour a
trainer typically hard-codes; third condition of the survey benchmark).

For behaviour comparison against real ambulance interactions, every mode
also records a 1 Hz dashcam-style ego-frame stream (``ego_frames`` in the
returned metrics) in the exact frame convention of the
emergency-vehicle-dataset-pipeline (see emv/groundtruth.py): x lateral,
+ = right of the EV; y forward of the EV front to the target's rear.
"""
import os
import numpy as np

from ..params import Params
from ..road import Road, build_corridor
from ..state import VehState, blank_state, UNAWARE, NOTICED, YIELDING, HOLD
from ..perception import Perception
from .. import forces
from . import make_scenario as sc
from . import traj as trajrec

DEFAULT_LC_MODE = 0b011001010101      # SUMO default (1621)
K_MAP = 1.0                           # m/s^2 per m: force -> lateral offset
MAX_OFF = 5.5                         # m, max commanded offset from lane centre
SEG = (400.0, 2600.0)                 # EV measurement segment on the edge

# scripted "rule" mode (survey condition C) - all fixed, no perception model
RULE_TRIGGER = 100.0   # m, react once the EV is within this (= bluelight
                       # reactiondist, so conditions B and C sense alike)
RULE_RELEASE = 30.0    # m the EV must be past before releasing
RULE_DELAY = 1.0       # s, fixed scripted reaction delay
RULE_SLOW = 0.6        # slow to this fraction of the allowed speed
EGO_FWD_MAX = 120.0    # m, ego-frame recorder forward field of view
EGO_LAT_PAD = 3.0      # m, lateral FOV beyond the road half-width (shoulder)

#: awareness-state colours for the GUI (match emv.viz dark-theme states)
STATE_RGBA = {UNAWARE: (125, 125, 120, 255), NOTICED: (201, 133, 0, 255),
              YIELDING: (217, 89, 38, 255), HOLD: (25, 158, 112, 255)}


class SumoBridge:
    def __init__(self, mode: str, p: Params | None = None, seed: int = 42,
                 out_dir: str = "out/sumo", gui: bool = False,
                 delay_ms: int = 70, colorize: bool = True,
                 record_traj: bool = False, ev_fixed_lane: bool = False):
        assert mode in ("none", "bluelight", "force", "rule")
        self.mode = mode
        #: ablation: pin the EV to its departure lane (laneChangeMode 0) so
        #: ALL corridor-making comes from the surrounding traffic. Note
        #: PROJECT_LOG gotcha 3: a pinned EV relies entirely on the yielding
        #: mechanism to clear stragglers - that dependence is the point.
        self.ev_fixed_lane = ev_fixed_lane
        self.p = p or Params()
        self.seed = seed
        self.out_dir = out_dir
        self.gui = gui
        self.delay_ms = delay_ms      # gui pacing (ms per 0.1 s step)
        self.colorize = colorize      # recolour vehicles by awareness state
        self.traj = trajrec.blank_traj() if record_traj else None
        self.rng = np.random.default_rng(seed + 1)
        self.road = Road(n_lanes=sc.N_LANES, lane_width=sc.LANE_W,
                         shoulder_right=0.0, shoulder_left=0.0,
                         length=sc.EDGE_LEN)
        # persistent per-vehicle bridge state
        self.reg: dict[str, dict] = {}
        self.dims: dict[str, tuple] = {}
        self.controlled: set[str] = set()

    # ------------------------------------------------------------------
    def run(self) -> dict:
        import traci
        import traci.constants as tc

        import time as _time
        cfg = sc.write_scenario(self.out_dir)
        cmd = [sc.sumo_bin("sumo-gui" if self.gui else "sumo"), "-c", cfg,
               "--seed", str(self.seed), "--start"]
        if self.gui:
            # pre-EV phase fast-forwards; python-side pacing takes over once
            # the EV is on the road (see sleep below)
            cmd += ["--delay", "0"]
        if self.mode == "bluelight":
            # device must exist at insertion: assign explicitly by vehicle id
            cmd += ["--device.bluelight.explicit", "EV",
                    "--device.bluelight.reactiondist", "100"]
        if self.gui:
            print("[bridge] starting sumo-gui (first launch may trigger a "
                  "Windows Firewall prompt - click Allow) ...", flush=True)
        traci.start(cmd, label=f"emv_{self.mode}_{self.seed}")
        conn = traci
        if self.gui:
            print("[bridge] sumo-gui connected; fast-forwarding to EV "
                  "departure (t = 120 s) ...", flush=True)

        subs = (tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
                tc.VAR_LANEPOSITION_LAT, tc.VAR_ROAD_ID, tc.VAR_ALLOWED_SPEED)
        per = None
        rec = dict(t=[], ev_x=[], ev_v=[], clearance=[], n_yield=[], bg_v=[],
                   ego=[])
        collisions = 0
        ev_added = False
        ev_gone_at = None
        dt = 0.1
        t = 0.0

        try:
            while t < sc.SIM_END:
                for vid in conn.simulation.getDepartedIDList():
                    conn.vehicle.subscribe(vid, subs)
                conn.simulationStep()
                t = conn.simulation.getTime()
                collisions += conn.simulation.getCollidingVehiclesNumber()

                if not ev_added and t >= sc.EV_DEPART:
                    conn.vehicle.add("EV", "r0", typeID="ev",
                                     departLane="1", departSpeed="max")
                    if self.ev_fixed_lane:
                        conn.vehicle.setLaneChangeMode("EV", 0)
                    ev_added = True
                    if self.gui:
                        self._gui_setup(conn)
                        print("[bridge] EV departed - camera locked, "
                              "real-time pacing on", flush=True)
                if ev_added and self.gui and self.delay_ms > 0:
                    _time.sleep(self.delay_ms / 1000.0)
                if ev_added and "EV" not in conn.vehicle.getIDList():
                    if t > sc.EV_DEPART + 5.0:
                        ev_gone_at = t
                        break
                    continue
                if not ev_added:
                    continue
                for vid in conn.simulation.getDepartedIDList():
                    conn.vehicle.subscribe(vid, subs)

                res = conn.vehicle.getAllSubscriptionResults()
                ids = [v for v, r in res.items() if r.get(tc.VAR_ROAD_ID) == "hw"]
                if "EV" not in ids:
                    continue
                st = self._assemble(conn, res, ids, tc)
                corr = build_corridor(
                    st.x[st.ev] + 0.5 * st.L[st.ev], st.vx[st.ev],
                    (res["EV"][tc.VAR_LANE_INDEX] + 0.5) * sc.LANE_W,
                    res["EV"][tc.VAR_ALLOWED_SPEED], self.p)

                if self.mode == "force":
                    if per is None:
                        per = Perception(st, self.p, self.road, self.rng,
                                         enabled=True, sample=False)
                    per.update(st, corr, t)
                    self._writeback(ids, st)
                    self._apply_forces(conn, st, corr, ids)
                    if self.gui and self.colorize:
                        self._colorize(conn, st, ids)
                elif self.mode == "rule":
                    self._apply_rule(conn, st, ids, t)

                self._record(rec, t, st, corr, ids)
        except traci.exceptions.FatalTraCIError:
            print("[bridge] GUI window closed - ending run.", flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return self._metrics(rec, collisions, ev_gone_at)

    # ------------------------------------------------------------------
    def _gui_setup(self, conn):
        """Camera on the EV, real-world rendering. Best effort - any of
        these may be unavailable depending on the SUMO build."""
        view = "View #0"
        for do in (lambda: conn.gui.setSchema(view, "real world"),
                   lambda: conn.gui.setZoom(view, 1500),
                   lambda: conn.gui.trackVehicle(view, "EV")):
            try:
                do()
            except Exception:
                pass

    def _colorize(self, conn, st, ids):
        """Recolour vehicles by awareness state (matches the paper/animation
        legend: grey unaware, amber noticed, orange yielding, teal hold)."""
        for k, vid in enumerate(ids):
            if vid == "EV":
                continue
            g = self.reg[vid]
            s_now = int(st.aware[k])
            if g.get("col_state") != s_now:
                try:
                    conn.vehicle.setColor(vid, STATE_RGBA[s_now])
                except Exception:
                    pass
                g["col_state"] = s_now

    # ------------------------------------------------------------------
    def _assemble(self, conn, res, ids, tc) -> VehState:
        n = len(ids)
        d = blank_state(n)
        p = self.p
        for k, vid in enumerate(ids):
            r = res[vid]
            if vid not in self.dims:
                self.dims[vid] = (conn.vehicle.getLength(vid),
                                  conn.vehicle.getWidth(vid))
            if vid not in self.reg:
                self.reg[vid] = dict(
                    aware=UNAWARE, side=0, t_react=np.inf, hold_until=np.inf,
                    delay=float(np.exp(self.rng.normal(np.log(p.delay_med),
                                                       p.delay_sig))),
                    comply=bool(self.rng.random() >= p.p_noncomply),
                    lc_saved=False, rule_t0=None, rule_on=False)
            g = self.reg[vid]
            d["x"][k] = r[tc.VAR_LANEPOSITION]
            d["y"][k] = ((r[tc.VAR_LANE_INDEX] + 0.5) * sc.LANE_W
                         + r[tc.VAR_LANEPOSITION_LAT])
            d["vx"][k] = r[tc.VAR_SPEED]
            d["v0"][k] = r[tc.VAR_ALLOWED_SPEED]
            d["L"][k], d["W"][k] = self.dims[vid]
            d["is_ev"][k] = vid == "EV"
            d["aware"][k], d["side"][k] = g["aware"], g["side"]
            d["t_react"][k], d["hold_until"][k] = g["t_react"], g["hold_until"]
            d["delay"][k], d["comply"][k] = g["delay"], g["comply"]
        return VehState(**d)

    def _writeback(self, ids, st: VehState):
        for k, vid in enumerate(ids):
            g = self.reg[vid]
            g["aware"], g["side"] = int(st.aware[k]), int(st.side[k])
            g["t_react"], g["hold_until"] = float(st.t_react[k]), float(st.hold_until[k])

    # ------------------------------------------------------------------
    def _apply_forces(self, conn, st: VehState, corr, ids):
        fx_e, fy_e = forces.ev_field(st, self.p)
        fx_c, fy_c = forces.corridor_field(st, self.p, corr)
        f_lat = fy_e + fy_c
        f_long = fx_c
        y_lane = self.road.nearest_lane_center(st.y)

        for k, vid in enumerate(ids):
            if vid == "EV":
                continue
            g = self.reg[vid]
            active = st.aware[k] in (YIELDING, HOLD)
            if active and not g["lc_saved"]:
                conn.vehicle.setLaneChangeMode(vid, 0)
                g["lc_saved"] = True
            elif not active and g["lc_saved"]:
                conn.vehicle.setLaneChangeMode(vid, DEFAULT_LC_MODE)
                g["lc_saved"] = False
                continue
            if not active:
                continue

            y_t = y_lane[k] + float(np.clip(f_lat[k] / K_MAP, -MAX_OFF, MAX_OFF))
            self._steer_to(conn, st, k, vid, y_t)
            # yield-braking only well ahead of the EV: braking right at its
            # nose would just brake the (car-following) EV as well
            if f_long[k] < -0.05 and (st.x[k] - st.x[st.ev]) > 15.0:
                v_cmd = max(st.vx[k] + f_long[k] * 0.8, 0.0)
                conn.vehicle.slowDown(vid, v_cmd, 0.8)

    def _steer_to(self, conn, st: VehState, k: int, vid: str, y_t: float):
        """Shared actuation for force and rule modes: identical rate-limited
        sublane steps + safety veto, so the survey setups differ only in the
        decision logic, never in how commands reach SUMO."""
        y_t = float(np.clip(y_t, self.road.y_min + 0.5 * st.W[k] + 0.1,
                            self.road.y_max - 0.5 * st.W[k] - 0.1))
        dy = y_t - st.y[k]
        # small per-step increments keep the safety veto binding at every
        # step (a large one-shot request would keep executing between
        # vetoes); explicit 0 cancels a running manoeuvre when unsafe
        if abs(dy) > 0.05:
            if self._lat_gap_ok(st, k, np.sign(dy)):
                conn.vehicle.changeSublane(vid, float(np.clip(dy, -0.25, 0.25)))
            else:
                conn.vehicle.changeSublane(vid, 0.0)

    # ------------------------------------------------------------------
    def _apply_rule(self, conn, st: VehState, ids, t: float):
        """Survey condition C: scripted Rettungsgasse. Deterministic trigger
        (EV within RULE_TRIGGER), fixed RULE_DELAY, lane-keyed targets
        (leftmost lane hugs its left edge, every other lane its right edge),
        scripted slowdown, everyone complies. No perception, no urgency."""
        ev = st.ev
        left_lane = self.road.n_lanes - 1
        for k, vid in enumerate(ids):
            if vid == "EV":
                continue
            g = self.reg[vid]
            d = st.x[k] - st.x[ev]              # + = ahead of the EV
            inside = -RULE_RELEASE < d < RULE_TRIGGER
            if inside and g["rule_t0"] is None:
                g["rule_t0"] = t
            elif not inside:
                g["rule_t0"] = None
            active = (g["rule_t0"] is not None
                      and t - g["rule_t0"] >= RULE_DELAY)
            if active and not g["lc_saved"]:
                conn.vehicle.setLaneChangeMode(vid, 0)
                g["lc_saved"] = True
            elif not active and g["lc_saved"]:
                conn.vehicle.setLaneChangeMode(vid, DEFAULT_LC_MODE)
                g["lc_saved"] = False
            if self.gui and self.colorize and g["rule_on"] != active:
                try:
                    conn.vehicle.setColor(
                        vid, STATE_RGBA[YIELDING if active else UNAWARE])
                except Exception:
                    pass
                g["rule_on"] = active
            if not active:
                continue
            lane = int(np.clip(st.y[k] // sc.LANE_W, 0, left_lane))
            if lane == left_lane:               # hug own lane's left edge
                y_t = (lane + 1) * sc.LANE_W - 0.5 * st.W[k] - 0.15
            else:                               # hug own lane's right edge
                y_t = lane * sc.LANE_W + 0.5 * st.W[k] + 0.15
            self._steer_to(conn, st, k, vid, y_t)
            v_slow = RULE_SLOW * st.v0[k]
            if d > 15.0 and st.vx[k] > v_slow:  # same EV-nose guard as force
                conn.vehicle.slowDown(vid, float(v_slow), 1.0)

    def _lat_gap_ok(self, st: VehState, k: int, sgn: float) -> bool:
        """Veto lateral pushes into an occupied sublane."""
        dx = st.x - st.x[k]
        dy = st.y - st.y[k]
        long_overlap = np.abs(dx) < 0.5 * (st.L + st.L[k]) + 2.0
        same_side = dy * sgn > 0.0
        near = long_overlap & same_side
        near[k] = False
        if not near.any():
            return True
        gap = np.abs(dy[near]) - 0.5 * (st.W[near] + st.W[k])
        return bool(gap.min() > 0.45)

    # ------------------------------------------------------------------
    def _record(self, rec, t, st: VehState, corr, ids):
        ev = st.ev
        if self.traj is not None:
            self._record_traj(t, st, corr, ids)
        if int(round(t * 10)) % 10 == 0:        # 1 Hz, like the dashcam data
            self._record_ego(rec, t, st, ids)
        w_need = 0.5 * (st.W[ev] + st.W) + self.p.margin_c
        dx = st.x - st.x[ev] - 0.5 * (st.L[ev] + st.L)
        blocking = (np.abs(st.y - corr.y_c) < w_need) & (dx > 0.0) & ~st.is_ev
        clr = float(dx[blocking].min()) if blocking.any() else 250.0
        others = ~st.is_ev
        rec["t"].append(t)
        rec["ev_x"].append(float(st.x[ev]))
        rec["ev_v"].append(float(st.vx[ev]))
        rec["clearance"].append(min(clr, 250.0))
        rec["n_yield"].append(int((st.aware == YIELDING).sum()))
        rec["bg_v"].append(float(st.vx[others].mean()) if others.any() else 0.0)

    def _record_ego(self, rec, t, st: VehState, ids):
        """Dashcam-style ego-frame snapshot in the dataset pipeline's frame:
        x lateral (+ = right of the EV; road y grows leftward, hence the
        sign flip), y forward from the EV front to the target's rear (what
        the camera's ground-plane projection measures). Only vehicles the
        dashcam would see: ahead, within EGO_FWD_MAX and the road width."""
        ev = st.ev
        fwd = (st.x - 0.5 * st.L) - (st.x[ev] + 0.5 * st.L[ev])
        lat = st.y[ev] - st.y
        half = 0.5 * self.road.n_lanes * self.road.lane_width
        keep = (~st.is_ev & (fwd > 0.0) & (fwd <= EGO_FWD_MAX)
                & (np.abs(lat) <= half + EGO_LAT_PAD))
        rec["ego"].append(dict(
            t=float(t), ev_v=float(st.vx[ev]),
            vehicles=[dict(id=ids[k], x=round(float(lat[k]), 2),
                           y=round(float(fwd[k]), 2))
                      for k in np.flatnonzero(keep)]))

    def _record_traj(self, t, st: VehState, corr, ids):
        """Full per-step trajectory frame (see emv/sumo/traj.py). State: the
        perception code in force mode, the scripted active flag mapped to
        YIELDING in rule mode (lc_saved tracks it exactly), UNAWARE elsewhere
        (bluelight/none expose no per-vehicle state)."""
        tr = self.traj
        idx = np.empty(len(ids), np.int32)
        for k, vid in enumerate(ids):
            j = tr["vindex"].get(vid)
            if j is None:
                j = len(tr["veh_ids"])
                tr["vindex"][vid] = j
                tr["veh_ids"].append(vid)
                tr["L"].append(float(st.L[k]))
                tr["W"].append(float(st.W[k]))
            idx[k] = j
        if self.mode == "force":
            state = st.aware.astype(np.int8)
        elif self.mode == "rule":
            state = np.array([YIELDING if (vid != "EV"
                                           and self.reg[vid]["lc_saved"])
                              else UNAWARE for vid in ids], np.int8)
        else:
            state = np.full(len(ids), UNAWARE, np.int8)
        tr["t"].append(float(t))
        tr["corr_y"].append(float(corr.y_c))
        tr["frames"].append((idx, st.x.astype(np.float32),
                             st.y.astype(np.float32),
                             st.vx.astype(np.float32), state))

    def _metrics(self, rec, collisions, ev_gone_at) -> dict:
        t = np.array(rec["t"]); x = np.array(rec["ev_x"]); v = np.array(rec["ev_v"])
        m = dict(mode=self.mode, seed=self.seed, collisions=int(collisions),
                 ev_v_trace=rec["ev_v"], t_trace=rec["t"],
                 clearance_trace=rec["clearance"], n_yield_trace=rec["n_yield"],
                 bg_v_trace=rec["bg_v"], ego_frames=rec["ego"])
        inside = (x >= SEG[0]) & (x <= SEG[1])
        if inside.sum() > 5:
            t0, t1 = t[inside][0], t[inside][-1]
            d = x[inside][-1] - x[inside][0]
            m["seg_time"] = float(t1 - t0)
            m["seg_speed"] = float(d / max(t1 - t0, 1e-6))
            m["seg_complete"] = bool(x[inside][-1] > SEG[1] - 60)
            m["clearance_mean"] = float(np.mean(np.array(rec["clearance"])[inside]))
        m["ev_mean_speed"] = float(v.mean()) if v.size else float("nan")
        m["ev_finished_at"] = ev_gone_at
        if self.traj is not None:
            m["traj"] = self.traj
        return m
