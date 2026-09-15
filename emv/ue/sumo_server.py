"""SUMO-backed brain for the UE game: the *baseline* conditions.

Same wire protocol and the same real-time contract as emv.ue.server.UEBridge,
but the NPCs are SUMO vehicles instead of force-model particles. You still drive
the EV in Unreal; its pose is written into SUMO with moveToXY each step, so
SUMO's own logic decides how traffic responds.

    mode="bluelight"  SUMO's native rescue-lane device (the baseline of [17]:
                      hard-coded special rights, reaction distance 100 m)
    mode="none"       no yielding behaviour at all - the control condition

This is what makes the demo a comparison rather than a tech demo: the same road,
the same density, the same car you drive, with the *only* difference being who
decides how traffic reacts.

Awareness colours are a display heuristic here (see _display_states): SUMO's
bluelight device exposes no per-driver state, so unlike the force model there is
no ground truth to colour by. The geometry - who moved aside, and when - is real.
"""
import math
import os
import socket
import time

import numpy as np

from ..params import Params
from ..road import Road
from ..state import blank_state, VehState, UNAWARE, NOTICED, YIELDING, HOLD
from ..scenarios import game_params
from ..sumo import make_scenario as sc
from . import protocol

EV_ID = "EV"
N_CAP = 400                 # fixed state capacity (the live view needs it fixed)
REACT_DIST = 100.0          # m, bluelight device reaction distance
WARMUP_S = 90.0             # s of traffic to build before the EV joins


class _SimShim:
    """Just enough of emv.simulate.Sim for the live 2D view to draw us."""

    def __init__(self, st: VehState, road: Road, y_corr: float):
        self.st = st
        self.road = road
        self.y_corr = y_corr
        self.t = 0.0


class SumoUEBridge:
    def __init__(self, mode: str = "bluelight", seed: int = 7,
                 window: float = 600.0, port: int = 7777,
                 host: str = "127.0.0.1", hz: float | None = None,
                 out_dir: str = "out/sumo_ue", warmup: float = WARMUP_S,
                 **_ignored):
        if mode not in ("bluelight", "none"):
            raise ValueError(f"unknown SUMO mode: {mode!r}")
        self.mode = mode
        self.params_name = mode
        self.seed = int(seed)
        self.window = float(window)
        self.port = int(port)
        self.host = host
        self.warmup = float(warmup)
        self.out_dir = out_dir
        # Params only supplies dt and the corridor geometry the view draws.
        self.p = game_params()
        if hz:
            self.p = self.p.copy(dt=1.0 / float(hz))
        self.sim = None
        self.live = False
        self.ids = None
        self._conn = None
        self._y0 = 0.0        # network y of lane-0 centre
        self._dy = 1.0        # network y per +1 m of emv y
        # Stable small integer per SUMO vehicle id. Hashing the string would
        # alias two vehicles onto one UE actor whenever the hashes collide.
        self._vid_ids: dict[str, int] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ setup
    def _build(self):
        road = Road(n_lanes=sc.N_LANES, lane_width=sc.LANE_W,
                    shoulder_right=0.0, shoulder_left=0.0, length=sc.EDGE_LEN)
        d = blank_state(N_CAP)
        st = VehState(**d)
        st.is_ev[0] = True
        st.L[0], st.W[0] = 6.2, 2.2
        st.x[:] = -1.0e6              # park unused slots far off-screen
        st.x[0] = 30.0
        st.y[0] = 7.0
        self.sim = _SimShim(st, road, y_corr=7.0)
        self.ids = np.arange(N_CAP, dtype=np.int64)

    def _start_sumo(self):
        import traci
        cfg = sc.write_scenario(self.out_dir)
        cmd = [sc.sumo_bin("sumo"), "-c", cfg, "--seed", str(self.seed),
               "--start", "--step-length", f"{self.p.dt:.4f}",
               "--lateral-resolution", "0.4", "--no-warnings", "true"]
        if self.mode == "bluelight":
            # the device must exist at insertion; assign it explicitly by id
            cmd += ["--device.bluelight.explicit", EV_ID,
                    "--device.bluelight.reactiondist", str(int(REACT_DIST))]
        traci.start(cmd, label=f"emv_ue_{self.mode}_{self.seed}")
        self._conn = traci

        # Self-calibrate emv-y -> network-y instead of assuming SUMO's lane
        # layout convention: read two lane centres and interpolate.
        p0 = traci.simulation.convert2D("hw", 100.0, 0)
        p1 = traci.simulation.convert2D("hw", 100.0, 1)
        self._y0 = p0[1]
        self._dy = (p1[1] - p0[1]) / sc.LANE_W

        n_steps = int(self.warmup / self.p.dt)
        print(f"[sumo] warming up {self.warmup:.0f} s of traffic "
              f"({n_steps} steps) ...", flush=True)
        for _ in range(n_steps):
            traci.simulationStep()
        print(f"[sumo] {len(traci.vehicle.getIDList())} vehicles on the road",
              flush=True)

        traci.vehicle.add(EV_ID, "r0", typeID="ev", departLane="1",
                          departSpeed="0", departPos="20")
        traci.simulationStep()
        # The human owns the EV completely: stop SUMO steering or braking it.
        traci.vehicle.setSpeedMode(EV_ID, 0)
        traci.vehicle.setLaneChangeMode(EV_ID, 0)
        n = self._clear_zone(30.0, 7.0)      # the UE PlayerStart, in emv metres
        print(f"[sumo] cleared {n} vehicles from the player's spawn box",
              flush=True)

    def _clear_zone(self, x: float, y: float, half_len: float = 20.0,
                    half_wid: float = 5.0) -> int:
        """Remove SUMO vehicles occupying the player's spawn box.

        The warm-up fills the road from x=0, so traffic is sitting exactly where
        the UE pawn spawns. An NPC actor posed inside the pawn is a kinematic
        body intersecting a dynamic one, and Chaos resolves that by launching
        the car into the air. Clearing the box before the player appears is the
        cheap, robust fix; SUMO refills the gap within seconds.
        """
        traci = self._conn
        removed = 0
        for vid in list(traci.vehicle.getIDList()):
            if vid == EV_ID:
                continue
            try:
                lp = traci.vehicle.getLanePosition(vid)
                if abs(lp - x) > half_len:
                    continue
                ly = ((traci.vehicle.getLaneIndex(vid) + 0.5) * sc.LANE_W
                      + traci.vehicle.getLateralLanePosition(vid))
                if abs(ly - y) > half_wid:
                    continue
                traci.vehicle.remove(vid)
                removed += 1
            except traci.TraCIException:
                continue
        return removed

    # ------------------------------------------------------------- coordinates
    def _emv_to_net(self, x: float, y: float) -> tuple[float, float]:
        return x, self._y0 + (y - 0.5 * sc.LANE_W) * self._dy

    # --------------------------------------------------------------- one step
    def _push_ev(self, ev: dict):
        traci = self._conn
        x, y = float(ev["x"]), float(ev["y"])
        lane = int(np.clip(int(y // sc.LANE_W), 0, sc.N_LANES - 1))
        nx, ny = self._emv_to_net(x, y)
        yaw = float(ev.get("yaw", 0.0))
        angle = (90.0 - math.degrees(yaw)) % 360.0     # SUMO: 0 = north, cw
        try:
            traci.vehicle.moveToXY(EV_ID, "hw", lane, nx, ny, angle=angle,
                                   keepRoute=2)
            traci.vehicle.setSpeed(EV_ID, max(float(ev.get("vx", 0.0)), 0.0))
        except traci.TraCIException:
            pass

    def _safe_speed(self, ev: dict) -> float:
        """SUMO's own car-following safe speed for the EV, given its leader.

        The player is authoritative for position, so SUMO cannot brake the EV
        itself - moveToXY would just teleport it into the leader and log a
        collision. Instead we ask SUMO what speed *would* be safe and hand that
        back to the game as a ceiling, so the pawn brakes itself. This mirrors
        the force model's arrangement, where the IDM layer stays authoritative
        over the social force. Returns a large number when the road is clear.
        """
        traci = self._conn
        v = max(float(ev.get("vx", 0.0)), 0.0)
        try:
            lead = traci.vehicle.getLeader(EV_ID, 250.0)
            if not lead:
                return 1.0e3
            lead_id, gap = lead
            if gap is None or gap < 0.0:
                gap = 0.0
            lead_v = traci.vehicle.getSpeed(lead_id)
            return float(traci.vehicle.getFollowSpeed(EV_ID, v, gap, lead_v,
                                                      self.p.b_emerg))
        except Exception:
            return 1.0e3

    def _display_states(self, x, y, vx, v0, x_ev) -> np.ndarray:
        """Heuristic colouring - SUMO exposes no per-driver yield state.
        noticed  = inside the device's reaction distance ahead of the EV
        yielding = also displaced from its lane centre or notably slowed
        hold     = displaced, but the EV is already past
        """
        s = x - x_ev
        lane_c = (np.floor(np.clip(y, 0.0, sc.N_LANES * sc.LANE_W - 1e-6)
                           / sc.LANE_W) + 0.5) * sc.LANE_W
        displaced = np.abs(y - lane_c) > 0.45
        slowed = vx < 0.85 * np.maximum(v0, 1.0)
        out = np.zeros(x.size, dtype=np.int8)
        out[(s > 0) & (s < REACT_DIST)] = NOTICED
        out[(s > 0) & (s < REACT_DIST) & (displaced | slowed)] = YIELDING
        out[(s <= 0) & (s > -60.0) & displaced] = HOLD
        return out

    def _read(self, ev: dict) -> list[dict]:
        """Pull SUMO's world into the shim state and build the UE frame."""
        traci = self._conn
        st = self.sim.st
        x_ev, y_ev = float(ev["x"]), float(ev["y"])
        st.x[0], st.y[0] = x_ev, y_ev
        st.vx[0] = float(ev.get("vx", 0.0))
        st.vy[0] = float(ev.get("vy", 0.0))

        ids = [v for v in traci.vehicle.getIDList() if v != EV_ID]
        xs, ys, vs, v0s, ls, ws, keep = [], [], [], [], [], [], []
        for vid in ids:
            try:
                lp = traci.vehicle.getLanePosition(vid)
                if abs(lp - x_ev) > self.window:
                    continue
                li = traci.vehicle.getLaneIndex(vid)
                lat = traci.vehicle.getLateralLanePosition(vid)
                xs.append(lp)
                ys.append((li + 0.5) * sc.LANE_W + lat)
                vs.append(traci.vehicle.getSpeed(vid))
                v0s.append(traci.vehicle.getAllowedSpeed(vid))
                ls.append(traci.vehicle.getLength(vid))
                ws.append(traci.vehicle.getWidth(vid))
                keep.append(vid)
            except traci.TraCIException:
                continue

        n = min(len(keep), N_CAP - 1)
        st.x[1:] = -1.0e6                       # park everything, then fill
        st.aware[1:] = UNAWARE
        out = []
        if n:
            X = np.asarray(xs[:n]); Y = np.asarray(ys[:n])
            V = np.asarray(vs[:n]); V0 = np.asarray(v0s[:n])
            S = self._display_states(X, Y, V, V0, x_ev)
            sl = slice(1, 1 + n)
            st.x[sl], st.y[sl], st.vx[sl] = X, Y, V
            st.L[sl] = np.asarray(ls[:n]); st.W[sl] = np.asarray(ws[:n])
            st.aware[sl] = S
            for k in range(n):
                vid = keep[k]
                if vid not in self._vid_ids:
                    self._vid_ids[vid] = self._next_id
                    self._next_id += 1
                out.append(dict(
                    id=self._vid_ids[vid],
                    x=round(float(X[k]), 3), y=round(float(Y[k]), 3),
                    yaw=0.0, vx=round(float(V[k]), 3), s=int(S[k])))
        return out

    # ------------------------------------------------------------------ serve
    def serve(self, duration: float | None = None,
              on_listening=None, on_step=None, keep_alive: bool = True) -> dict:
        import traci
        if self.sim is None:
            self._build()

        latest_ev = dict(x=30.0, y=7.0, vx=0.0, vy=0.0, yaw=0.0)
        dt = self.p.dt

        # Listen (and let Unreal start) BEFORE the SUMO warm-up: the warm-up
        # takes tens of seconds and the editor takes about as long to load, so
        # running them concurrently hides it. Early EV frames just queue in the
        # socket buffer; the recv loop keeps only the newest.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[sumo] listening on {self.host}:{self.port} (mode={self.mode}, "
              f"seed {self.seed}, dt={dt:.3f}s)", flush=True)
        if on_listening is not None:
            try:
                on_listening()
            except Exception as exc:
                print(f"[sumo] on_listening failed: {exc!r}", flush=True)

        self._start_sumo()
        print("[sumo] waiting for UE to connect ...", flush=True)

        seq = steps = sessions = 0
        ev_speeds = []
        t_wall0 = time.perf_counter()
        try:
          while True:
            conn, addr = srv.accept()
            sessions += 1
            print(f"[sumo] UE connected from {addr[0]}:{addr[1]}"
                  f"{f' (session {sessions})' if sessions > 1 else ''}", flush=True)
            conn.setblocking(False)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.live = True
            buf = b""
            # Every Play respawns the pawn at the PlayerStart, so the spawn box
            # has to be cleared again for each session, not just at startup.
            self._clear_zone(30.0, 7.0)
            next_t = time.perf_counter()

            try:
              while duration is None or self.sim.t < duration:
                try:
                    while True:
                        data = conn.recv(65536)
                        if not data:
                            raise ConnectionResetError("UE closed the connection")
                        buf += data
                except BlockingIOError:
                    pass
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = protocol.decode(line)
                    except Exception:
                        continue
                    if "ev" in msg:
                        latest_ev = msg["ev"]

                self._push_ev(latest_ev)
                traci.simulationStep()
                self.sim.t += dt
                npcs = self._read(latest_ev)
                seq += 1
                steps += 1
                ev_speeds.append(float(latest_ev.get("vx", 0.0)))
                if on_step is not None:
                    on_step(self)
                try:
                    conn.sendall(protocol.npc_frame(self.sim.t, seq, npcs, False,
                                                    vcap=self._safe_speed(latest_ev)))
                except ConnectionError:
                    print("[sumo] UE closed the connection", flush=True)
                    break

                next_t += dt
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_t = time.perf_counter()
            # ConnectionError covers Reset/Aborted/BrokenPipe - Windows raises
            # ConnectionAbortedError on a normal client close, not Reset.
            except ConnectionError as exc:
                print(f"[sumo] session ended: {type(exc).__name__}", flush=True)
            finally:
                self.live = False
                try:
                    conn.close()
                except Exception:
                    pass

            if not keep_alive or (duration is not None and self.sim.t >= duration):
                break
            print("[sumo] brain still up - press Play again to reconnect "
                  "(Ctrl-C here to stop)", flush=True)
        except KeyboardInterrupt:
            print("[sumo] interrupted", flush=True)
        finally:
            self.live = False
            srv.close()
            try:
                traci.close()
            except Exception:
                pass

        wall = time.perf_counter() - t_wall0
        return dict(steps=steps, sim_time=float(self.sim.t),
                    wall_s=round(wall, 2),
                    ev_mean_kmh=round(3.6 * float(np.mean(ev_speeds)), 1)
                    if ev_speeds else 0.0)
