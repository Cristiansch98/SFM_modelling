"""UEBridge: the Python "brain" server for the UE5.8 first-person EV game.

Authority split (the mirror of emv/sumo/bridge.py): the human drives the EV in
Unreal, so UE is authoritative for the EV; the social-force model is
authoritative for every surrounding car. Each brain step:

    1. inject the latest EV kinematics (from UE) into the EV row of the state,
    2. run one Sim.step (builds the predicted corridor + perception from the EV
       and integrates ALL vehicles with emv.dynamics),
    3. restore the EV row to the UE-supplied kinematics (the model's own EV
       integration is discarded - the human stays authoritative),
    4. emit the NPC transforms within a window of the EV.

The NPC physics are byte-for-byte the calibrated surrogate that produced
out/anim_surrogate_sumo.gif (params default to the fitted theta). The `mock`
mode replays the surrogate's own EV path through this exact loop with no socket,
so the coupling can be verified offline against out/surrogate_sumo.json.
"""
import math
import os
import socket
import time

import numpy as np

from ..params import Params, tuned_params
from ..scenarios import make_sumo_like, surrogate_params, game_params
from ..simulate import History
from ..metrics import evaluate
from . import protocol

# scenario constants (must match emv.scenarios.make_sumo_like)
N_LANES, LANE_W, EDGE_LEN, V_MAX = 3, 3.5, 3000.0, 27.78
FLOW = 1700.0
SEG = (400.0, 2600.0)               # EV measurement segment (parity with surrogate)


def _make_params(which: str) -> Params:
    if which == "game":
        return game_params()
    if which == "surrogate":
        return surrogate_params()
    if which == "tuned":
        return tuned_params()
    if which == "default":
        return Params()
    raise ValueError(f"unknown params preset: {which!r}")


class UEBridge:
    def __init__(self, seed: int = 7, params: str = "surrogate",
                 window: float = 600.0, port: int = 7777,
                 host: str = "127.0.0.1", endless: bool = False,
                 hz: float | None = None):
        self.seed = seed
        self.params_name = params
        self.p = _make_params(params)
        # Brain tick rate == the model's integration step. Raising it shortens
        # the age of the newest frame UE can show (and lets the NPC manager run
        # a smaller interpolation delay); semi-implicit Euler only gets more
        # accurate as dt shrinks. Cost is ~5 ms/step regardless of dt.
        if hz:
            self.p = self.p.copy(dt=1.0 / float(hz))
        self.window = float(window)
        self.port = int(port)
        self.host = host
        self.endless = endless
        v_mean = 0.92 * V_MAX
        self.spacing = 1000.0 / (FLOW / (v_mean * 3.6))
        self.rng = np.random.default_rng(seed + 104729)
        self.sim = None
        self.ids = None
        self._id_counter = 0
        self.live = False                 # True while a UE client is connected

    # ------------------------------------------------------------------ setup
    def _build(self):
        self.sim = make_sumo_like(self.seed, self.p)
        n = self.sim.st.n
        self.ids = np.arange(n, dtype=np.int64)   # id == state index; 0 is the EV
        self._id_counter = n

    def _ev_default(self) -> dict:
        st = self.sim.st
        e = st.ev
        return dict(x=float(st.x[e]), y=float(st.y[e]),
                    vx=float(st.vx[e]), vy=float(st.vy[e]), yaw=0.0)

    # ------------------------------------------------------------------- step
    def step(self, ev: dict) -> tuple[list[dict], bool]:
        """Advance NPCs one dt with the EV clamped to the UE-supplied state.
        Returns (npc_list, reset_flag)."""
        st = self.sim.st
        e = st.ev
        st.x[e], st.y[e] = float(ev["x"]), float(ev["y"])
        st.vx[e], st.vy[e] = float(ev.get("vx", 0.0)), float(ev.get("vy", 0.0))
        self.sim.step()                                   # corridor+perception+integrate
        # discard the model's EV integration; the human is authoritative
        st.x[e], st.y[e] = float(ev["x"]), float(ev["y"])
        st.vx[e], st.vy[e] = float(ev.get("vx", 0.0)), float(ev.get("vy", 0.0))
        if self.endless:
            self._recycle()
        return self._npc_list(), False

    def _npc_list(self) -> list[dict]:
        st = self.sim.st
        e = st.ev
        dx = st.x - st.x[e]
        keep = (~st.is_ev) & (np.abs(dx) <= self.window)
        out = []
        for i in np.flatnonzero(keep):
            out.append(dict(
                id=int(self.ids[i]),
                x=round(float(st.x[i]), 3), y=round(float(st.y[i]), 3),
                yaw=round(float(math.atan2(st.vy[i], st.vx[i])), 5),
                vx=round(float(st.vx[i]), 3), s=int(st.aware[i])))
        return out

    def _recycle(self):
        """Endless mode: NPCs that fall a full window behind the EV are respawned
        ahead with fresh desired speed + cleared awareness and a NEW id (so UE
        spawns a fresh actor rather than lerping across the teleport)."""
        st = self.sim.st
        e = st.ev
        ex = st.x[e]
        behind = np.flatnonzero((~st.is_ev) & (st.x < ex - self.window))
        for i in behind:
            lane = int(self.rng.integers(0, N_LANES))
            st.x[i] = ex + self.window + float(self.rng.uniform(5.0, self.spacing))
            st.y[i] = (lane + 0.5) * LANE_W + float(self.rng.uniform(-0.25, 0.25))
            f = float(np.clip(self.rng.normal(0.92, 0.08), 0.6, 1.2))
            st.v0[i] = f * V_MAX
            st.vx[i] = st.v0[i] * 0.97
            st.vy[i] = 0.0
            st.aware[i] = 0
            st.side[i] = 0
            st.t_react[i] = np.inf
            st.hold_until[i] = np.inf
            st.u[i] = 0.0
            st.y_init[i] = st.y[i]
            self.ids[i] = self._id_counter
            self._id_counter += 1

    # ------------------------------------------------------------------ serve
    def serve(self, duration: float | None = None,
              on_listening=None, on_step=None, keep_alive: bool = True) -> dict:
        """Accept one UE client and run the real-time co-simulation loop.

        `on_listening` (if given) is called once the socket is accepting, i.e.
        the earliest moment a UE client can connect - start_demo.py uses it to
        launch Unreal only after the brain is reachable.
        `on_step` (if given) is called with self after every step; it runs
        inside the real-time budget, so it must be cheap (the live view uses it
        to copy state into shared memory for its own process).
        `keep_alive` keeps the brain up across PIE sessions: stopping Play in
        Unreal tears down the game instance and with it the socket, so without
        this every Play/Stop cycle would need the brain restarted by hand."""
        if self.sim is None:               # caller may have built it already
            self._build()
        latest_ev = self._ev_default()
        dt = self.p.dt

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[ue] brain ready ({self.params_name} params, seed {self.seed}, "
              f"{self.sim.st.n - 1} NPCs, dt={dt:.3f}s)", flush=True)
        print(f"[ue] waiting for UE to connect on {self.host}:{self.port} ...",
              flush=True)
        print("[ue] (first connection may trigger a Windows Firewall prompt - "
              "click Allow)", flush=True)
        if on_listening is not None:
            try:
                on_listening()
            except Exception as exc:
                print(f"[ue] on_listening failed: {exc!r}", flush=True)
        seq = 0
        steps = 0
        ev_speeds = []
        sessions = 0
        t_wall0 = time.perf_counter()
        try:
          while True:
            conn, addr = srv.accept()
            sessions += 1
            print(f"[ue] UE connected from {addr[0]}:{addr[1]}"
                  f"{f' (session {sessions})' if sessions > 1 else ''}", flush=True)
            conn.setblocking(False)
            # A ~2 KB NPC frame straddles two TCP segments; Nagle holds the
            # trailing partial one until the first is ACKed, adding a
            # delayed-ACK period of latency and jitter to every frame. This is
            # a real-time stream - send each frame the moment it exists.
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.live = True
            buf = b""
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

                npcs, reset = self.step(latest_ev)
                seq += 1
                steps += 1
                if on_step is not None:
                    on_step(self)
                ev_speeds.append(float(self.sim.st.vx[self.sim.st.ev]))
                try:
                    conn.sendall(protocol.npc_frame(self.sim.t, seq, npcs, reset))
                except ConnectionError:
                    print("[ue] UE closed the connection", flush=True)
                    break

                next_t += dt
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_t = time.perf_counter()          # fell behind; catch up
            # ConnectionError covers Reset/Aborted/BrokenPipe - Windows raises
            # ConnectionAbortedError on a normal client close, not Reset.
            except ConnectionError as exc:
                print(f"[ue] session ended: {type(exc).__name__}", flush=True)
            finally:
                self.live = False
                try:
                    conn.close()
                except Exception:
                    pass

            if not keep_alive or (duration is not None and self.sim.t >= duration):
                break
            print("[ue] brain still up - press Play again to reconnect "
                  "(Ctrl-C here to stop)", flush=True)
        except KeyboardInterrupt:
            print("[ue] interrupted", flush=True)
        finally:
            self.live = False
            srv.close()

        wall = time.perf_counter() - t_wall0
        return dict(steps=steps, sim_time=float(self.sim.t), wall_s=round(wall, 2),
                    ev_mean_kmh=round(3.6 * float(np.mean(ev_speeds)), 1)
                    if ev_speeds else 0.0)

    # ------------------------------------------------------------------- mock
    def mock(self, duration: float | None = None,
             out_gif: str | None = None) -> dict:
        """No-socket self-test: replay the surrogate's own EV path through the
        exact inject/step/restore loop, record a History, optionally render it,
        and report parity metrics. If the coupling is correct the NPC evolution
        reproduces the surrogate (out/surrogate_sumo.json) frame-for-frame."""
        T = duration if duration is not None else 115.0
        stop_x = SEG[1] + 60.0
        ref = make_sumo_like(self.seed, self.p)
        hist_ref = ref.run(T, rec_dt=self.p.dt, stop_when_ev_x=stop_x)

        self._build()
        st = self.sim.st
        e = st.ev
        ev_col = hist_ref.ev
        rec = {k: [] for k in ("t", "x", "y", "vx", "vy", "ax",
                               "aware", "u", "side")}
        for k in range(hist_ref.n_frames):
            st.x[e] = hist_ref.x[k, ev_col]
            st.y[e] = hist_ref.y[k, ev_col]
            st.vx[e] = hist_ref.vx[k, ev_col]
            st.vy[e] = hist_ref.vy[k, ev_col]
            rec["t"].append(self.sim.t)
            rec["x"].append(st.x.copy()); rec["y"].append(st.y.copy())
            rec["vx"].append(st.vx.copy()); rec["vy"].append(st.vy.copy())
            rec["ax"].append(np.zeros(st.n))
            rec["aware"].append(st.aware.copy())
            rec["u"].append(st.u.astype(np.float32))
            rec["side"].append(st.side.copy())
            self.sim.step()

        hist = History(
            t=np.array(rec["t"]), x=np.array(rec["x"]), y=np.array(rec["y"]),
            vx=np.array(rec["vx"]), vy=np.array(rec["vy"]), ax=np.array(rec["ax"]),
            aware=np.array(rec["aware"]), u=np.array(rec["u"]),
            side=np.array(rec["side"]),
            L=st.L.copy(), W=st.W.copy(), v0=st.v0.copy(), ev=e,
            y_corr=self.sim.y_corr, road=self.sim.road, params=self.p,
            name=f"ue_mock_s{self.seed}")

        seg_kmh, clearance, collisions = self._segment_metrics(hist)
        rseg_kmh, rclear, rcoll = self._segment_metrics(hist_ref)
        out = dict(seg_kmh=round(seg_kmh, 1), clearance=round(clearance, 1),
                   collisions=collisions, ref_seg_kmh=round(rseg_kmh, 1),
                   ref_clearance=round(rclear, 1), ref_collisions=rcoll,
                   n_frames=hist.n_frames)
        if out_gif:
            from .. import viz
            viz.animate(hist, out_gif,
                        title="UE bridge (mock EV replay) - NPC brain parity")
            out["gif"] = out_gif
        return out

    def _segment_metrics(self, hist: History) -> tuple[float, float, int]:
        m = evaluate(hist)
        x_ev = hist.x[:, hist.ev]
        inside = (x_ev >= SEG[0]) & (x_ev <= SEG[1])
        if inside.sum() > 5:
            seg_kmh = 3.6 * float((x_ev[inside][-1] - x_ev[inside][0])
                                  / max(hist.t[inside][-1] - hist.t[inside][0], 1e-6))
            clr = float(np.mean(m["clearance_trace"][inside]))
        else:
            seg_kmh, clr = 0.0, 0.0
        return seg_kmh, clr, int(m["collisions"])
