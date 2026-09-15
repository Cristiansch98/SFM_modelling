"""Offline checks for the UE5.8 bridge: wire round-trip, coordinate round-trip,
and one co-simulation step with a clamped (human-driven) EV. No sockets, no UE.

    python tests/test_ue_bridge.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from emv.ue import protocol
from emv.ue.server import UEBridge


def test_protocol_roundtrip():
    npcs = [dict(id=3, x=123.456, y=5.25, yaw=0.12, vx=27.7, s=2)]
    raw = protocol.npc_frame(1.5, 42, npcs, reset=False)
    assert raw.endswith(b"\n")
    msg = protocol.decode(raw.rstrip(b"\n"))
    assert msg["seq"] == 42 and msg["reset"] is False
    assert msg["npc"][0]["id"] == 3 and msg["npc"][0]["s"] == 2

    ev = protocol.ev_message(2.0, 7, 100.0, 7.0, 25.0, -0.5, 0.1)
    m = protocol.decode(ev.rstrip(b"\n"))
    assert m["ev"]["x"] == 100.0 and m["ev"]["vy"] == -0.5
    print("ok  protocol round-trip")


def test_coordinate_roundtrip():
    # middle lane maps to UE Y = 0; right lane (y=1.75) is to the UE +Y (right)
    assert abs(protocol.emv_to_ue(0.0, 5.25, 0.0)[1]) < 1e-9
    assert protocol.emv_to_ue(0.0, 1.75, 0.0)[1] > 0.0     # rightmost -> +Y
    assert protocol.emv_to_ue(0.0, 8.75, 0.0)[1] < 0.0     # leftmost  -> -Y

    for x, y, yaw in [(0.0, 1.75, 0.0), (1234.5, 8.75, 0.3),
                      (30.0, 7.0, -0.25), (2600.0, 5.25, 1.1)]:
        ux, uy, uyaw = protocol.emv_to_ue(x, y, yaw)
        x2, y2, yaw2 = protocol.ue_to_emv(ux, uy, uyaw)
        assert abs(x - x2) < 1e-6 and abs(y - y2) < 1e-6 and abs(yaw - yaw2) < 1e-6
    print("ok  coordinate round-trip (emv->UE->emv)")


def test_step_clamps_ev():
    b = UEBridge(seed=7, params="surrogate", window=400.0)
    b._build()
    e = b.sim.st.ev
    # drive the EV to a chosen state and confirm the model does not move it
    ev = dict(x=500.0, y=7.0, vx=30.0, vy=0.2)
    npcs, reset = b.step(ev)
    st = b.sim.st
    assert st.x[e] == 500.0 and st.y[e] == 7.0
    assert st.vx[e] == 30.0 and st.vy[e] == 0.2
    assert reset is False
    # every NPC reported is within the window and is not the EV
    assert len(npcs) > 0
    for v in npcs:
        assert v["id"] != int(b.ids[e])
        assert abs(v["x"] - ev["x"]) <= b.window + 1e-6
        assert 0 <= v["s"] <= 3
    # NPCs ahead should begin to react over a few seconds as the EV closes in
    for _ in range(200):                      # ~12 s at dt=0.06
        ev["x"] += 30.0 * b.p.dt              # EV advancing at 30 m/s
        b.step(ev)
    aware = b.sim.st.aware[~b.sim.st.is_ev]
    assert (aware > 0).any(), "no NPC ever became aware of the EV"
    print(f"ok  co-sim step (EV clamped; {int((aware>0).sum())} NPCs reacting)")


if __name__ == "__main__":
    test_protocol_roundtrip()
    test_coordinate_roundtrip()
    test_step_clamps_ev()
    print("\nALL UE-BRIDGE TESTS PASSED")
