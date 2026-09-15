"""Wire protocol + coordinate contract for the UE5.8 <-> Python bridge.

Transport: TCP, newline-delimited JSON (one message per '\\n'). Python is the
server, UE the client. Everything on the wire is in the **emv road frame** in
SI units (metres, m/s, radians) so the Python side stays engine-agnostic; the
Unreal side does the unit/axis conversion (see emv_to_ue / ue_to_emv below,
which are the reference for the C++ implementation in EmvBridgeComponent).

emv road frame
    x  metres along the road (forward, +x = driving direction)
    y  metres lateral; lane 0 (rightmost) center = 1.75, middle = 5.25,
       left lane = 8.75; increasing y = toward the driver's LEFT.
    yaw  radians, heading measured from +x toward +y (atan2(vy, vx)).

Messages
    UE  -> Python : {"t": <s>, "seq": <int>,
                     "ev": {"x","y","vx","vy","yaw"}}
    Python -> UE  : {"t": <s>, "seq": <int>, "reset": <bool>,
                     "npc": [{"id","x","y","yaw","vx","s"}, ...]}
        s = awareness state (0 unaware, 1 noticed, 2 yielding, 3 hold)
        reset = true on the frame where the scenario was re-initialised
                (UE should drop its actor pool and respawn).
"""
import json
import math

#: middle-lane center in the emv frame; the UE highway is laid out so this maps
#: to UE world Y = 0 (the road is centred on the middle lane).
Y_CENTER = 5.25
#: metres -> Unreal centimetres.
M_TO_CM = 100.0


# --------------------------------------------------------------- (de)serialise
def encode(msg: dict) -> bytes:
    """Serialise one message to a newline-terminated JSON byte string."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: str | bytes) -> dict:
    """Parse one JSON message (without the trailing newline)."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return json.loads(line)


def npc_frame(t: float, seq: int, npcs: list[dict], reset: bool = False,
              vcap: float | None = None) -> bytes:
    """`vcap` (m/s, optional) is a speed ceiling the EV should not exceed - the
    SUMO modes send their car-following safe speed so the game can govern the
    player's throttle instead of letting them drive into a leader. Absent or
    negative = no cap."""
    msg = {"t": round(float(t), 4), "seq": int(seq),
           "reset": bool(reset), "npc": npcs}
    if vcap is not None:
        msg["vcap"] = round(float(vcap), 3)
    return encode(msg)


def ev_message(t: float, seq: int, x: float, y: float,
               vx: float, vy: float, yaw: float) -> bytes:
    """Build a UE->Python EV message (used by tests / mock clients)."""
    return encode({"t": round(float(t), 4), "seq": int(seq),
                   "ev": {"x": round(float(x), 4), "y": round(float(y), 4),
                          "vx": round(float(vx), 4), "vy": round(float(vy), 4),
                          "yaw": round(float(yaw), 5)}})


# ------------------------------------------------------- coordinate transform
# Reference implementation of the emv<->UE mapping. Authoritative copy lives in
# the UE C++ (EmvBridgeComponent); kept here so it can be unit-tested and so the
# two never drift. UE is left-handed with +Y to the right of +X, centimetres.
def emv_to_ue(x: float, y: float, yaw: float) -> tuple[float, float, float]:
    """emv (m, m, rad) -> UE (cm, cm, degrees)."""
    ue_x = x * M_TO_CM
    ue_y = -(y - Y_CENTER) * M_TO_CM        # emv +y (left) -> UE -Y
    ue_yaw = -math.degrees(yaw)             # sign flips with the Y axis
    return ue_x, ue_y, ue_yaw


def ue_to_emv(ue_x: float, ue_y: float, ue_yaw: float) -> tuple[float, float, float]:
    """UE (cm, cm, degrees) -> emv (m, m, rad)."""
    x = ue_x / M_TO_CM
    y = -ue_y / M_TO_CM + Y_CENTER
    yaw = -math.radians(ue_yaw)
    return x, y, yaw


def emv_to_ue_vel(vx: float, vy: float) -> tuple[float, float]:
    """Velocity transforms like position (Y sign-flipped), m/s -> cm/s."""
    return vx * M_TO_CM, -vy * M_TO_CM
