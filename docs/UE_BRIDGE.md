# UE5.8 first-person EV game — bridge & setup

A first-person driving game where **you drive the emergency vehicle** and the
surrounding traffic is puppeteered by the physics-inspired force model that
produced `out/anim_surrogate_sumo.gif`. Unreal Engine 5.8 is the renderer + the
drivable EV; a small Python server (`emv/ue/`) is the authoritative "brain" for
every NPC. They talk over a local TCP socket.

```
  UE5.8 game (C++)                              Python 3.13 (this repo)
  EmergencyVehiclePawn  --- EV state (emv m) --> UEBridge (emv/ue/server.py)
  NpcManager/Actors     <-- NPC frame (emv m) -- forces + perception + corridor
```

The EV row lives in the model state so NPCs react to the real player, but its
kinematics are overwritten from UE every step (the model's own EV integration is
discarded). This is the exact mirror of `emv/sumo/bridge.py`, with authority
reversed: UE owns the EV, Python owns the NPCs.

---

## 1. Wire protocol (authoritative: `emv/ue/protocol.py`)

Transport: **TCP, newline-delimited JSON**. Python is the **server**, UE the
**client**. Everything on the wire is in the **emv road frame, SI units**
(metres, m/s, radians) so Python stays engine-agnostic; UE does the unit/axis
conversion.

**UE → Python** (each UE tick, throttled to ~60 Hz):
```json
{"t": 12.34, "seq": 811, "ev": {"x": 812.5, "y": 7.0, "vx": 30.1, "vy": -0.2, "yaw": -0.01}}
```

**Python → UE** (each brain step, ~16.7 Hz at dt = 0.06 s):
```json
{"t": 12.30, "seq": 205, "reset": false,
 "npc": [{"id": 42, "x": 845.1, "y": 5.2, "yaw": 0.0, "vx": 25.6, "s": 2}, ...]}
```
`s` = awareness (0 unaware · 1 noticed · 2 yielding · 3 hold). `reset` = true on
the frame after a scenario re-init (UE drops its actor pool). Only NPCs within
`--window` metres of the EV are streamed.

## 2. Coordinate frame (must match on both sides)

emv road frame: `x` metres forward, `y` metres lateral, **+y = driver's left**;
lane centres `y = 1.75 / 5.25 / 8.75` (right / middle / left). UE world is
centimetres, left-handed (+Y to the right of +X). Lay the highway **straight
along UE +X with the middle lane on Y = 0**:

```
UE_X_cm  =  emv_x * 100
UE_Y_cm  = -(emv_y - 5.25) * 100        # +y (left) -> -Y
UE_yaw_deg = -degrees(emv_yaw)
# velocity transforms like position (Y sign-flipped); cm/s <-> m/s via *100
```

C++ implementation: `EmvBridgeTypes.h` (`namespace EmvCoords`). Python reference
+ unit test: `emv/ue/protocol.py` (`emv_to_ue` / `ue_to_emv`), `tests/test_ue_bridge.py`.

---

## 3. Run it

**A. Start the brain (Python 3.13, this repo):**
```
python experiments/play_ue.py                 # 127.0.0.1:7777, surrogate params
python experiments/play_ue.py --seed 8 --window 500
python experiments/play_ue.py --endless        # continuous traffic (endless road)
```
Leave it printing `waiting for UE to connect ...`.

**B. Launch the UE game** (Play-In-Editor or packaged). It connects on start;
allow the one-time Windows Firewall prompt. Override the target with
`-emvhost=127.0.0.1 -emvport=7777` on the UE command line if needed.

**Offline self-test (no UE):** proves the co-sim loop reproduces the surrogate —
```
python experiments/play_ue.py --mock          # -> out/anim_ue_mock.gif
```
Expected parity with `out/surrogate_sumo.json` (seed 7: 100.5 km/h, clearance
11.7 m, 0 collisions). Also run `python tests/test_ue_bridge.py`.

---

## 4. Unreal editor assembly checklist

The C++ under `ue_game/EmvHighway/Source/EmvHighway/` is complete; these are the
GUI-only steps to assemble the game around it.

1. **Project**: open `EmvHighway.uproject` in UE5.8 (it will offer to build the
   C++ module — accept). Confirm the **ChaosVehiclesPlugin** and **EnhancedInput**
   plugins are enabled. If you started from Epic's *Vehicle* template you already
   have a working Chaos vehicle to borrow the mesh/physics-asset/wheels from.

2. **Highway level** (`HighwayLevel`): build a straight **3-lane, 3.5 m-lane,
   ~3 km** road along **+X** with the **middle lane centred on Y = 0**. Use a
   spline-mesh road or tiled road mesh; add ground planes / guardrails, a
   `DirectionalLight` + `SkyAtmosphere` + `SkyLight` + `ExponentialHeightFog` for
   a realistic look. Lane markings via a road material or decals. Note the road
   surface Z (cm).

3. **Emergency vehicle** (`BP_EmergencyVehicle`, parent = `AEmergencyVehiclePawn`):
   assign the skeletal mesh, physics asset and wheel setups (reuse the template
   vehicle, reskinned; add roof light-bar meshes with emissive materials).
   - Adjust `CockpitCamera` to the driver eye point; verify the chase camera.
   - Assign `SirenSound` (import any looping siren `.wav`).
   - Implement the `OnSirenToggled(bool)` event to flash the light-bar emissives
     (a material scalar + a looping timeline).

4. **Enhanced Input** assets, assigned on `BP_EmergencyVehicle`:
   `IMC_Emv` (mapping context) + actions — `IA_Throttle` (Axis1D: **W +1 =
   throttle / S −1 = brake+reverse**, one axis), `IA_Steer` (Axis1D: D +1 /
   A −1), `IA_Handbrake` (Digital, `Space`), `IA_Siren` (Digital, `L`),
   `IA_Camera` (Digital, `C`).

5. **NPC meshes**: place an `ANpcManager` in the level; fill its `NpcMeshes`
   array with a few car static meshes (scale to ≈ 4.5 × 1.8 m; the EV is
   6.2 × 2.2 m), set `RoadZCm` to the road surface height, and (optionally)
   `bTintByState` for a debug view. For state tint, give the NPC meshes a
   material with a `StateColor` vector parameter.

6. **Game mode**: create `BP_EmvGameMode` (parent = `AEmvGameMode`, which already
   sets the canvas HUD) and set its **DefaultPawnClass = BP_EmergencyVehicle**.
   Set it as the level's GameMode Override, and set a `PlayerStart` at the start
   of the highway: **UE X = 3000, Y = -175, Z = road surface + ~50, yaw = 0**
   (= emv x = 30 m, y = 7.0 m on the rescue-gap line, facing +X, like the
   surrogate EV).

7. The `UEmvBridgeSubsystem` auto-starts (GameInstanceSubsystem) — no placement
   needed. It connects, streams the EV up and NPCs down, and reconnects if the
   Python server restarts.

### Notes / tuning
- **dt is calibrated (0.06 s)** — do not change it; smoothness comes from the
  UE-side interpolation in `ANpcManager::Tick`, not a faster brain.
- **Performance**: only NPCs within `--window` (default ±600 m ⇒ ~60 cars) are
  streamed and pooled; raise/lower `--window` to trade visible traffic for FPS.
- **Finite vs endless**: the default 3 km scenario matches the GIF exactly;
  `--endless` recycles cars that fall behind (needs an endless/rebased UE road,
  otherwise the EV drives off the end of the mesh).
- **No-C++ fallback**: a Blueprint TCP-socket plugin can replace
  `UEmvBridgeSubsystem` while keeping the same JSON protocol above.
