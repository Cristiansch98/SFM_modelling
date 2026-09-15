# Brief for the UE5.8 agent — finish the "drive-the-EV" highway game

## Mission
Assemble a first-person driving game in **Unreal Engine 5.8**. The player drives
an **emergency vehicle** down a realistic 3-lane motorway; the surrounding
traffic is puppeteered by an external Python "brain" that connects over a local
TCP socket. **The C++ is already written** — your job is the visuals + level +
vehicle + input + asset wiring, and to make it all line up with the brain.

- UE project:  `C:\Users\cschr\Desktop\physics_behaviour_EMV\ue_game\EmvHighway\EmvHighway.uproject`
- Contract + full checklist: `C:\Users\cschr\Desktop\physics_behaviour_EMV\docs\UE_BRIDGE.md` (read this first)
- The brain (start it to test): from `C:\Users\cschr\Desktop\physics_behaviour_EMV` run
  `python experiments/play_ue.py`  (listens on `127.0.0.1:7777`)

## Do NOT
- **Do not re-architect or rewrite the existing C++** in `Source/EmvHighway/`
  (`EmvBridgeSubsystem`, `NpcManager`, `NpcVehicleActor`, `EmergencyVehiclePawn`,
  `EmvHUD`, `EmvGameMode`, `EmvBridgeTypes.h`). Assemble Blueprints/levels around
  it. If it fails to compile under 5.8, make the **minimal** API fix and keep the
  design. The socket protocol and coordinate transform are fixed — don't touch
  `EmvBridgeTypes.h::EmvCoords` or the JSON in the subsystem.
- **Do not change the coordinate convention** (below). If the road isn't laid out
  this way, the traffic will not line up with the road and nothing works.

## Hard constraints — the coordinate contract (non-negotiable)
The Python brain streams positions in metres in a "road frame"; the C++ converts
them assuming you built the world like this:
- Highway runs **straight along UE +X**. Forward = +X.
- **1 metre = 100 UE units (cm).** So the road is ~**300,000 cm** long (3 km).
- **The middle lane is centred on Y = 0.** Lanes are 3.5 m = **350 cm** wide.
  Lane centres in UE world: **right lane Y = +350, middle Y = 0, left Y = -350**.
  Road spans Y ∈ [-525, +525] (right edge at +525, left edge at -525).
- Note the road surface Z you build at, and set it on the NpcManager (`RoadZCm`).
- Player start: **X = 3000, Y = -175, Z = roadZ + ~50, Yaw = 0** (this is the
  surrogate EV's start: emv x=30 m on the rescue-gap line, facing +X).

## Tasks (in order) with acceptance criteria

1. **Compile.** Open the .uproject in UE5.8, build the `EmvHighway` module
   (Development Editor). Fix only compile blockers. ✅ Editor opens, module loaded.

2. **Highway level** `HighwayLevel`: a straight 3-lane road along +X, ~3 km,
   middle lane on Y=0, per the constraints above. Add a road material with
   painted **lane markings** (dashed centre lines, solid edges), **guardrails**,
   flanking **ground/landscape**, and a realistic sky: `DirectionalLight` (sun) +
   `SkyAtmosphere` + `SkyLight` + `VolumetricCloud` + `ExponentialHeightFog`, and
   a `PostProcessVolume` for exposure/tonemapping. Lumen + Nanite on. ✅ Looks
   like a real daytime motorway; you can stand on lane centre Y=0 and drive along
   +X for 3 km without falling off.

3. **Player vehicle** `BP_EmergencyVehicle` (parent = `AEmergencyVehiclePawn`):
   - Assign a wheeled-vehicle skeletal mesh + physics asset + Chaos wheel setups
     (reuse/reskin Epic's free **Vehicle template** vehicle; give it an ambulance
     / police livery and a **roof light bar** with emissive materials).
   - Position `CockpitCamera` at the driver's eye point (first person).
   - Assign `SirenSound` (import a looping siren `.wav`) and implement the
     `OnSirenToggled(bool)` event to flash the light-bar emissives (material
     scalar + looping timeline).
   ✅ Drivable with real physics; siren toggles sound + flashing lights.

4. **Enhanced Input** assets, assigned on `BP_EmergencyVehicle`:
   `IMC_Emv` + `IA_Throttle` (Axis1D: **W = +1 throttle, S = -1 brake/reverse**),
   `IA_Steer` (Axis1D: D +1 / A -1), `IA_Handbrake` (Digital, `Space`),
   `IA_Siren` (Digital, `L`), `IA_Camera` (Digital, `C`, toggles cockpit/chase).
   ✅ WASD drives; L toggles siren; C toggles camera.

5. **NPC traffic actors.** Place an `ANpcManager` in the level. Fill its
   `NpcMeshes` array with **3–6 low/mid-poly car static meshes** (variety). Set
   `RoadZCm` = your road surface Z. Ensure each NPC mesh's **pivot is at the
   wheel-contact plane** (so cars sit on the road when placed at Z=RoadZCm) and
   collision **blocks** the player. Leave `bTintByState = false` for a realistic
   look (set true only for debugging awareness states). ✅ At runtime, cars stream
   in ahead of the player, sit on the road, and you can crash into stopped ones.

6. **Game mode** `BP_EmvGameMode` (parent = `AEmvGameMode`, which already sets the
   HUD): set **DefaultPawnClass = BP_EmergencyVehicle**. Make it the level's
   GameMode Override. Add a `PlayerStart` at X=3000, Y=-175, Z=roadZ+50, Yaw=0.
   ✅ Play spawns you in the ambulance in first person on the rescue-gap line.

7. (Automatic) `UEmvBridgeSubsystem` auto-connects — no placement needed.

## End-to-end test (definition of done)
1. In the repo folder run `python experiments/play_ue.py` (leave it waiting).
2. Play-In-Editor the UE game; allow the one-time Windows Firewall prompt.
3. The HUD top-left shows **"brain: connected"** and an NPC/yielding count.
4. Drive forward with W. Expected: cars populate the lanes ahead; as you close
   in with the siren on, cars in your path **pull aside and slow to open a
   corridor**, some fail to yield, and the counts update — matching the reference
   `out/anim_surrogate_sumo.gif`. Motion is smooth (no teleport jitter).
5. If cars float above/below the road → fix `RoadZCm` / mesh pivots. If cars
   appear in the wrong lane or perpendicular → the road isn't on +X / middle lane
   isn't on Y=0 (re-check the contract).

## Notes
- Performance: only ~60 cars stream at once (±600 m window); keep NPC meshes
  reasonable. FPS is on the HUD.
- The brain runs at a fixed 16.7 Hz; the C++ interpolates to your frame rate —
  do not try to "speed up" the brain.
- Audio works in UE (unlike CARLA) — a good siren + engine sound adds a lot.
