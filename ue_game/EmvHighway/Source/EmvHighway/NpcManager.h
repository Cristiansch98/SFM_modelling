#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EmvBridgeTypes.h"
#include "NpcManager.generated.h"

class ANpcVehicleActor;
class UEmvBridgeSubsystem;

/**
 * Owns the pool of ANpcVehicleActor and drives them every frame from the brain's
 * buffered frames.
 *
 * Playback runs on a local clock in *brain sim time*, held a fixed delay behind
 * the newest received frame, and interpolates between the two frames bracketing
 * it. Sim time advances by exactly dt per frame, so playback speed is constant
 * even though arrivals are not: measured arrival gaps are 60 +/- 17 ms, which
 * under the old arrival-time interpolation showed up directly as a +/-29 %
 * swing in apparent NPC speed (the visible jitter). The clock is slewed gently
 * toward its target instead of snapped, so correcting drift is invisible.
 */
UCLASS()
class EMVHIGHWAY_API ANpcManager : public AActor
{
	GENERATED_BODY()

public:
	ANpcManager();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;

	/** NPC car meshes to cycle through for visual variety. Assign in the editor
	 *  (e.g. a few sedan/hatch static meshes). Falls back to a unit cube. */
	UPROPERTY(EditAnywhere, Category = "EMV")
	TArray<TObjectPtr<UStaticMesh>> NpcMeshes;

	/** Ground height (UE cm) of the road surface the cars sit on. */
	UPROPERTY(EditAnywhere, Category = "EMV")
	float RoadZCm = 0.0f;

	/** Tint cars by awareness state (debug). Off = realistic look. */
	UPROPERTY(EditAnywhere, Category = "EMV")
	bool bTintByState = false;

	/** Let NPCs physically block the player instead of overlapping them.
	 *  Off by default: the cars are kinematic, so a block resolves as a launch.
	 *  Only sensible in physics mode, where a corridor actually opens. */
	UPROPERTY(EditAnywhere, Category = "EMV")
	bool bNpcsBlockPlayer = false;

	/** How far behind the newest brain frame to render, in seconds. Must exceed
	 *  the worst late arrival or playback starves and holds; 0.08 s covers four
	 *  frames at the default 50 Hz brain rate plus the odd scheduling spike.
	 *  Costs latency on NPCs only - the player's own car is simulated locally. */
	UPROPERTY(EditAnywhere, Category = "EMV|Smoothing", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float InterpDelaySeconds = 0.08f;

	/** Max playback speed correction while catching up, as a fraction of real
	 *  time. 0.10 = the clock may run 10 % fast or slow; large enough to absorb
	 *  drift within a second or so, small enough to be invisible. */
	UPROPERTY(EditAnywhere, Category = "EMV|Smoothing", meta = (ClampMin = "0.0", ClampMax = "0.5"))
	float MaxSlewFraction = 0.10f;

	/** Proportional gain converting clock error (s) into a speed correction. */
	UPROPERTY(EditAnywhere, Category = "EMV|Smoothing", meta = (ClampMin = "0.0"))
	float SlewGain = 2.0f;

	/** Error beyond which the clock hard-resyncs instead of slewing (a stall,
	 *  a hitch, or a restarted brain). */
	UPROPERTY(EditAnywhere, Category = "EMV|Smoothing", meta = (ClampMin = "0.05"))
	float ResyncThresholdSeconds = 0.5f;

	/** Radius (cm) around the player inside which a *new* NPC is not spawned.
	 *  Posing a kinematic car inside the dynamic player pawn makes Chaos
	 *  resolve the overlap by launching it; the car simply appears once the
	 *  spot is clear. Existing cars are unaffected, so you can still be hit. */
	UPROPERTY(EditAnywhere, Category = "EMV|Smoothing", meta = (ClampMin = "0.0"))
	float SpawnKeepoutCm = 700.0f;

private:
	ANpcVehicleActor* AcquireActor(int32 Id, int32 MeshPick);
	void ReleaseUnseen(const TSet<int32>& SeenIds);
	FTransform PoseFor(const FEmvNpcState& N) const;
	/** Advance RenderSimTime toward (newest - InterpDelay). Returns false if no
	 *  frames have arrived yet. */
	bool AdvancePlaybackClock(float DeltaSeconds);

	UPROPERTY()
	TMap<int32, TObjectPtr<ANpcVehicleActor>> Active;   // id -> actor

	UPROPERTY()
	TArray<TObjectPtr<ANpcVehicleActor>> FreePool;

	UEmvBridgeSubsystem* Bridge = nullptr;
	int32 SpawnCounter = 0;

	/** Local playback position in brain sim time. */
	double RenderSimTime = 0.0;
	bool bClockValid = false;
};
