#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "EmvAutopilotComponent.generated.h"

class UChaosWheeledVehicleMovementComponent;
class UEmvBridgeSubsystem;

/**
 * Drives the player EV the way the Python force model drives its EV row, so
 * the scenario can be watched hands-off instead of played.
 *
 * Mirrors emv/forces.py for the EV:
 *   longitudinal  ax = (v0 - vx)/tau  +  min(0, a_idm)
 *                 a_idm = -amax (sstar / gap)^2 x overlap, worst leader
 *                 sstar = s0 + max(vx T + vx dv / (2 sqrt(amax b)), 0)
 *   lateral       ay = w^2 (y_target - y) - 2 zeta w vy   (critically damped)
 *                 y_target = corridor line, weaved around the nearest
 *                 straggler still straddling it
 *
 * Leader states come from the bridge NPC frames (emv metres, m/s), so
 * this reasons over exactly the data the model produced. The resulting
 * accelerations are mapped onto throttle / brake / steering because UE drives
 * a real Chaos vehicle rather than a point mass.
 */
UCLASS(ClassGroup = (EMV), meta = (BlueprintSpawnableComponent))
class EMVHIGHWAY_API UEmvAutopilotComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UEmvAutopilotComponent();

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	/** Master switch - the Blueprint stops applying keyboard input while on. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	bool bEnabled = true;

	// ---- longitudinal (emv Params: EV row of make_sumo_like) ----
	/** m/s, EV desired speed = min(41, 1.40 * 27.78). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float DesiredSpeed = 38.892f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float Tau = 0.40f;              // s, Params::tau_ev

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float TimeHeadway = 0.6f;       // s, Params::ev_T

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float MinGap = 1.5f;            // m, Params::ev_s0

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float AccelMax = 3.2f;          // m/s^2, Params::ev_amax

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float BrakeComfort = 4.0f;      // m/s^2, Params::ev_b

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float OverlapMargin = 0.35f;    // m, Params::ev_overlap_margin

	// ---- lateral ----
	/** m, corridor line in the emv frame: (n_lanes-1) * lane_width. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float CorridorY = 7.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float OmegaLat = 1.3f;          // rad/s, Params::ev_omega_lat

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float ZetaLat = 1.0f;           // -, Params::ev_zeta_lat

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float WeaveLookahead = 30.0f;   // m, Params::ev_weave_lookahead

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float WeaveMargin = 0.4f;       // m, Params::ev_weave_margin

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float WeaveMax = 1.6f;          // m, Params::ev_weave_max

	// ---- vehicle geometry / actuation ----
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float EvLength = 6.2f;          // m

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float EvWidth = 2.2f;           // m

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float NpcLength = 4.5f;         // m, streamed cars carry no extent

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float NpcWidth = 1.8f;          // m

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float Wheelbase = 2.9f;         // m, for the steering bicycle model

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EMV|Autopilot")
	float MaxSteerAngleDeg = 40.0f;

	/** Last commanded values, for the HUD / debugging. */
	UPROPERTY(BlueprintReadOnly, Category = "EMV|Autopilot")
	float LastAccelCmd = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "EMV|Autopilot")
	float LastSteerCmd = 0.0f;

private:
	UChaosWheeledVehicleMovementComponent* Movement = nullptr;
	UEmvBridgeSubsystem* Bridge = nullptr;
};
