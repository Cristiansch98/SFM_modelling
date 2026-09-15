#pragma once

#include "CoreMinimal.h"
#include "WheeledVehiclePawn.h"
#include "EmergencyVehiclePawn.generated.h"

class UCameraComponent;
class USpringArmComponent;
class UAudioComponent;
class USoundBase;
class UInputMappingContext;
class UInputAction;
class UEmvBridgeSubsystem;
struct FInputActionValue;

/**
 * Player emergency vehicle. Extends the Chaos wheeled-vehicle pawn (assign the
 * skeletal mesh, physics asset and wheel setups in a derived Blueprint), adds a
 * first-person cockpit camera + chase camera, a siren, keyboard driving, and
 * uploads its kinematics to the Python brain every tick so the surrounding
 * traffic reacts to the human driver.
 */
UCLASS()
class EMVHIGHWAY_API AEmergencyVehiclePawn : public AWheeledVehiclePawn
{
	GENERATED_BODY()

public:
	AEmergencyVehiclePawn();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;
	virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;

	/** BP hook so the vehicle Blueprint can flash its light bars / emissives. */
	UFUNCTION(BlueprintImplementableEvent, Category = "EMV")
	void OnSirenToggled(bool bOn);

protected:
	// ---- camera ----
	UPROPERTY(VisibleAnywhere, Category = "EMV|Camera")
	TObjectPtr<UCameraComponent> CockpitCamera;

	UPROPERTY(VisibleAnywhere, Category = "EMV|Camera")
	TObjectPtr<USpringArmComponent> ChaseArm;

	UPROPERTY(VisibleAnywhere, Category = "EMV|Camera")
	TObjectPtr<UCameraComponent> ChaseCamera;

	// ---- siren ----
	UPROPERTY(VisibleAnywhere, Category = "EMV|Siren")
	TObjectPtr<UAudioComponent> SirenAudio;

	UPROPERTY(EditAnywhere, Category = "EMV|Siren")
	TObjectPtr<USoundBase> SirenSound;

	// ---- Enhanced Input (assign the assets in the derived Blueprint) ----
	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputMappingContext> InputMapping;

	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputAction> ThrottleAction;   // Axis1D, W/S or Up/Down

	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputAction> SteerAction;       // Axis1D, A/D or Left/Right

	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputAction> HandbrakeAction;    // Digital, Space

	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputAction> SirenAction;        // Digital (toggle)

	UPROPERTY(EditAnywhere, Category = "EMV|Input")
	TObjectPtr<UInputAction> CameraAction;       // Digital (toggle)

	bool bSirenOn = false;
	bool bCockpitView = true;

	/** Obey a speed ceiling sent by the brain (the SUMO modes send their own
	 *  car-following safe speed). Off = raw player input, which lets you drive
	 *  into a leader that SUMO has no way to brake for you. */
	UPROPERTY(EditAnywhere, Category = "EMV|Governor")
	bool bObeySpeedCap = true;

	/** Speed overshoot (m/s) above the cap at which braking reaches full. */
	UPROPERTY(EditAnywhere, Category = "EMV|Governor", meta = (ClampMin = "0.1"))
	float GovernorBandMs = 2.0f;

	/** Deadband (m/s): SUMO's safe speed sits a hair under the current speed
	 *  almost continuously in dense traffic, so without this the governor would
	 *  feather the brake constantly instead of only when it matters. */
	UPROPERTY(EditAnywhere, Category = "EMV|Governor", meta = (ClampMin = "0.0"))
	float GovernorToleranceMs = 1.0f;

	/** True while the governor is actually overriding the player (for the HUD). */
	UPROPERTY(BlueprintReadOnly, Category = "EMV|Governor")
	bool bGovernorActive = false;

	/** Last cap received, m/s; < 0 when the brain sends none. */
	UPROPERTY(BlueprintReadOnly, Category = "EMV|Governor")
	float LastSpeedCapMs = -1.0f;

private:
	/** Clamp throttle / add brake so the EV cannot exceed the brain's cap. */
	void ApplySpeedGovernor();

	void Input_Throttle(const FInputActionValue& Value);
	void Input_Steer(const FInputActionValue& Value);
	void Input_HandbrakeStart(const FInputActionValue& Value);
	void Input_HandbrakeStop(const FInputActionValue& Value);
	void Input_ToggleSiren(const FInputActionValue& Value);
	void Input_ToggleCamera(const FInputActionValue& Value);

	void UploadEvState();

	UEmvBridgeSubsystem* Bridge = nullptr;
};
