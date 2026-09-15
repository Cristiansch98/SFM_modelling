#include "EmergencyVehiclePawn.h"

#include "EmvBridgeSubsystem.h"
#include "EmvBridgeTypes.h"
#include "ChaosWheeledVehicleMovementComponent.h"
#include "Camera/CameraComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Components/AudioComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputMappingContext.h"
#include "InputAction.h"
#include "GameFramework/PlayerController.h"
#include "Engine/World.h"

AEmergencyVehiclePawn::AEmergencyVehiclePawn()
{
	PrimaryActorTick.bCanEverTick = true;

	// First-person cockpit camera (position tuned in the derived Blueprint to
	// sit at the driver's eye point of the actual mesh).
	CockpitCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("CockpitCamera"));
	CockpitCamera->SetupAttachment(GetMesh());
	CockpitCamera->SetRelativeLocation(FVector(30.0f, -35.0f, 150.0f));
	CockpitCamera->bUsePawnControlRotation = false;

	// Chase camera for a third-person toggle.
	ChaseArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("ChaseArm"));
	ChaseArm->SetupAttachment(GetMesh());
	ChaseArm->TargetArmLength = 750.0f;
	ChaseArm->SetRelativeLocation(FVector(0.0f, 0.0f, 200.0f));
	ChaseArm->SetRelativeRotation(FRotator(-12.0f, 0.0f, 0.0f));
	ChaseArm->bDoCollisionTest = false;
	ChaseArm->bEnableCameraLag = true;
	ChaseArm->CameraLagSpeed = 8.0f;

	ChaseCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("ChaseCamera"));
	ChaseCamera->SetupAttachment(ChaseArm);
	ChaseCamera->SetActive(false);

	SirenAudio = CreateDefaultSubobject<UAudioComponent>(TEXT("SirenAudio"));
	SirenAudio->SetupAttachment(GetMesh());
	SirenAudio->bAutoActivate = false;
}

void AEmergencyVehiclePawn::BeginPlay()
{
	Super::BeginPlay();

	if (const UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			Bridge = GI->GetSubsystem<UEmvBridgeSubsystem>();
		}
	}

	if (SirenSound && SirenAudio)
	{
		SirenAudio->SetSound(SirenSound);
	}

	if (APlayerController* PC = Cast<APlayerController>(GetController()))
	{
		if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
			ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PC->GetLocalPlayer()))
		{
			if (InputMapping)
			{
				Subsystem->AddMappingContext(InputMapping, 0);
			}
		}
	}
}

void AEmergencyVehiclePawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	if (UEnhancedInputComponent* EIC = Cast<UEnhancedInputComponent>(PlayerInputComponent))
	{
		if (ThrottleAction)
		{
			EIC->BindAction(ThrottleAction, ETriggerEvent::Triggered, this, &AEmergencyVehiclePawn::Input_Throttle);
			EIC->BindAction(ThrottleAction, ETriggerEvent::Completed, this, &AEmergencyVehiclePawn::Input_Throttle);
		}
		if (SteerAction)
		{
			EIC->BindAction(SteerAction, ETriggerEvent::Triggered, this, &AEmergencyVehiclePawn::Input_Steer);
			EIC->BindAction(SteerAction, ETriggerEvent::Completed, this, &AEmergencyVehiclePawn::Input_Steer);
		}
		if (HandbrakeAction)
		{
			EIC->BindAction(HandbrakeAction, ETriggerEvent::Started, this, &AEmergencyVehiclePawn::Input_HandbrakeStart);
			EIC->BindAction(HandbrakeAction, ETriggerEvent::Completed, this, &AEmergencyVehiclePawn::Input_HandbrakeStop);
		}
		if (SirenAction)
		{
			EIC->BindAction(SirenAction, ETriggerEvent::Started, this, &AEmergencyVehiclePawn::Input_ToggleSiren);
		}
		if (CameraAction)
		{
			EIC->BindAction(CameraAction, ETriggerEvent::Started, this, &AEmergencyVehiclePawn::Input_ToggleCamera);
		}
	}
}

// ------------------------------------------------------------------ input
void AEmergencyVehiclePawn::Input_Throttle(const FInputActionValue& Value)
{
	// One axis: W (+1) = throttle, S (-1) = brake / reverse. Single source for
	// the brake channel so nothing fights it each tick.
	const float Axis = Value.Get<float>();
	if (UChaosWheeledVehicleMovementComponent* Mv =
		Cast<UChaosWheeledVehicleMovementComponent>(GetVehicleMovementComponent()))
	{
		Mv->SetThrottleInput(FMath::Max(Axis, 0.0f));
		Mv->SetBrakeInput(FMath::Max(-Axis, 0.0f));
	}
}

void AEmergencyVehiclePawn::Input_Steer(const FInputActionValue& Value)
{
	if (UChaosWheeledVehicleMovementComponent* Mv =
		Cast<UChaosWheeledVehicleMovementComponent>(GetVehicleMovementComponent()))
	{
		Mv->SetSteeringInput(Value.Get<float>());
	}
}

void AEmergencyVehiclePawn::Input_HandbrakeStart(const FInputActionValue&)
{
	if (UChaosWheeledVehicleMovementComponent* Mv =
		Cast<UChaosWheeledVehicleMovementComponent>(GetVehicleMovementComponent()))
	{
		Mv->SetHandbrakeInput(true);
	}
}

void AEmergencyVehiclePawn::Input_HandbrakeStop(const FInputActionValue&)
{
	if (UChaosWheeledVehicleMovementComponent* Mv =
		Cast<UChaosWheeledVehicleMovementComponent>(GetVehicleMovementComponent()))
	{
		Mv->SetHandbrakeInput(false);
	}
}

void AEmergencyVehiclePawn::Input_ToggleSiren(const FInputActionValue&)
{
	bSirenOn = !bSirenOn;
	if (SirenAudio)
	{
		if (bSirenOn)
		{
			SirenAudio->Play();
		}
		else
		{
			SirenAudio->Stop();
		}
	}
	OnSirenToggled(bSirenOn);
}

void AEmergencyVehiclePawn::Input_ToggleCamera(const FInputActionValue&)
{
	bCockpitView = !bCockpitView;
	CockpitCamera->SetActive(bCockpitView);
	ChaseCamera->SetActive(!bCockpitView);
}

// ------------------------------------------------------------------ tick
void AEmergencyVehiclePawn::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	ApplySpeedGovernor();
	UploadEvState();
}

// ---------------------------------------------------------------- governor
void AEmergencyVehiclePawn::ApplySpeedGovernor()
{
	bGovernorActive = false;
	LastSpeedCapMs = -1.0f;
	if (!bObeySpeedCap || !Bridge)
	{
		return;
	}
	double CapMs = -1.0;
	if (!Bridge->GetSpeedCap(CapMs) || CapMs < 0.0)
	{
		return;    // force-model mode governs itself; nothing to do
	}
	LastSpeedCapMs = static_cast<float>(CapMs);

	UChaosWheeledVehicleMovementComponent* Mv =
		Cast<UChaosWheeledVehicleMovementComponent>(GetVehicleMovementComponent());
	if (!Mv)
	{
		return;
	}

	// Forward speed in m/s (GetVelocity is cm/s in world space).
	const float SpeedMs = static_cast<float>(
		FVector::DotProduct(GetVelocity(), GetActorForwardVector())) * 0.01f;
	const float Over = SpeedMs - static_cast<float>(CapMs) - GovernorToleranceMs;
	if (Over <= 0.0f)
	{
		return;    // player is within the safe speed: leave their input alone
	}

	// Ramp rather than latch: full brake only once well past the cap, so
	// hugging the limit does not oscillate between coast and emergency stop.
	const float Strength = FMath::Clamp(Over / FMath::Max(GovernorBandMs, 0.1f),
		0.0f, 1.0f);
	Mv->SetThrottleInput(0.0f);
	Mv->SetBrakeInput(FMath::Max(Mv->GetBrakeInput(), Strength));
	bGovernorActive = true;
}

void AEmergencyVehiclePawn::UploadEvState()
{
	if (!Bridge)
	{
		return;
	}
	const FVector Loc = GetActorLocation();          // cm
	const FVector Vel = GetVelocity();               // cm/s
	const float YawDeg = GetActorRotation().Yaw;

	FEmvEvState Ev;
	EmvCoords::UeToEmvXY(Loc.X, Loc.Y, Ev.X, Ev.Y);
	EmvCoords::UeToEmvVel(Vel.X, Vel.Y, Ev.Vx, Ev.Vy);
	Ev.Yaw = EmvCoords::UeYawToEmvRad(YawDeg);
	Bridge->SetEvState(Ev);
}
