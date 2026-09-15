#include "EmvAutopilotComponent.h"

#include "EmvBridgeSubsystem.h"
#include "EmvBridgeTypes.h"
#include "ChaosWheeledVehicleMovementComponent.h"
#include "WheeledVehiclePawn.h"
#include "Engine/World.h"
#include "Engine/GameInstance.h"

UEmvAutopilotComponent::UEmvAutopilotComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UEmvAutopilotComponent::BeginPlay()
{
	Super::BeginPlay();

	if (AWheeledVehiclePawn* Pawn = Cast<AWheeledVehiclePawn>(GetOwner()))
	{
		Movement = Cast<UChaosWheeledVehicleMovementComponent>(
			Pawn->GetVehicleMovementComponent());
	}
	if (const UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			Bridge = GI->GetSubsystem<UEmvBridgeSubsystem>();
		}
	}
	if (!Movement || !Bridge)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[emv] autopilot: movement=%d bridge=%d - staying idle"),
			Movement != nullptr, Bridge != nullptr);
	}
}

void UEmvAutopilotComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!bEnabled || !Movement || !Bridge)
	{
		return;
	}

	// ---- our own state, in the emv frame -----------------------------------
	const AActor* Owner = GetOwner();
	const FVector Loc = Owner->GetActorLocation();
	const FVector Vel = Owner->GetVelocity();
	double Ex = 0.0, Ey = 0.0, Evx = 0.0, Evy = 0.0;
	EmvCoords::UeToEmvXY(Loc.X, Loc.Y, Ex, Ey);
	EmvCoords::UeToEmvVel(Vel.X, Vel.Y, Evx, Evy);

	// ---- leaders: the newest brain frame, already in emv metres / m/s -------
	FEmvNpcFrame Prev, Next;
	const bool bHaveNpcs = Bridge->GetInterpolationFrames(Prev, Next);

	const double HalfL = 0.5 * (EvLength + NpcLength);
	const double HalfW = 0.5 * (EvWidth + NpcWidth);

	// IDM interaction bound over every candidate ahead (forces.py::car_car)
	double AIdm = 0.0;              // 0 = unconstrained (the model uses +BIG)
	bool bConstrained = false;
	// nearest straggler still straddling the corridor line (_ev_lat_target)
	double YTarget = CorridorY;
	double NearestBlockerDx = TNumericLimits<double>::Max();

	if (bHaveNpcs)
	{
		for (const FEmvNpcState& N : Next.Npcs)
		{
			const double Dx = N.X - Ex;
			const double Dy = N.Y - Ey;

			if (Dx > 0.1)
			{
				const double LatGap = FMath::Max(FMath::Abs(Dy) - HalfW, 0.0);
				const double Overlap = FMath::Clamp(
					(OverlapMargin - LatGap) / FMath::Max(OverlapMargin, KINDA_SMALL_NUMBER),
					0.0, 1.0);
				if (Overlap > 0.0)
				{
					const double Gap = FMath::Max(Dx - HalfL, 0.3);
					const double Dv = Evx - N.Vx;             // closing speed
					const double SStar = MinGap + FMath::Max(
						Evx * TimeHeadway
						+ Evx * Dv / (2.0 * FMath::Sqrt(AccelMax * BrakeComfort)), 0.0);
					const double APair = -AccelMax * FMath::Square(SStar / Gap) * Overlap;
					AIdm = bConstrained ? FMath::Min(AIdm, APair) : APair;
					bConstrained = true;
				}
			}

			// weave target: nearest blocker straddling the corridor line ahead
			if (Dx > 0.0 && Dx < WeaveLookahead && Dx < NearestBlockerDx
				&& FMath::Abs(N.Y - CorridorY) < HalfW + WeaveMargin)
			{
				NearestBlockerDx = Dx;
				const double Esc = (CorridorY >= N.Y) ? 1.0 : -1.0;
				YTarget = FMath::Clamp(N.Y + Esc * (HalfW + WeaveMargin),
					CorridorY - WeaveMax, CorridorY + WeaveMax);
			}
		}
	}

	// ---- longitudinal: relaxation to v0, bounded by the IDM demand ---------
	const double AxDrive = (DesiredSpeed - Evx) / FMath::Max(Tau, KINDA_SMALL_NUMBER);
	const double Ax = AxDrive + FMath::Min(0.0, AIdm);
	LastAccelCmd = static_cast<float>(Ax);

	Movement->SetThrottleInput(static_cast<float>(
		FMath::Clamp(Ax / AccelMax, 0.0, 1.0)));
	Movement->SetBrakeInput(static_cast<float>(
		FMath::Clamp(-Ax / BrakeComfort, 0.0, 1.0)));
	Movement->SetHandbrakeInput(false);

	// ---- lateral: critically damped spring -> steering angle ---------------
	const double Ay = OmegaLat * OmegaLat * (YTarget - Ey)
		- 2.0 * ZetaLat * OmegaLat * Evy;

	// bicycle model: kappa = ay / v^2, delta = atan(kappa * wheelbase).
	// +y in the emv frame is the driver's LEFT, steering is positive to the
	// right, hence the sign flip.
	const double Speed = FMath::Max(FMath::Abs(Evx), 3.0);
	const double Curvature = Ay / (Speed * Speed);
	const double SteerRad = FMath::Atan(Curvature * Wheelbase);
	const double SteerNorm = FMath::Clamp(
		-FMath::RadiansToDegrees(SteerRad) / FMath::Max(MaxSteerAngleDeg, 1.0f),
		-1.0, 1.0);
	LastSteerCmd = static_cast<float>(SteerNorm);
	Movement->SetSteeringInput(static_cast<float>(SteerNorm));
}
