#include "NpcVehicleActor.h"

#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"

ANpcVehicleActor::ANpcVehicleActor()
{
	PrimaryActorTick.bCanEverTick = false;   // the manager drives us

	Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
	RootComponent = Mesh;
	Mesh->SetMobility(EComponentMobility::Movable);
	Mesh->SetSimulatePhysics(false);
	Mesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	Mesh->SetCollisionResponseToAllChannels(ECR_Block);
	SetBlockPlayer(false);
}

void ANpcVehicleActor::SetBlockPlayer(bool bBlock)
{
	if (!Mesh)
	{
		return;
	}
	// These cars are KINEMATIC bodies teleported to a pose every frame. When one
	// is moved into the player's dynamic pawn, the solver has no mass to push
	// against and depenetrates with an effectively unbounded impulse - the car
	// launches. Overlap instead of block means contact is reported but no
	// impulse is generated, so the worst case is clipping rather than a crash.
	// This matters most in the SUMO modes, where traffic never opens a corridor
	// and squeezing past with less clearance than your own width is the only
	// way through.
	const ECollisionResponse R = bBlock ? ECR_Block : ECR_Overlap;
	Mesh->SetCollisionResponseToChannel(ECC_Pawn, R);
	Mesh->SetCollisionResponseToChannel(ECC_Vehicle, R);
	Mesh->SetCollisionResponseToChannel(ECC_PhysicsBody, R);
}

void ANpcVehicleActor::ApplyPose(const FTransform& InTransform)
{
	// Sweep = false: teleport-style move (we are authoritative over position).
	SetActorTransform(InTransform, /*bSweep=*/false, nullptr, ETeleportType::TeleportPhysics);
}

void ANpcVehicleActor::ApplyState(int32 State)
{
	if (!bTintByState || State == LastState)
	{
		return;
	}
	LastState = State;

	if (!MID && Mesh && Mesh->GetMaterial(0))
	{
		MID = Mesh->CreateAndSetMaterialInstanceDynamic(0);
	}
	if (!MID)
	{
		return;
	}
	// grey unaware / amber noticed / orange yielding / teal hold (matches the
	// emv.viz + SUMO-bridge legend).
	static const FLinearColor Colors[4] = {
		FLinearColor(0.49f, 0.49f, 0.47f),
		FLinearColor(0.79f, 0.52f, 0.0f),
		FLinearColor(0.85f, 0.35f, 0.15f),
		FLinearColor(0.10f, 0.62f, 0.44f)
	};
	MID->SetVectorParameterValue(TEXT("StateColor"), Colors[FMath::Clamp(State, 0, 3)]);
}
