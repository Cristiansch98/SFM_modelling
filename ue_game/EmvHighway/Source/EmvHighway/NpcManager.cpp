#include "NpcManager.h"

#include "NpcVehicleActor.h"
#include "EmvBridgeSubsystem.h"
#include "EmvBridgeTypes.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "HAL/PlatformTime.h"
#include "HAL/IConsoleManager.h"

// Diagnostic: log the interpolation alpha each tick ("emv.LogInterp 1").
static TAutoConsoleVariable<int32> CVarEmvLogAlpha(
	TEXT("emv.LogInterp"), 0,
	TEXT("Log NpcManager interpolation alpha each tick (0=off, 1=on)."));

ANpcManager::ANpcManager()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ANpcManager::BeginPlay()
{
	Super::BeginPlay();
	if (const UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			Bridge = GI->GetSubsystem<UEmvBridgeSubsystem>();
		}
	}
	if (!Bridge)
	{
		UE_LOG(LogTemp, Warning, TEXT("[emv] NpcManager: bridge subsystem not found"));
	}
}

FTransform ANpcManager::PoseFor(const FEmvNpcState& N) const
{
	const FVector2D XY = EmvCoords::EmvToUeXY(N.X, N.Y);
	const FVector Location(XY.X, XY.Y, RoadZCm);
	const FRotator Rotation(0.0f, static_cast<float>(EmvCoords::EmvYawToUeDeg(N.Yaw)), 0.0f);
	return FTransform(Rotation, Location);
}

bool ANpcManager::AdvancePlaybackClock(float DeltaSeconds)
{
	double Newest = 0.0;
	if (!Bridge->GetNewestSimTime(Newest))
	{
		return false;   // no data yet
	}

	const double Target = Newest - static_cast<double>(InterpDelaySeconds);
	if (!bClockValid || FMath::Abs(Target - RenderSimTime) > ResyncThresholdSeconds)
	{
		RenderSimTime = Target;      // cold start, hitch, or restarted brain
		bClockValid = true;
		return true;
	}

	// Slew rather than snap: a small, bounded speed correction is invisible,
	// whereas jumping the clock is exactly the stutter we are removing.
	const double Error = Target - RenderSimTime;
	const double Rate = FMath::Clamp(1.0 + SlewGain * Error,
		1.0 - MaxSlewFraction, 1.0 + MaxSlewFraction);
	RenderSimTime += static_cast<double>(DeltaSeconds) * Rate;
	return true;
}

void ANpcManager::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (!Bridge)
	{
		return;
	}

	if (!AdvancePlaybackClock(DeltaSeconds))
	{
		return;   // no data yet
	}

	FEmvNpcFrame Prev, Next;
	float Alpha = 1.0f;
	if (!Bridge->GetFramesForSimTime(RenderSimTime, Prev, Next, Alpha))
	{
		return;
	}

	// A full-population reset: drop everything so ids re-bind cleanly.
	if (Next.bReset)
	{
		for (const TPair<int32, TObjectPtr<ANpcVehicleActor>>& Pair : Active)
		{
			if (Pair.Value)
			{
				Pair.Value->BoundId = -1;
				Pair.Value->SetActorHiddenInGame(true);
				Pair.Value->SetActorEnableCollision(false);
				FreePool.Add(Pair.Value);
			}
		}
		Active.Reset();
	}

#if !UE_BUILD_SHIPPING
	if (CVarEmvLogAlpha.GetValueOnGameThread() != 0)
	{
		double Newest = 0.0;
		Bridge->GetNewestSimTime(Newest);
		UE_LOG(LogTemp, Log,
			TEXT("[emv-interp] render=%.3f newest=%.3f lag=%.3f alpha=%.3f npcs=%d"),
			RenderSimTime, Newest, Newest - RenderSimTime, Alpha, Next.Npcs.Num());
	}
#endif

	// index Prev by id for quick lookup
	TMap<int32, const FEmvNpcState*> PrevById;
	PrevById.Reserve(Prev.Npcs.Num());
	for (const FEmvNpcState& P : Prev.Npcs)
	{
		PrevById.Add(P.Id, &P);
	}

	TSet<int32> Seen;
	Seen.Reserve(Next.Npcs.Num());

	// Player location, for the spawn keep-out below.
	FVector PlayerLoc = FVector::ZeroVector;
	bool bHavePlayer = false;
	if (const UWorld* World = GetWorld())
	{
		if (const APlayerController* PC = World->GetFirstPlayerController())
		{
			if (const APawn* Pawn = PC->GetPawn())
			{
				PlayerLoc = Pawn->GetActorLocation();
				bHavePlayer = true;
			}
		}
	}

	for (const FEmvNpcState& N : Next.Npcs)
	{
		Seen.Add(N.Id);
		const FTransform NextPose = PoseFor(N);
		FTransform Pose = NextPose;
		if (const FEmvNpcState* const* Found = PrevById.Find(N.Id))
		{
			const FTransform PrevPose = PoseFor(**Found);
			Pose.Blend(PrevPose, NextPose, Alpha);   // Prev(0) -> Next(1) by Alpha
		}
		// Never *introduce* a car on top of the player: a kinematic body posed
		// inside the dynamic pawn is resolved by Chaos as a launch. Cars that
		// are already bound keep updating, so real contact still behaves.
		if (bHavePlayer && !Active.Contains(N.Id))
		{
			const double DistSq = FVector::DistSquared2D(Pose.GetLocation(), PlayerLoc);
			if (DistSq < static_cast<double>(SpawnKeepoutCm) * SpawnKeepoutCm)
			{
				Seen.Remove(N.Id);   // stay unbound; retry next frame
				continue;
			}
		}
		ANpcVehicleActor* Actor = AcquireActor(N.Id, N.Id % FMath::Max(1, NpcMeshes.Num()));
		if (Actor)
		{
			Actor->ApplyPose(Pose);
			Actor->ApplyState(N.State);
		}
	}

	ReleaseUnseen(Seen);
}

ANpcVehicleActor* ANpcManager::AcquireActor(int32 Id, int32 MeshPick)
{
	if (const TObjectPtr<ANpcVehicleActor>* Existing = Active.Find(Id))
	{
		return *Existing;
	}

	ANpcVehicleActor* Actor = nullptr;
	if (FreePool.Num() > 0)
	{
		Actor = FreePool.Pop();
		Actor->SetActorHiddenInGame(false);
		Actor->SetActorEnableCollision(true);
	}
	else if (UWorld* World = GetWorld())
	{
		FActorSpawnParameters Params;
		Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		Actor = World->SpawnActor<ANpcVehicleActor>(ANpcVehicleActor::StaticClass(),
			FVector::ZeroVector, FRotator::ZeroRotator, Params);
	}
	if (Actor)
	{
		// Re-pick the mesh on every bind, not just on spawn: a recycled actor
		// would otherwise keep the body of the car it used to be.
		if (NpcMeshes.IsValidIndex(MeshPick) && NpcMeshes[MeshPick] && Actor->GetMesh())
		{
			Actor->GetMesh()->SetStaticMesh(NpcMeshes[MeshPick]);
		}
		Actor->BoundId = Id;
		Actor->bTintByState = bTintByState;
		Actor->SetBlockPlayer(bNpcsBlockPlayer);
		Active.Add(Id, Actor);
		++SpawnCounter;
	}
	return Actor;
}

void ANpcManager::ReleaseUnseen(const TSet<int32>& SeenIds)
{
	TArray<int32> ToRelease;
	for (const TPair<int32, TObjectPtr<ANpcVehicleActor>>& Pair : Active)
	{
		if (!SeenIds.Contains(Pair.Key))
		{
			ToRelease.Add(Pair.Key);
		}
	}
	for (int32 Id : ToRelease)
	{
		if (TObjectPtr<ANpcVehicleActor> Actor = Active.FindRef(Id))
		{
			Actor->BoundId = -1;
			Actor->SetActorHiddenInGame(true);
			Actor->SetActorEnableCollision(false);
			FreePool.Add(Actor);
		}
		Active.Remove(Id);
	}
}
