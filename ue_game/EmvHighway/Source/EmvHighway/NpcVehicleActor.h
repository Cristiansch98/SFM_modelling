#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NpcVehicleActor.generated.h"

class UStaticMeshComponent;

/**
 * A single surrounding car. Purely kinematic: the NpcManager sets its transform
 * every frame from the interpolated brain output; the car itself never runs
 * physics (its motion is authored entirely by the brain).
 *
 * Because it is kinematic, *blocking* the player is violent - see
 * SetBlockPlayer. It overlaps the player by default, and blocks the rest of
 * the world.
 */
UCLASS()
class EMVHIGHWAY_API ANpcVehicleActor : public AActor
{
	GENERATED_BODY()

public:
	ANpcVehicleActor();

	/** NPC id this actor is currently bound to (-1 = free in the pool). */
	UPROPERTY(VisibleAnywhere, Category = "EMV")
	int32 BoundId = -1;

	/** Optional debug tint by awareness state (0..3). No-op unless a dynamic
	 *  material with a 'StateColor' vector parameter is assigned to the mesh. */
	UPROPERTY(EditAnywhere, Category = "EMV")
	bool bTintByState = false;

	void ApplyPose(const FTransform& InTransform);
	void ApplyState(int32 State);

	/** Block the player physically (true) or merely overlap them (false).
	 *  Blocking is only sane where traffic actually clears a corridor; in the
	 *  SUMO modes it guarantees launches, because the gaps are narrower than
	 *  the car and the NPC bodies are kinematic. */
	void SetBlockPlayer(bool bBlock);

	UStaticMeshComponent* GetMesh() const { return Mesh; }

protected:
	UPROPERTY(VisibleAnywhere, Category = "EMV")
	TObjectPtr<UStaticMeshComponent> Mesh;

	UPROPERTY()
	TObjectPtr<class UMaterialInstanceDynamic> MID;

	int32 LastState = -1;
};
