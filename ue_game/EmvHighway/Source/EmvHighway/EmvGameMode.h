#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "EmvGameMode.generated.h"

/**
 * Sets the HUD class in C++. Assign the DefaultPawnClass to your derived
 * emergency-vehicle Blueprint (BP_EmergencyVehicle) in the editor, since that
 * Blueprint carries the mesh / physics asset / wheel + input assignments.
 */
UCLASS()
class EMVHIGHWAY_API AEmvGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AEmvGameMode();
};
