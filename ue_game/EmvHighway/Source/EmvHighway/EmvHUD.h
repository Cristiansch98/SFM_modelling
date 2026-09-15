#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "EmvHUD.generated.h"

/**
 * Minimal canvas HUD (no UMG assets needed): speed, siren state, brain
 * connection status, and the number of surrounding cars currently yielding.
 */
UCLASS()
class EMVHIGHWAY_API AEmvHUD : public AHUD
{
	GENERATED_BODY()

public:
	virtual void DrawHUD() override;
};
