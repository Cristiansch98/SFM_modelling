#include "EmvHUD.h"

#include "EmvBridgeSubsystem.h"
#include "EmvBridgeTypes.h"
#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"

void AEmvHUD::DrawHUD()
{
	Super::DrawHUD();
	if (!Canvas)
	{
		return;
	}

	// speed from the possessed pawn (cm/s -> km/h)
	float SpeedKmh = 0.0f;
	if (const APawn* P = GetOwningPawn())
	{
		SpeedKmh = P->GetVelocity().Size() * 0.036f;
	}

	// bridge state + yielding count
	bool bConnected = false;
	int32 Yielding = 0;
	int32 NpcCount = 0;
	if (const UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			if (UEmvBridgeSubsystem* Bridge = GI->GetSubsystem<UEmvBridgeSubsystem>())
			{
				bConnected = Bridge->IsConnected();
				FEmvNpcFrame Prev, Next;
				if (Bridge->GetInterpolationFrames(Prev, Next))
				{
					NpcCount = Next.Npcs.Num();
					for (const FEmvNpcState& N : Next.Npcs)
					{
						if (N.State >= 2)   // YIELDING or HOLD
						{
							++Yielding;
						}
					}
				}
			}
		}
	}

	const float X = 40.0f;
	float Y = 40.0f;
	const float LineH = 26.0f;

	DrawText(FString::Printf(TEXT("%.0f km/h"), SpeedKmh), FLinearColor::White, X, Y, nullptr, 1.6f);
	Y += LineH * 1.6f;
	DrawText(FString::Printf(TEXT("brain: %s"), bConnected ? TEXT("connected") : TEXT("waiting...")),
		bConnected ? FLinearColor::Green : FLinearColor::Yellow, X, Y);
	Y += LineH;
	DrawText(FString::Printf(TEXT("NPCs: %d   yielding: %d"), NpcCount, Yielding),
		FLinearColor(0.85f, 0.85f, 0.9f), X, Y);
	Y += LineH;
	DrawText(FString::Printf(TEXT("FPS: %.0f"), 1.0f / FMath::Max(GetWorld()->GetDeltaSeconds(), 1e-4f)),
		FLinearColor(0.7f, 0.7f, 0.7f), X, Y);
}
