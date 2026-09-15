#include "EmvGameMode.h"
#include "EmvHUD.h"

AEmvGameMode::AEmvGameMode()
{
	HUDClass = AEmvHUD::StaticClass();
	// DefaultPawnClass is left to the derived Blueprint game mode so the mesh /
	// wheels / input assets can be authored in the editor.
}
