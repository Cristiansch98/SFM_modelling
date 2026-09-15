using UnrealBuildTool;

public class EmvHighway : ModuleRules
{
	public EmvHighway(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",     // keyboard driving
			"ChaosVehicles",     // AWheeledVehiclePawn + Chaos wheeled movement
			"PhysicsCore",
			"Sockets",           // FSocket / ISocketSubsystem
			"Networking",        // FInternetAddr helpers
			"Json"               // parse the NPC frames
		});
	}
}
