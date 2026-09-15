using UnrealBuildTool;
using System.Collections.Generic;

public class EmvHighwayTarget : TargetRules
{
	public EmvHighwayTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("EmvHighway");
	}
}
