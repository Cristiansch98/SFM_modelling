using UnrealBuildTool;
using System.Collections.Generic;

public class EmvHighwayEditorTarget : TargetRules
{
	public EmvHighwayEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("EmvHighway");
	}
}
