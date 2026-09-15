"""Attach the wheel meshes of BP_EmergencyVehicle to their skeleton bones.

AttachSocketName isn't reachable through the editor toolset's property setter,
so it is set here on the Blueprint's component templates:

  UnrealEditor-Cmd.exe EmvHighway.uproject -run=pythonscript
      -script="<this file>" -unattended -nosplash
"""
import unreal

BP_PATH = "/Game/Blueprints/BP_EmergencyVehicle"
SOCKETS = {"WheelFL": "Phys_Wheel_FL", "WheelFR": "Phys_Wheel_FR",
           "WheelBL": "Phys_Wheel_BL", "WheelBR": "Phys_Wheel_BR"}

bp = unreal.load_asset(BP_PATH)
gen = bp.generated_class()

for var, bone in SOCKETS.items():
    comp = unreal.load_object(gen, "%s_GEN_VARIABLE" % var)
    if comp is None:
        unreal.log_warning("[emv-veh] no template for %s" % var)
        continue
    comp.set_editor_property("attach_socket_name", bone)
    unreal.log("[emv-veh] %s -> socket %s (parent %s)"
               % (var, comp.get_editor_property("attach_socket_name"),
                  comp.get_attach_parent()))

unreal.BlueprintEditorLibrary.compile_blueprint(bp)
unreal.EditorAssetLibrary.save_asset(BP_PATH)
unreal.log("[emv-veh] done")
