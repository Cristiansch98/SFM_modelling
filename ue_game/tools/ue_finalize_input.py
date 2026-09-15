"""Add the Negate modifiers to IMC_Emv.

Enhanced Input modifiers are instanced UObjects, which the editor toolset's
JSON property setter cannot construct, so this runs as a python commandlet:

  UnrealEditor-Cmd.exe EmvHighway.uproject -run=pythonscript
      -script="<this file>" -unattended -nosplash

W/Up = +1 throttle, S/Down = -1 (brake+reverse); D/Right = +1 steer,
A/Left = -1.
"""
import unreal

IMC = "/Game/Input/IMC_Emv.IMC_Emv"
NEGATED = {("IA_Throttle", "S"), ("IA_Throttle", "Down"),
           ("IA_Steer", "A"), ("IA_Steer", "Left")}

imc = unreal.load_asset(IMC)
mappings = list(imc.get_editor_property("mappings"))
out = []
for m in mappings:
    action = m.get_editor_property("action")
    key = m.get_editor_property("key")
    name = (action.get_name() if action else "",
            str(key.get_editor_property("key_name")))
    if name in NEGATED:
        neg = unreal.InputModifierNegate()
        neg.set_editor_property("x", True)
        neg.set_editor_property("y", True)
        neg.set_editor_property("z", True)
        m.set_editor_property("modifiers", [neg])
        unreal.log("[emv-input] negate %s / %s" % name)
    else:
        m.set_editor_property("modifiers", [])
    out.append(m)

imc.set_editor_property("mappings", out)
unreal.EditorAssetLibrary.save_asset(IMC)

# report what the asset ended up with
for m in unreal.load_asset(IMC).get_editor_property("mappings"):
    a = m.get_editor_property("action")
    unreal.log("[emv-input] %s <- %s  modifiers=%d"
               % (a.get_name() if a else "?",
                  str(m.get_editor_property("key").get_editor_property("key_name")),
                  len(m.get_editor_property("modifiers"))))
