"""Unreal-side importer for the generated source assets.

Run headless:
  UnrealEditor-Cmd.exe EmvHighway.uproject -run=pythonscript
      -script="<this file>"

Imports SourceAssets/*.obj as StaticMeshes under /Game/Vehicles/Npc and
siren_loop.wav as a looping SoundWave under /Game/Audio.
"""
import os
import unreal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "SourceAssets")

MESH_DIR = "/Game/Vehicles/Npc"
AUDIO_DIR = "/Game/Audio"

atools = unreal.AssetToolsHelpers.get_asset_tools()


def task(src, dest_path, name, options=None):
    t = unreal.AssetImportTask()
    t.filename = src
    t.destination_path = dest_path
    t.destination_name = name
    t.automated = True
    t.replace_existing = True
    t.save = True
    if options:
        t.options = options
    return t


def mesh_options():
    opts = unreal.FbxImportUI()
    opts.import_mesh = True
    opts.import_as_skeletal = False
    opts.import_materials = False
    opts.import_textures = False
    opts.static_mesh_import_data.combine_meshes = True
    opts.static_mesh_import_data.generate_lightmap_u_vs = True
    opts.static_mesh_import_data.auto_generate_collision = True
    opts.static_mesh_import_data.import_uniform_scale = 1.0
    return opts


def main():
    tasks = []
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".obj"):
            tasks.append(task(os.path.join(SRC, fn), MESH_DIR, fn[:-4],
                              mesh_options()))
    wav = os.path.join(SRC, "siren_loop.wav")
    if os.path.exists(wav):
        tasks.append(task(wav, AUDIO_DIR, "S_SirenLoop"))

    atools.import_asset_tasks(tasks)

    # post-process: collision + looping siren
    for t in tasks:
        for path in t.get_editor_property("imported_object_paths") or []:
            obj = unreal.load_asset(path)
            unreal.log("[emv-import] %s -> %s" % (t.filename, path))
            if isinstance(obj, unreal.StaticMesh):
                unreal.EditorStaticMeshLibrary.remove_collisions(obj)
                unreal.EditorStaticMeshLibrary.add_simple_collisions(
                    obj, unreal.ScriptingCollisionShapeType.BOX)
                obj.set_editor_property("nanite_settings",
                                        unreal.MeshNaniteSettings(enabled=False))
            elif isinstance(obj, unreal.SoundWave):
                obj.set_editor_property("looping", True)
            unreal.EditorAssetLibrary.save_asset(path)

    unreal.log("[emv-import] done, %d tasks" % len(tasks))


main()
