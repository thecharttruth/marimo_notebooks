# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "bpy==5.2.0",
#     "diffusers==0.39.0",
#     "einops",
#     "huggingface-hub==1.24.0",
#     "hy3dgen==2.0.2",
#     "mathutils==5.1.0",
#     "numpy==2.5.1",
#     "omegaconf==2.3.1",
#     "opencv-python-headless==5.0.0.93",
#     "pillow==12.3.0",
#     "pygltflib==1.16.5",
#     "pymeshlab",
#     "requests==2.34.2",
#     "torch==2.11.0",
#     "transformers",
#     "trimesh==4.12.2",
#     "xatlas==0.0.11",
# ]
# ///
# NOTE: do NOT list hy3dgen here. Its rembg dependency pulls llvmlite
# versions that cannot build on Python 3.13. Install it with --no-deps.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def intro(mo):
    mo.md(r"""
    # Hunyuan3D → Blender Turntable (fixed rear projection)

    End-to-end MoLab pipeline:

    1. **Image → 3D** with Hunyuan3D-2 (turbo shape model)
    2. **Texture** with Hunyuan3D-Paint (requires `custom_rasterizer`)
    3. **Normalize + studio light** in Blender (`bpy`)
    4. **Cycles CUDA** PNG frames → **FFmpeg** MP4 → `mo.video`

    Verified here: Python 3.13, RTX PRO 6000 (~96 GB), `bpy==5.2.0`, FFmpeg present.

    **Install notes**
    - `hy3dgen` is installed with `--no-deps` (`rembg` → `llvmlite` cannot build on Py3.13)
    - Texture needs a one-time `custom_rasterizer` CUDA build (handled in `install_texgen`)
    """)
    return


@app.cell
def install_hy3d():
    """Install hy3dgen without rembg (Python 3.13 incompatible)."""

    def _ensure_hy3d():
        import importlib.util as _ilu
        import subprocess as _sp
        import sys as _sys

        if _ilu.find_spec("hy3dgen") is None:
            _sp.check_call(
                [
                    "uv",
                    "pip",
                    "install",
                    "hy3dgen==2.0.2",
                    "--no-deps",
                    "-p",
                    _sys.executable,
                ]
            )
        for _pkg, _mod in (
            ("fast-simplification", "fast_simplification"),
            ("trimesh", "trimesh"),
            ("xatlas", "xatlas"),
        ):
            if _ilu.find_spec(_mod) is None:
                _sp.check_call(["uv", "pip", "install", _pkg, "-p", _sys.executable])
        return True

    hy3d_ready = _ensure_hy3d()
    return (hy3d_ready,)


@app.cell
def install_texgen(hy3d_ready):
    """Ensure Hunyuan paint + custom_rasterizer are importable."""

    def _ensure_texgen(_hy3d_ready):
        import importlib as _il
        import importlib.util as _ilu
        import os as _os
        import sys as _sys
        from pathlib import Path as _Path

        assert _hy3d_ready

        # Diffusers needs trust_remote_code for hunyuanpaint custom pipeline code.
        _candidates = [
            _Path(_sys.prefix)
            / "lib"
            / f"python{_sys.version_info.major}.{_sys.version_info.minor}"
            / "site-packages"
            / "hy3dgen"
            / "texgen"
            / "utils"
            / "multiview_utils.py",
            _Path("/tmp/uv-venv/lib/python3.13/site-packages/hy3dgen/texgen/utils/multiview_utils.py"),
        ]
        for _cand in _candidates:
            if _cand.exists():
                _txt = _cand.read_text()
                if (
                    "trust_remote_code=True" not in _txt
                    and "custom_pipeline=custom_pipeline_path" in _txt
                ):
                    _cand.write_text(
                        _txt.replace(
                            "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)",
                            "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16, trust_remote_code=True)",
                        )
                    )
                break

        for _rr in (
            _Path("/tmp/uv-venv/lib/python3.13/site-packages/custom_rasterizer/render.py"),
            _Path(_sys.prefix) / "lib/python3.13/site-packages/custom_rasterizer/render.py",
        ):
            if _rr.exists():
                _rt = _rr.read_text()
                if "clamp_depth=torch.zeros(0)" in _rt:
                    _rt = _rt.replace("clamp_depth=torch.zeros(0)", "clamp_depth=None")
                    if "if clamp_depth is None:" not in _rt:
                        _rt = _rt.replace(
                            "assert (pos.device == tri.device)\n",
                            "assert (pos.device == tri.device)\n    if clamp_depth is None:\n        import torch as _torch\n        clamp_depth = _torch.zeros(0, device=pos.device, dtype=_torch.float32)\n",
                        )
                    _rr.write_text(_rt)

        # custom_rasterizer links against PyTorch (libc10.so) + CUDA runtime.
        _lib_dirs = []
        try:
            import torch as _torch

            _torch_lib = _Path(_torch.__file__).resolve().parent / "lib"
            if _torch_lib.is_dir():
                _lib_dirs.append(str(_torch_lib))
        except Exception as _torch_err:
            print("torch import for LD_LIBRARY_PATH failed:", _torch_err)

        for _cuda_lib in (
            "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib",
            str(_Path(_sys.prefix) / "lib/python3.13/site-packages/nvidia/cu13/lib"),
            "/tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13/lib",
        ):
            if _Path(_cuda_lib).is_dir():
                _lib_dirs.append(_cuda_lib)
                break

        if _lib_dirs:
            _os.environ["LD_LIBRARY_PATH"] = ":".join(
                _lib_dirs + [_os.environ.get("LD_LIBRARY_PATH", "")]
            )

        _ready = _ilu.find_spec("custom_rasterizer") is not None
        if _ready:
            try:
                _il.import_module("custom_rasterizer")
            except Exception as _e:
                print("custom_rasterizer import warning:", _e)
                _ready = False

        if not _ready:
            print(
                "WARNING: custom_rasterizer not importable (missing shared libs "
                "like libc10.so, or extension not built). "
                "Shape generation still works; texture step will be skipped."
            )
        else:
            print("texgen ready: custom_rasterizer OK")
        return _ready

    texgen_ready = _ensure_texgen(hy3d_ready)
    return (texgen_ready,)


@app.cell
def imports(hy3d_ready, texgen_ready):
    import math
    import shutil
    import subprocess
    from pathlib import Path

    assert hy3d_ready

    import bpy
    import cv2
    import marimo as mo
    import numpy as np
    import requests
    import torch
    from mathutils import Vector
    from PIL import Image
    from hy3dgen.shapegen import (
        DegenerateFaceRemover,
        FaceReducer,
        FloaterRemover,
        Hunyuan3DDiTFlowMatchingPipeline,
    )

    Hunyuan3DPaintPipeline = None
    if texgen_ready:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline as _Paint

            Hunyuan3DPaintPipeline = _Paint
        except Exception as _tex_err:
            print("paint import failed:", _tex_err)

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    print("bpy", bpy.app.version_string, "ffmpeg", shutil.which("ffmpeg"))
    print("paint pipeline:", "yes" if Hunyuan3DPaintPipeline else "no")
    return (
        DegenerateFaceRemover,
        FaceReducer,
        FloaterRemover,
        Hunyuan3DDiTFlowMatchingPipeline,
        Hunyuan3DPaintPipeline,
        Image,
        Path,
        Vector,
        bpy,
        cv2,
        math,
        mo,
        np,
        requests,
        shutil,
        subprocess,
    )


@app.cell
def config(Path):
    OUTPUT_DIR = Path.cwd() / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_IMAGE = OUTPUT_DIR / "hunyuan_input.png"
    GENERATED_GLB = OUTPUT_DIR / "generated_asset.glb"
    FRAMES_DIR = OUTPUT_DIR / "turntable_frames"
    OUTPUT_MP4 = OUTPUT_DIR / "hunyuan_turntable.mp4"

    MODEL_ID = "tencent/Hunyuan3D-2"
    MODEL_SUBFOLDER = "hunyuan3d-dit-v2-0-turbo"
    NUM_INFERENCE_STEPS = 20
    GUIDANCE_SCALE = 5.0
    TARGET_FACE_COUNT = 40000

    FPS = 30
    FRAME_COUNT = 60
    RESOLUTION = 768
    TARGET_HEIGHT = 2.0
    CYCLES_SAMPLES = 32

    DEMO_IMAGE_URL = (
        "https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/assets/demo.png"
    )

    (OUTPUT_DIR, GENERATED_GLB, OUTPUT_MP4)
    return (
        CYCLES_SAMPLES,
        DEMO_IMAGE_URL,
        FPS,
        FRAMES_DIR,
        FRAME_COUNT,
        GENERATED_GLB,
        GUIDANCE_SCALE,
        INPUT_IMAGE,
        MODEL_ID,
        MODEL_SUBFOLDER,
        NUM_INFERENCE_STEPS,
        OUTPUT_DIR,
        OUTPUT_MP4,
        RESOLUTION,
        TARGET_FACE_COUNT,
        TARGET_HEIGHT,
    )


@app.cell
def input_ui(mo):
    image_upload = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        label="Reference image (optional — demo used if empty)",
    )
    regenerate = mo.ui.run_button(label="Generate 3D + turntable")
    mo.hstack([image_upload, regenerate], justify="start", gap=1)
    return image_upload, regenerate


@app.cell
def prepare_image(
    DEMO_IMAGE_URL,
    INPUT_IMAGE,
    Image,
    OUTPUT_DIR,
    cv2,
    image_upload,
    mo,
    np,
    requests,
):
    def _ensure_demo_image(path):
        if path.exists():
            return path
        resp = requests.get(DEMO_IMAGE_URL, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return path


    def _load_rgba_from_upload_or_demo():
        if image_upload.value:
            import io
            raw = image_upload.value[0].contents
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            img.save(INPUT_IMAGE)
            return img, INPUT_IMAGE
        path = _ensure_demo_image(INPUT_IMAGE)
        return Image.open(path).convert("RGBA"), path


    def _ensure_alpha(image):
        arr = np.array(image)
        if arr.shape[2] == 4 and float(arr[:, :, 3].mean()) < 250:
            return image
        rgb = arr[:, :, :3].copy()
        h, w = rgb.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(
                rgb,
                mask,
                seed,
                (0, 0, 0),
                loDiff=(20, 20, 20),
                upDiff=(20, 20, 20),
                flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
            )
        alpha = np.where(mask[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
        out = Image.fromarray(np.dstack([arr[:, :, :3], alpha]), mode="RGBA")
        out.save(OUTPUT_DIR / "hunyuan_input_nobg.png")
        return out


    _source_image, _source_path = _load_rgba_from_upload_or_demo()
    input_image = _ensure_alpha(_source_image)
    input_image.save(INPUT_IMAGE)
    mo.hstack([
        mo.image(src=str(INPUT_IMAGE), width=280),
        mo.md(f"""
    **Input:** `{_source_path.name}`  
    Size: {input_image.size[0]}×{input_image.size[1]}  
    Click **Generate 3D + turntable** to (re)run Hunyuan + Blender.
    """),
    ], gap=1)
    return (input_image,)


@app.cell
def generate(
    DegenerateFaceRemover,
    FaceReducer,
    FloaterRemover,
    GENERATED_GLB,
    GUIDANCE_SCALE,
    Hunyuan3DDiTFlowMatchingPipeline,
    Hunyuan3DPaintPipeline,
    Image,
    MODEL_ID,
    MODEL_SUBFOLDER,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    TARGET_FACE_COUNT,
    TARGET_HEIGHT,
    Vector,
    bpy,
    input_image,
    math,
    mo,
    np,
    regenerate,
    texgen_ready,
):

    _should_run = bool(regenerate.value) or (not GENERATED_GLB.exists())


    def _glb_has_textures(_path):
        """True if GLB embeds at least one image (albedo/etc)."""
        try:
            import pygltflib as _gltf

            _g = _gltf.GLTF2().load(str(_path))
            return bool(getattr(_g, "images", None))
        except Exception:
            try:
                return _path.stat().st_size > 1_500_000
            except Exception:
                return False


    def _project_photo_albedo(_glb_path, _image):
        """Front-camera photo projection fallback when Hunyuan Paint is unavailable.

        Zooms UVs into the photo center so a full-body subject covers the bulkier
        Hunyuan silhouette better than a naive full-frame map.
        """
        from bpy_extras.object_utils import world_to_camera_view as _w2c

        _albedo_path = OUTPUT_DIR / "projected_albedo.png"
        _arr = np.array(_image.convert("RGBA"))
        _rgb = _arr[:, :, :3].astype(np.float32)
        _a = _arr[:, :, 3:4].astype(np.float32) / 255.0
        if float(_a.mean()) < 0.05:
            _out = _rgb
        else:
            # Mid-grey composite — avoid the white wipe that killed earlier albedos.
            _out = _rgb * _a + 128.0 * (1.0 - _a)
        _out = np.clip(_out, 0, 255).astype(np.uint8)
        Image.fromarray(_out, "RGB").save(_albedo_path)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        _scene = bpy.context.scene
        bpy.ops.import_scene.gltf(filepath=str(_glb_path))
        for _o in list(bpy.data.objects):
            if _o.type != "MESH":
                bpy.data.objects.remove(_o, do_unlink=True)
        _meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        if not _meshes:
            raise RuntimeError("projection fallback: no mesh in GLB")
        _obj = _meshes[0]
        _mesh = _obj.data

        _root = bpy.data.objects.new("ProjRoot", None)
        _scene.collection.objects.link(_root)
        _wm = _obj.matrix_world.copy()
        _obj.parent = _root
        _obj.matrix_world = _wm

        def _bounds(_objs):
            _pts = [o.matrix_world @ Vector(c) for o in _objs for c in o.bound_box]
            _mn = Vector((min(p.x for p in _pts), min(p.y for p in _pts), min(p.z for p in _pts)))
            _mx = Vector((max(p.x for p in _pts), max(p.y for p in _pts), max(p.z for p in _pts)))
            return _mn, _mx

        bpy.context.view_layer.update()
        _mn, _mx = _bounds([_obj])
        _root.scale = (TARGET_HEIGHT / max(_mx.z - _mn.z, 1e-6),) * 3
        bpy.context.view_layer.update()
        _mn, _mx = _bounds([_obj])
        _root.location -= Vector(((_mn.x + _mx.x) / 2, (_mn.y + _mx.y) / 2, _mn.z))
        bpy.context.view_layer.update()
        _mn, _mx = _bounds([_obj])
        _height = _mx.z - _mn.z
        _width = max(_mx.x - _mn.x, _mx.y - _mn.y)

        _cam = bpy.data.objects.new("ProjCam", bpy.data.cameras.new("ProjCam"))
        _scene.collection.objects.link(_cam)
        _scene.camera = _cam
        _cam.location = (0.0, -max(3.2, _height * 2.4, _width * 2.8), _height * 0.52)
        _cam.rotation_euler = (math.radians(78), 0.0, 0.0)
        bpy.context.view_layer.update()

        if not _mesh.uv_layers:
            _mesh.uv_layers.new(name="UVMap")
        _uv = _mesh.uv_layers.active
        _uv.name = "UVMap"
        _cam_dir = (_cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
        _ndc, _front, _is_front = [], [], []
        for _loop in _mesh.loops:
            _world = _obj.matrix_world @ _mesh.vertices[_loop.vertex_index].co
            _p = _w2c(_scene, _cam, _world)
            _u, _v = float(_p.x), float(_p.y)
            _ndc.append((_u, _v))
            _n = (_obj.matrix_world.to_3x3() @ _mesh.vertices[_loop.vertex_index].normal).normalized()
            _facing = _n.dot(-_cam_dir) > 0.05
            _is_front.append(_facing)
            if _facing:
                _front.append((_u, _v))
        _ndc = np.asarray(_ndc)
        _is_front = np.asarray(_is_front, dtype=bool)
        _front = np.asarray(_front) if _front else _ndc
        _u0, _v0 = _front.min(0)
        _u1, _v1 = _front.max(0)
        _pu = max((_u1 - _u0) * 0.01, 1e-4)
        _pv = max((_v1 - _v0) * 0.01, 1e-4)
        _u0, _u1, _v0, _v1 = _u0 - _pu, _u1 + _pu, _v0 - _pv, _v1 + _pv
        # Zoom into photo center so subject fills the bulkier mesh silhouette.
        _U_MARGIN, _V_MARGIN = 0.29, 0.04
        # Only front-facing loops receive the photo projection. Rear-facing loops
        # are parked on a neutral patch of the texture so the face is not duplicated
        # across the back of the head/body.
        _BACK_UV = (0.02, 0.02)
        for _li, (_u, _v) in enumerate(_ndc):
            if not _is_front[_li]:
                _uv.data[_li].uv = _BACK_UV
                continue
            _un = float(np.clip((_u - _u0) / (_u1 - _u0), 0, 1))
            _vn = float(np.clip((_v - _v0) / (_v1 - _v0), 0, 1))
            _uv.data[_li].uv = (
                _U_MARGIN + (1.0 - 2.0 * _U_MARGIN) * _un,
                _V_MARGIN + (1.0 - 2.0 * _V_MARGIN) * _vn,
            )

        _mat = bpy.data.materials.new("PhotoAlbedo")
        _mat.use_nodes = True
        _nt = _mat.node_tree
        _bsdf = _nt.nodes.get("Principled BSDF")
        _tex = _nt.nodes.new("ShaderNodeTexImage")
        _tex.image = bpy.data.images.load(str(_albedo_path))
        _tex.image.pack()
        _tex.extension = "EXTEND"
        _nt.links.new(_tex.outputs["Color"], _bsdf.inputs["Base Color"])
        _bsdf.inputs["Roughness"].default_value = 0.7
        if "Specular IOR Level" in _bsdf.inputs:
            _bsdf.inputs["Specular IOR Level"].default_value = 0.05
        if not _obj.material_slots:
            _mesh.materials.append(_mat)
        _obj.material_slots[0].link = "DATA"
        _mesh.materials[0] = _mat

        _mw = _obj.matrix_world.copy()
        _obj.parent = None
        _obj.matrix_world = _mw
        bpy.data.objects.remove(_root, do_unlink=True)
        bpy.data.objects.remove(_cam, do_unlink=True)

        bpy.ops.object.select_all(action="DESELECT")
        _obj.select_set(True)
        bpy.context.view_layer.objects.active = _obj
        bpy.ops.export_scene.gltf(
            filepath=str(_glb_path),
            export_format="GLB",
            use_selection=True,
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            export_image_format="AUTO",
        )
        print(
            "projection fallback wrote",
            _glb_path,
            _glb_path.stat().st_size,
            "bytes; albedo",
            _albedo_path,
        )


    if _should_run:
        mo.status.toast("Generating mesh", description="Hunyuan3D-2 turbo on CUDA...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID,
            subfolder=MODEL_SUBFOLDER,
        )
        shape_pipeline.to("cuda")
        mesh = shape_pipeline(
            image=input_image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )[0]
        try:
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=TARGET_FACE_COUNT)
        except Exception as _post_err:
            print("postprocess warning:", _post_err)
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACE_COUNT)
            except Exception as _simp_err:
                print("simplify warning:", _simp_err)

        _paint_ok = False
        # Texture with Hunyuan Paint when the CUDA rasterizer is available.
        if texgen_ready and Hunyuan3DPaintPipeline is not None:
            mo.status.toast("Painting texture", description="Hunyuan3D-Paint multiview bake...")
            try:
                _arr = np.array(input_image.convert("RGBA")).astype(np.float32)
                _alpha = np.nan_to_num(_arr[:, :, 3], nan=255.0)
                _rgb = np.nan_to_num(_arr[:, :, :3], nan=0.0)
                _a = (_alpha / 255.0)[..., None]
                # Prefer mid-grey over white so empty alpha cannot wipe the photo.
                if float(_a.mean()) < 0.05:
                    _paint_rgb = _rgb
                else:
                    _paint_rgb = _rgb * _a + 128.0 * (1.0 - _a)
                _paint_image = Image.fromarray(
                    np.clip(_paint_rgb, 0, 255).astype(np.uint8), mode="RGB"
                ).convert("RGBA")
                _paint = Hunyuan3DPaintPipeline.from_pretrained(MODEL_ID)
                mesh = _paint(mesh, image=_paint_image)
                print("Textured mesh visual:", type(getattr(mesh, "visual", None)).__name__)
                _paint_ok = True
            except Exception as _tex_err:
                print("texture warning (will try Blender projection fallback):", _tex_err)
                _paint_ok = False

        mesh.export(str(GENERATED_GLB))
        print("Wrote", GENERATED_GLB, GENERATED_GLB.stat().st_size, "bytes")

        if (not _paint_ok) or (not _glb_has_textures(GENERATED_GLB)):
            mo.status.toast(
                "Projection fallback",
                description="Baking photo albedo onto mesh in Blender...",
            )
            try:
                _project_photo_albedo(GENERATED_GLB, input_image)
            except Exception as _proj_err:
                print("projection fallback failed:", _proj_err)
    else:
        print("Using existing", GENERATED_GLB, GENERATED_GLB.stat().st_size, "bytes")
        # Ensure heavy meshes are decimated once for turntable performance.
        try:
            import trimesh as _trimesh

            _mesh = _trimesh.load(str(GENERATED_GLB), force="mesh")
            _faces = len(_mesh.faces) if hasattr(_mesh, "faces") else 0
            if _faces > TARGET_FACE_COUNT:
                print(f"Decimating existing mesh faces {_faces} -> ~{TARGET_FACE_COUNT}")
                try:
                    _mesh = FaceReducer()(_mesh, max_facenum=TARGET_FACE_COUNT)
                except Exception:
                    _mesh = _mesh.simplify_quadric_decimation(face_count=TARGET_FACE_COUNT)
                _mesh.export(str(GENERATED_GLB))
                print("Rewrote", GENERATED_GLB, GENERATED_GLB.stat().st_size, "bytes")
        except Exception as _dec_err:
            print("decimate-existing warning:", _dec_err)

        if not _glb_has_textures(GENERATED_GLB):
            mo.status.toast(
                "Projection fallback",
                description="Existing GLB has no textures — baking photo albedo...",
            )
            try:
                _project_photo_albedo(GENERATED_GLB, input_image)
            except Exception as _proj_err:
                print("projection fallback failed:", _proj_err)

    assert GENERATED_GLB.exists(), "GLB missing — generation failed"
    INPUT_GLB = GENERATED_GLB
    f"{INPUT_GLB.name} ({INPUT_GLB.stat().st_size / 1e6:.1f} MB)"

    return (INPUT_GLB,)


@app.cell
def render_turntable(
    CYCLES_SAMPLES,
    FPS,
    FRAMES_DIR,
    FRAME_COUNT,
    INPUT_GLB,
    OUTPUT_MP4,
    RESOLUTION,
    TARGET_HEIGHT,
    Vector,
    bpy,
    math,
    shutil,
    subprocess,
):
    def world_bounds(objects):
        points = []
        for obj in objects:
            for corner in obj.bound_box:
                points.append(obj.matrix_world @ Vector(corner))
        minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
        maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
        return minimum, maximum


    def add_area_light(scene, name, location, energy, size, focus):
        data = bpy.data.lights.new(name=name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = location
        scene.collection.objects.link(light)
        constraint = light.constraints.new(type="TRACK_TO")
        constraint.target = focus
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"
        return light


    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = False
    _prefs = bpy.context.preferences.addons["cycles"].preferences
    _prefs.compute_device_type = "CUDA"
    _prefs.get_devices()
    for _d in _prefs.devices:
        _d.use = _d.type == "CUDA"

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.025, 0.025, 0.025)

    _before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(INPUT_GLB))
    _imported = list(set(bpy.data.objects) - _before)
    mesh_objects = [obj for obj in _imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Imported GLB has no mesh objects")

    root = bpy.data.objects.new("TurntableRoot", None)
    scene.collection.objects.link(root)
    _imported_set = set(_imported)
    for obj in _imported:
        if obj.parent not in _imported_set:
            _wm = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = _wm

    bpy.context.view_layer.update()
    minimum, maximum = world_bounds(mesh_objects)
    height = maximum.z - minimum.z
    if height <= 1e-8:
        raise RuntimeError("Invalid model bounds")
    root.scale = (TARGET_HEIGHT / height,) * 3
    bpy.context.view_layer.update()
    minimum, maximum = world_bounds(mesh_objects)
    root.location -= Vector((
        (minimum.x + maximum.x) / 2,
        (minimum.y + maximum.y) / 2,
        minimum.z,
    ))
    bpy.context.view_layer.update()
    minimum, maximum = world_bounds(mesh_objects)
    model_height = maximum.z - minimum.z
    model_width = max(maximum.x - minimum.x, maximum.y - minimum.y)

    bpy.ops.mesh.primitive_plane_add(size=max(10.0, model_width * 6), location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "StudioFloor"
    floor_material = bpy.data.materials.new("StudioFloorMaterial")
    floor_material.use_nodes = True
    floor_bsdf = floor_material.node_tree.nodes.get("Principled BSDF")
    floor_bsdf.inputs["Base Color"].default_value = (0.06, 0.06, 0.06, 1)
    floor_bsdf.inputs["Roughness"].default_value = 0.72
    floor.data.materials.append(floor_material)

    focus = bpy.data.objects.new("CameraFocus", None)
    focus.location = (0, 0, model_height * 0.52)
    scene.collection.objects.link(focus)

    camera_data = bpy.data.cameras.new("StudioCamera")
    camera = bpy.data.objects.new("StudioCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 55
    distance = max(4.0, model_height * 2.5, model_width * 2.5)
    camera.location = (0, -distance, model_height * 0.58)
    cam_c = camera.constraints.new(type="TRACK_TO")
    cam_c.target = focus
    cam_c.track_axis = "TRACK_NEGATIVE_Z"
    cam_c.up_axis = "UP_Y"

    add_area_light(scene, "KeyLight", (model_height * 1.8, -model_height * 2.0, model_height * 2.3), 1100, model_height * 1.4, focus)
    add_area_light(scene, "FillLight", (-model_height * 1.7, -model_height * 1.1, model_height * 1.4), 500, model_height * 1.7, focus)
    add_area_light(scene, "RimLight", (0, model_height * 1.6, model_height * 2.1), 900, model_height, focus)

    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0, 0, 0)
    root.keyframe_insert(data_path="rotation_euler", frame=1)
    root.rotation_euler = (0, 0, math.tau)
    root.keyframe_insert(data_path="rotation_euler", frame=FRAME_COUNT + 1)
    if root.animation_data and root.animation_data.action and hasattr(root.animation_data.action, "fcurves"):
        for fcurve in root.animation_data.action.fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"

    scene.render.filepath = str(FRAMES_DIR / "frame_")
    print(f"Rendering {FRAME_COUNT} Cycles CUDA frames from Hunyuan GLB...")
    bpy.ops.render.render(animation=True)
    print("frames", len(list(FRAMES_DIR.glob("*.png"))))

    _ffmpeg = [
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(OUTPUT_MP4),
    ]
    _result = subprocess.run(_ffmpeg, text=True, capture_output=True)
    if _result.returncode != 0:
        raise RuntimeError("FFmpeg failed:\n" + _result.stderr[-4000:])
    f"Saved {OUTPUT_MP4} ({OUTPUT_MP4.stat().st_size / 1e6:.2f} MB)"
    return


@app.cell
def show_video(INPUT_GLB, OUTPUT_MP4, mo):
    mo.vstack([
        mo.md("## Turntable preview"),
        mo.video(
            src=str(OUTPUT_MP4),
            autoplay=True,
            muted=True,
            loop=True,
            width=720,
        ),
        mo.md(f"`{OUTPUT_MP4}` · source mesh `{INPUT_GLB.name}`"),
    ])
    return


if __name__ == "__main__":
    app.run()
