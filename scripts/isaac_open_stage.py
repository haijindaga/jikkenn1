#!/usr/bin/env python3
"""Open an existing USD stage for visual inspection without playing physics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument(
    "--view",
    choices=("scene", "side", "oblique", "capture"),
    default="side",
    help=(
        "scene preserves the ordinary viewport, side creates a temporary upright "
        "overview, oblique creates the usual receiver-right presentation view, "
        "and capture reuses the authored RGB-D camera"
    ),
)
parser.add_argument(
    "--viewer-camera-position",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    help="Optional presentation-only viewport camera position in world metres",
)
parser.add_argument(
    "--viewer-camera-target",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    help="Optional presentation-only viewport look-at target in world metres",
)
parser.add_argument("--panda-prim", default="/World/Panda")
parser.add_argument(
    "--camera-prim",
    default="/World/camera_0",
    help="Existing authored camera used when --view capture is selected",
)
parser.add_argument(
    "--handover-region-json",
    type=Path,
    help=(
        "Presentation-only reviewed handover box to display while the timeline "
        "remains stopped"
    ),
)
parser.add_argument(
    "--handover-region-opacity",
    type=float,
    default=0.18,
    help="Opacity of --handover-region-json; must be in (0, 1]",
)
args = parser.parse_args()
if (args.viewer_camera_position is None) != (args.viewer_camera_target is None):
    parser.error(
        "--viewer-camera-position and --viewer-camera-target must be supplied together"
    )
stage_path = args.stage.expanduser().resolve()
if not stage_path.is_file():
    parser.error(f"stage does not exist: {stage_path}")
region_path = None
if args.handover_region_json is not None:
    region_path = args.handover_region_json.expanduser().resolve()
    if not region_path.is_file():
        parser.error(f"handover region does not exist: {region_path}")
if not math.isfinite(args.handover_region_opacity) or not (
    0.0 < args.handover_region_opacity <= 1.0
):
    parser.error("--handover-region-opacity must be in (0, 1]")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_active_viewport_camera
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from panda_handover.geometry import look_at_quaternion_world
    from panda_handover.handover_overlay import handover_region_overlay_spec

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    result = context.open_stage(str(stage_path))
    print(f"open_stage returned: {result}", flush=True)
    for _ in range(120):
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"USD stage did not open: {stage_path}")

    if region_path is not None:
        region_layout = json.loads(region_path.read_text(encoding="utf-8"))
        region_spec = handover_region_overlay_spec(
            region_layout,
            source=region_path,
            opacity=args.handover_region_opacity,
        )
        panda_root = stage.GetPrimAtPath(args.panda_prim)
        if not panda_root.IsValid():
            raise RuntimeError(
                "cannot place robot-base handover overlay; missing "
                f"{args.panda_prim}"
            )
        overlay_root_path = "/World/presentation_handover_region"
        overlay_box_path = f"{overlay_root_path}/Box"
        original_edit_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
            overlay_root = UsdGeom.Xform.Define(stage, overlay_root_path)
            base_to_world = UsdGeom.Xformable(panda_root).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            UsdGeom.Xformable(overlay_root).MakeMatrixXform().Set(base_to_world)

            overlay_box = UsdGeom.Cube.Define(stage, overlay_box_path)
            overlay_box.CreateSizeAttr(1.0)
            common_api = UsdGeom.XformCommonAPI(overlay_box)
            common_api.SetTranslate(Gf.Vec3d(*region_spec.center_robot_base_m))
            common_api.SetRotate(
                Gf.Vec3f(0.0, 0.0, region_spec.yaw_degrees),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
            common_api.SetScale(Gf.Vec3f(*region_spec.size_m))
            color = Gf.Vec3f(*region_spec.color_rgb)
            overlay_box.GetDisplayColorAttr().Set([color])
            overlay_box.GetDisplayOpacityAttr().Set([region_spec.opacity])
            overlay_box.CreateDoubleSidedAttr(True)

            material = UsdShade.Material.Define(
                stage, f"{overlay_root_path}/Material"
            )
            shader = UsdShade.Shader.Define(
                stage, f"{overlay_root_path}/Material/PreviewSurface"
            )
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
                region_spec.opacity
            )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(overlay_box.GetPrim()).Bind(material)
        finally:
            stage.SetEditTarget(original_edit_target)
        for overlay_prim in Usd.PrimRange(stage.GetPrimAtPath(overlay_root_path)):
            if overlay_prim.HasAPI(UsdPhysics.RigidBodyAPI) or overlay_prim.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                raise RuntimeError("presentation handover overlay acquired physics APIs")
        print(
            "static handover region: "
            f"center={region_spec.center_robot_base_m}, "
            f"size={region_spec.size_m}, opacity={region_spec.opacity}",
            flush=True,
        )

    active_camera_path = None
    custom_viewer_camera = args.viewer_camera_position is not None
    if args.view == "capture" and not custom_viewer_camera:
        camera_prim = stage.GetPrimAtPath(args.camera_prim)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(
                "authored camera does not exist or is not a Camera: "
                f"{args.camera_prim}"
            )
        active_camera_path = args.camera_prim
    elif args.view in {"side", "oblique"} or custom_viewer_camera:
        # Presentation-only, Z-up views.  Positive world Y places the receiver
        # at frame right.  The oblique preset also exposes the tabletop depth,
        # which makes the reviewed handover volume easier to read.
        # Camera.set_world_pose(..., camera_axes="world") is the same tested
        # axis conversion used by the RGB-D capture pipeline.
        if custom_viewer_camera:
            eye = np.asarray(args.viewer_camera_position, dtype=np.float64)
            target = np.asarray(args.viewer_camera_target, dtype=np.float64)
        elif args.view == "oblique":
            eye = np.asarray((1.55, 2.60, 1.45), dtype=np.float64)
            target = np.asarray((-0.20, 0.0, 0.35), dtype=np.float64)
        else:
            eye = np.asarray((-0.325, -3.0, 1.4), dtype=np.float64)
            target = np.asarray((-0.325, 0.0, 0.35), dtype=np.float64)
        orientation = look_at_quaternion_world(eye, target)
        active_camera_path = "/ViewerCamera"
        original_edit_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
            viewer_camera = Camera(
                prim_path=active_camera_path,
                position=eye,
                orientation=orientation,
            )
            viewer_camera.set_world_pose(
                eye, orientation, camera_axes="world"
            )
        finally:
            stage.SetEditTarget(original_edit_target)
    if active_camera_path is not None:
        set_active_viewport_camera(active_camera_path)
    for _ in range(2):
        simulation_app.update()
    print(f"opened stage: {stage.GetRootLayer().identifier}", flush=True)
    if active_camera_path is None:
        print("viewport camera: existing scene viewport (scene)", flush=True)
    else:
        camera_label = "custom" if custom_viewer_camera else args.view
        print(f"viewport camera: {active_camera_path} ({camera_label})", flush=True)
        if active_camera_path == "/ViewerCamera":
            print(
                f"viewer position: {eye.tolist()}, target: {target.tolist()}",
                flush=True,
            )
    print("Timeline is stopped. Close the Isaac Sim window when done.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
