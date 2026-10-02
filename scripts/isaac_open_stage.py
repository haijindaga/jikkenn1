#!/usr/bin/env python3
"""Open an existing USD stage for visual inspection without playing physics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument(
    "--view",
    choices=("side", "capture"),
    default="side",
    help=(
        "side creates a temporary upright overview of the robot and receiver; "
        "capture reuses the authored RGB-D camera"
    ),
)
parser.add_argument(
    "--camera-prim",
    default="/World/camera_0",
    help="Existing authored camera used when --view capture is selected",
)
args = parser.parse_args()
stage_path = args.stage.expanduser().resolve()
if not stage_path.is_file():
    parser.error(f"stage does not exist: {stage_path}")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_active_viewport_camera
    from pxr import UsdGeom

    from panda_handover.geometry import look_at_quaternion_world

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
    if args.view == "capture":
        camera_prim = stage.GetPrimAtPath(args.camera_prim)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(
                "authored camera does not exist or is not a Camera: "
                f"{args.camera_prim}"
            )
        active_camera_path = args.camera_prim
    else:
        # Side-on, Z-up presentation view.  Its X target is the midpoint of the
        # table and reviewed receiver roots, so both fit laterally in frame.
        # Camera.set_world_pose(..., camera_axes="world") is the same tested
        # axis conversion used by the RGB-D capture pipeline.
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
    set_active_viewport_camera(active_camera_path)
    for _ in range(2):
        simulation_app.update()
    print(f"opened stage: {stage.GetRootLayer().identifier}", flush=True)
    print(f"viewport camera: {active_camera_path} ({args.view})", flush=True)
    print("Timeline is stopped. Close the Isaac Sim window when done.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
