#!/usr/bin/env python3
"""Open an existing USD stage for visual inspection without playing physics."""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument(
    "--camera-prim",
    default="/World/camera_0",
    help="Existing authored camera to use for the viewport",
)
args = parser.parse_args()
stage_path = args.stage.expanduser().resolve()
if not stage_path.is_file():
    parser.error(f"stage does not exist: {stage_path}")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})
try:
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.viewports import set_active_viewport_camera
    from pxr import UsdGeom

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
    camera_prim = stage.GetPrimAtPath(args.camera_prim)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f"authored camera does not exist or is not a Camera: {args.camera_prim}"
        )
    # Reuse the exact authored RGB-D camera pose that the capture pipeline
    # already validated.  Do not reconstruct a look-at rotation: doing so can
    # introduce a 180-degree roll through a camera-axis convention mismatch.
    set_active_viewport_camera(args.camera_prim)
    for _ in range(2):
        simulation_app.update()
    print(f"opened stage: {stage.GetRootLayer().identifier}", flush=True)
    print(f"viewport camera: {args.camera_prim}", flush=True)
    print("Timeline is stopped. Close the Isaac Sim window when done.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
