#!/usr/bin/env python3
"""Isaac Sim 5.1 smoke test: Panda scene plus one calibrated RGB-D frame.

Run from the repository root inside ``env_isaaclab``.  No grasping, SAM3 or
motion planning is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/capture_smoke"))
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(640, 480))
    parser.add_argument("--horizontal-fov", type=float, default=69.0)
    return parser.parse_args()


args = parse_args()

# Isaac requires SimulationApp construction before importing other Isaac modules.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

try:
    import numpy as np

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.capture import RgbdCapture
    from panda_handover.geometry import (
        depth_mask_to_points,
        look_at_quaternion_world,
        matrix_from_pose,
        transform_points,
    )

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    world.scene.add_default_ground_plane()
    world.scene.add(Franka(prim_path="/World/Panda", name="panda"))

    # The table and primitives are only geometry/camera checks.  Tool assets are
    # introduced after the RGB-D contract is verified.
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.50, 0.0, 0.35]),
            scale=np.array([0.80, 1.00, 0.70]),
            color=np.array([0.45, 0.32, 0.20]),
        )
    )
    test_object = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestObject",
            name="test_object",
            position=np.array([0.50, 0.0, 0.76]),
            scale=np.array([0.20, 0.05, 0.05]),
            color=np.array([0.1, 0.5, 0.9]),
            mass=0.05,
        )
    )
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Obstacle",
            name="obstacle",
            position=np.array([0.48, -0.20, 0.79]),
            scale=np.array([0.10, 0.10, 0.10]),
            color=np.array([0.9, 0.2, 0.1]),
            mass=0.1,
        )
    )

    camera_position = np.array([1.25, 0.0, 1.35])
    camera_target = np.array([0.48, 0.0, 0.73])
    camera_orientation = look_at_quaternion_world(camera_position, camera_target)
    width, height = args.resolution
    camera = Camera(
        prim_path="/World/Camera0",
        position=camera_position,
        orientation=camera_orientation,
        frequency=30,
        resolution=(width, height),
    )

    world.reset()
    camera.initialize()
    camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")
    camera.set_clipping_range(0.05, 3.0)
    aperture = float(camera.get_horizontal_aperture())
    focal_length = aperture / (2.0 * np.tan(np.deg2rad(args.horizontal_fov) / 2.0))
    camera.set_focal_length(focal_length)
    camera.add_distance_to_image_plane_to_frame()

    for _ in range(args.warmup_frames):
        world.step(render=True)

    frame = camera.get_current_frame()
    rgba = frame.get("rgba")
    if rgba is None or np.asarray(rgba).size <= 1:
        rgba = frame.get("rgb")
    depth = frame.get("distance_to_image_plane")
    if rgba is None or depth is None:
        raise RuntimeError(f"camera frame is incomplete; available keys: {sorted(frame)}")

    rgb = np.asarray(rgba)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8, copy=False)
    depth_m = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)

    # Request ROS/optical axes so pinhole back-projection (+z forward, x
    # right, y down) and the saved transform use the same convention.
    optical_position, optical_orientation = camera.get_world_pose(camera_axes="ros")
    T_world_camera = matrix_from_pose(optical_position, optical_orientation)
    capture = RgbdCapture(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_world_camera=T_world_camera,
        camera_name="camera_0",
    )
    saved = capture.save(args.output)

    # Independent round-trip sanity check using Isaac's world-to-image API.
    # Depth reaches the visible surface rather than the object centre, so the
    # acceptance bound includes half the primitive size and rasterization.
    object_position = np.asarray(test_object.get_world_pose()[0], dtype=np.float64)
    image_xy = np.asarray(
        camera.get_image_coords_from_world_points(object_position[None, :]),
        dtype=np.float64,
    )[0]
    u = int(np.rint(image_xy[0]))
    v = int(np.rint(image_xy[1]))
    if not (0 <= u < width and 0 <= v < height):
        raise RuntimeError(f"test object projected outside image at ({u}, {v})")
    v0, v1 = max(0, v - 2), min(height, v + 3)
    u0, u1 = max(0, u - 2), min(width, u + 3)
    depth_window = depth_m[v0:v1, u0:u1]
    valid_window = depth_window[np.isfinite(depth_window) & (depth_window > 0.05)]
    if valid_window.size == 0:
        raise RuntimeError("no valid depth around projected test-object centre")
    sampled_depth = float(np.median(valid_window))
    one_pixel_depth = np.full(depth_m.shape, np.nan, dtype=np.float32)
    one_pixel_depth[v, u] = sampled_depth
    camera_point = depth_mask_to_points(one_pixel_depth, intrinsics, max_depth_m=3.0)
    reconstructed_world = transform_points(T_world_camera, camera_point)[0]
    reconstruction_error_m = float(np.linalg.norm(reconstructed_world - object_position))
    geometry_check = {
        "object_world_m": object_position.tolist(),
        "object_image_px": image_xy.tolist(),
        "sampled_depth_m": sampled_depth,
        "reconstructed_world_m": reconstructed_world.tolist(),
        "surface_to_center_error_m": reconstruction_error_m,
        "acceptance_bound_m": 0.20,
        "passed": reconstruction_error_m <= 0.20,
    }
    (saved / "geometry_check.json").write_text(
        json.dumps(geometry_check, indent=2) + "\n", encoding="utf-8"
    )
    if not geometry_check["passed"]:
        raise RuntimeError(
            "camera projection round-trip failed: "
            f"surface-to-centre error={reconstruction_error_m:.3f} m; "
            f"details saved to {saved / 'geometry_check.json'}"
        )
    print(f"saved calibrated RGB-D capture to {saved}")
    print(f"rgb={rgb.shape} depth={depth_m.shape} valid_depth={np.isfinite(depth_m).sum()}")
    print(f"intrinsics=\n{intrinsics}")
    print(f"T_world_camera=\n{T_world_camera}")
    print(f"camera geometry round-trip error={reconstruction_error_m:.3f} m")
finally:
    simulation_app.close()
