#!/usr/bin/env python3
"""Isaac Sim 5.1 smoke test: calibrated RGB-D and optional SAM3 segmentation.

Run from the repository root inside ``env_isaaclab``.  No grasping or motion
planning is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/capture_smoke"))
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(640, 480))
    parser.add_argument("--horizontal-fov", type=float, default=69.0)
    parser.add_argument("--sam3-prompt", help="Optional short noun phrase, for example 'blue block'")
    parser.add_argument("--sam3-output", type=Path)
    parser.add_argument("--sam3-model-id", default="facebook/sam3")
    parser.add_argument("--sam3-score-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-mask-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--sam3-allow-download",
        action="store_true",
        help="Allow Hugging Face network access instead of requiring the local model cache",
    )
    return parser.parse_args()


args = parse_args()

# Isaac requires SimulationApp construction before importing other Isaac modules.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

try:
    import numpy as np

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.capture import RgbdCapture
    from panda_handover.geometry import (
        look_at_quaternion_world,
        matrix_from_pose,
        transform_points,
    )
    from panda_handover.robot_state import RobotStateCapture

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    world.scene.add_default_ground_plane()
    panda = world.scene.add(Franka(prim_path="/World/Panda", name="panda"))

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
    # Calibration fixtures are fixed and rest exactly on the 0.70 m table
    # surface.  Keeping them off the Panda centreline prevents the asset's
    # default arm pose from occluding or contacting the fixtures.
    world.scene.add(
        FixedCuboid(
            prim_path="/World/TestObject",
            name="test_object",
            position=np.array([0.55, 0.20, 0.725]),
            scale=np.array([0.20, 0.05, 0.05]),
            color=np.array([0.1, 0.5, 0.9]),
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Obstacle",
            name="obstacle",
            position=np.array([0.55, -0.20, 0.75]),
            scale=np.array([0.10, 0.10, 0.10]),
            color=np.array([0.9, 0.2, 0.1]),
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
    # Follow Isaac Sim 5.1's own Camera point-cloud implementation: use pixel
    # centres and the built-in projection APIs.  The serialized transform is
    # independently checked against the official world-frame result so later
    # processes can safely consume T_world_camera.
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_vu = np.argwhere(valid)
    if valid_vu.size == 0:
        raise RuntimeError("camera produced no valid metric depth")
    valid_uv = np.column_stack(
        (valid_vu[:, 1].astype(np.float32) + 0.5, valid_vu[:, 0].astype(np.float32) + 0.5)
    )
    valid_depth = depth_m[valid_vu[:, 0], valid_vu[:, 1]].astype(np.float32, copy=False)

    official_camera_points_all = np.asarray(
        camera.get_camera_points_from_image_coords(valid_uv, valid_depth), dtype=np.float64
    )
    official_world_points_all = np.asarray(
        camera.get_world_points_from_image_coords(valid_uv, valid_depth), dtype=np.float64
    )
    points_camera = np.full((*depth_m.shape, 3), np.nan, dtype=np.float32)
    points_world = np.full((*depth_m.shape, 3), np.nan, dtype=np.float32)
    points_camera[valid_vu[:, 0], valid_vu[:, 1]] = official_camera_points_all
    points_world[valid_vu[:, 0], valid_vu[:, 1]] = official_world_points_all

    capture = RgbdCapture(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_world_camera=T_world_camera,
        camera_name="camera_0",
        points_camera=points_camera,
        points_world=points_world,
    )
    saved = capture.save(args.output)

    # cuRobo's official RobotSegmenter requires the measured articulation state
    # and the camera pose relative to the robot base. Save the actual Isaac
    # state rather than borrowing a default pose from a separate demo.
    robot_base_position, robot_base_orientation = panda.get_world_pose()
    robot_state = RobotStateCapture(
        joint_names=tuple(str(name) for name in panda.dof_names),
        joint_positions=np.asarray(panda.get_joint_positions(), dtype=np.float64),
        T_world_robot_base=matrix_from_pose(robot_base_position, robot_base_orientation),
        prim_path="/World/Panda",
    )
    robot_state_path = robot_state.save(saved, T_world_camera)

    sample_count = min(256, valid_vu.shape[0])
    sample_indices = np.linspace(0, valid_vu.shape[0] - 1, sample_count, dtype=np.int64)
    sampled_uv = valid_uv[sample_indices]
    official_camera_points = official_camera_points_all[sample_indices]
    official_world_points = official_world_points_all[sample_indices]
    reprojected_uv = np.asarray(
        camera.get_image_coords_from_world_points(official_world_points), dtype=np.float64
    )
    serialized_world_points = transform_points(T_world_camera, official_camera_points)

    pixel_errors = np.linalg.norm(reprojected_uv - sampled_uv, axis=1)
    world_errors = np.linalg.norm(serialized_world_points - official_world_points, axis=1)
    max_pixel_error = float(np.max(pixel_errors))
    max_world_error_m = float(np.max(world_errors))
    pixel_error_bound = 1e-3
    world_error_bound_m = 1e-5
    geometry_check = {
        "reference": "Isaac Sim 5.1 Camera projection and point-cloud APIs",
        "pixel_convention": "pixel centres at (u + 0.5, v + 0.5)",
        "sample_count": sample_count,
        "max_pixel_roundtrip_error_px": max_pixel_error,
        "pixel_error_bound_px": pixel_error_bound,
        "max_serialized_transform_error_m": max_world_error_m,
        "world_error_bound_m": world_error_bound_m,
        "passed": max_pixel_error <= pixel_error_bound and max_world_error_m <= world_error_bound_m,
    }
    (saved / "geometry_check.json").write_text(
        json.dumps(geometry_check, indent=2) + "\n", encoding="utf-8"
    )
    if not geometry_check["passed"]:
        raise RuntimeError(
            "camera projection/serialization validation failed: "
            f"pixel_error={max_pixel_error:.6g} px, world_error={max_world_error_m:.6g} m; "
            f"details saved to {saved / 'geometry_check.json'}"
        )
    segmentation_directory = None
    if args.sam3_prompt:
        from panda_handover.segmentation import infer_sam3_instances, save_segmentation_artifacts

        segmentation_directory = args.sam3_output or (args.output / "sam3")
        prediction = infer_sam3_instances(
            rgb,
            args.sam3_prompt,
            model_id=args.sam3_model_id,
            device=args.sam3_device,
            dtype=args.sam3_dtype,
            score_threshold=args.sam3_score_threshold,
            mask_threshold=args.sam3_mask_threshold,
            local_files_only=not args.sam3_allow_download,
        )
        segmentation_report = save_segmentation_artifacts(
            segmentation_directory,
            rgb=rgb,
            depth_m=depth_m,
            points_camera=points_camera,
            points_world=points_world,
            prediction=prediction,
            prompt=args.sam3_prompt,
            model_id=args.sam3_model_id,
            score_threshold=args.sam3_score_threshold,
            mask_threshold=args.sam3_mask_threshold,
        )
        if not segmentation_report["automatic_checks_passed"]:
            raise RuntimeError(
                "SAM3 returned no mask with valid 3D points; inspect "
                f"{segmentation_directory / 'segmentation_check.json'} and try a simpler noun phrase"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "capture_directory": str(saved),
                "segmentation_directory": (
                    str(segmentation_directory) if segmentation_directory is not None else None
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved calibrated RGB-D capture to {saved}", flush=True)
    print(f"rgb={rgb.shape} depth={depth_m.shape} valid_depth={valid.sum()}", flush=True)
    print(f"intrinsics=\n{intrinsics}", flush=True)
    print(f"T_world_camera=\n{T_world_camera}", flush=True)
    print(f"saved Panda state to {robot_state_path}", flush=True)
    print(
        f"camera validation max errors: {max_pixel_error:.6g} px, {max_world_error_m:.6g} m",
        flush=True,
    )
    if segmentation_directory is not None:
        print(f"saved SAM3 segmentation and masked point clouds to {segmentation_directory}", flush=True)
except Exception as exc:
    failure_traceback = traceback.format_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "failure",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": failure_traceback,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(failure_traceback, file=sys.stderr, flush=True)
    raise
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
