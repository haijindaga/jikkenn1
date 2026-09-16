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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT


LAYOUT = DEFAULT_TABLETOP_LAYOUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/capture_smoke"))
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(640, 480))
    parser.add_argument("--horizontal-fov", type=float, default=69.0)
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=LAYOUT.camera_position_m,
        help="Camera position in Isaac world metres",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=LAYOUT.camera_target_m,
        help="World point at which the camera looks",
    )
    parser.add_argument(
        "--camera-name",
        default="camera_0",
        help="Capture directory name; use a unique name for each viewpoint",
    )
    parser.add_argument(
        "--scene-usd",
        type=Path,
        help="Open an authored USD scene instead of generating the legacy block scene",
    )
    parser.add_argument("--panda-prim", default="/World/Panda")
    parser.add_argument("--table-prim", default="/World/Table")
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument("--camera-prim", default="/World/camera_0")
    parser.add_argument("--sam3-prompt", help="Optional short noun phrase, for example 'blue block'")
    parser.add_argument(
        "--sam3-part-prompt",
        action="append",
        default=[],
        help="Additional part phrase; repeat for blade/handle and keep --sam3-prompt as the whole object",
    )
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
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.experimental.utils import stage as stage_utils
    from pxr import PhysxSchema, Usd, UsdPhysics

    from panda_handover.capture import RgbdCapture
    from panda_handover.geometry import (
        look_at_quaternion_world,
        matrix_from_pose,
        transform_points,
    )
    from panda_handover.robot_state import RobotStateCapture

    scene_usd = None
    if args.scene_usd is not None:
        scene_usd = args.scene_usd.expanduser().resolve()
        if not scene_usd.is_file():
            raise FileNotFoundError(f"authored scene USD does not exist: {scene_usd}")
        if scene_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError("--scene-usd must end in .usd, .usda, or .usdc")
        stage_opened, stage = stage_utils.open_stage(str(scene_usd))
        if not stage_opened or stage is None:
            raise RuntimeError(f"Isaac Sim could not open authored scene: {scene_usd}")
        required_prim_paths = (
            args.panda_prim,
            args.table_prim,
            args.target_prim,
            args.camera_prim,
        )
        missing_prims = [
            prim_path
            for prim_path in required_prim_paths
            if not stage.GetPrimAtPath(prim_path).IsValid()
        ]
        if missing_prims:
            raise RuntimeError(
                "authored scene is missing required prims: " + ", ".join(missing_prims)
            )

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    if scene_usd is None:
        world.scene.add_default_ground_plane(z_position=LAYOUT.ground_z_m)
        panda = world.scene.add(
            Franka(
                prim_path=args.panda_prim,
                name="panda",
                position=np.asarray(LAYOUT.robot_base_position_m),
            )
        )

        # Follow the Isaac Lab Franka lift convention: the robot mount and
        # tabletop share z=0 and the room floor is below them.
        world.scene.add(
            FixedCuboid(
                prim_path=args.table_prim,
                name="table",
                position=np.asarray(LAYOUT.table_center_m),
                scale=np.asarray(LAYOUT.table_size_m),
                color=np.array([0.45, 0.32, 0.20]),
            )
        )
        legacy_target_prim = "/World/TestObject"
        world.scene.add(
            FixedCuboid(
                prim_path=legacy_target_prim,
                name="test_object",
                position=np.asarray(LAYOUT.target_center_m),
                scale=np.asarray(LAYOUT.target_size_m),
                color=np.array([0.1, 0.5, 0.9]),
            )
        )
        target_prim_path = legacy_target_prim
        target_asset = {
            "kind": "fixed_cuboid",
            "source": "Isaac Sim FixedCuboid",
            "usd_path": None,
            "physics_ready": False,
        }
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Obstacle",
                name="obstacle",
                position=np.asarray(LAYOUT.obstacle_center_m),
                scale=np.asarray(LAYOUT.obstacle_size_m),
                color=np.array([0.9, 0.2, 0.1]),
            )
        )
    else:
        panda = world.scene.add(
            SingleArticulation(
                prim_path=args.panda_prim,
                name="panda",
            )
        )
        target_prim_path = args.target_prim
        target_prim = stage.GetPrimAtPath(target_prim_path)
        target_prims = tuple(Usd.PrimRange(target_prim))
        physics_apis = {
            "rigid_body": any(
                prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in target_prims
            ),
            "collision": any(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                for prim in target_prims
            ),
            "mass": any(prim.HasAPI(UsdPhysics.MassAPI) for prim in target_prims),
        }
        if not all(physics_apis.values()):
            missing_apis = [name for name, present in physics_apis.items() if not present]
            raise RuntimeError(
                f"saved-scene target {target_prim_path} is not physics-ready; "
                f"missing USD APIs: {missing_apis}"
            )
        target_asset = {
            "kind": "authored_usd_scene_object",
            "source": str(scene_usd),
            "prim_path": target_prim_path,
            "physics_ready": True,
            "physics_apis": physics_apis,
            "settled_by_physics": True,
        }

    camera_position = np.asarray(args.camera_position, dtype=np.float64)
    camera_target = np.asarray(args.camera_target, dtype=np.float64)
    camera_orientation = look_at_quaternion_world(camera_position, camera_target)
    width, height = args.resolution
    if scene_usd is None:
        camera = Camera(
            prim_path=f"/World/{args.camera_name}",
            position=camera_position,
            orientation=camera_orientation,
            frequency=30,
            resolution=(width, height),
        )
    else:
        camera = Camera(
            prim_path=args.camera_prim,
            frequency=30,
            resolution=(width, height),
        )

    world.reset()
    camera.initialize()
    if scene_usd is None:
        camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")
        camera.set_clipping_range(0.05, 3.0)
        aperture = float(camera.get_horizontal_aperture())
        focal_length = aperture / (2.0 * np.tan(np.deg2rad(args.horizontal_fov) / 2.0))
        camera.set_focal_length(focal_length)
    camera.add_distance_to_image_plane_to_frame()

    for _ in range(args.warmup_frames):
        world.step(render=True)

    target_aabb = np.asarray(
        compute_aabb(create_bbox_cache(), target_prim_path, include_children=True),
        dtype=np.float64,
    )
    if target_aabb.shape != (6,) or not np.all(np.isfinite(target_aabb)):
        raise RuntimeError(f"invalid target AABB after settling: {target_aabb}")
    table_aabb = np.asarray(
        compute_aabb(create_bbox_cache(), args.table_prim, include_children=True),
        dtype=np.float64,
    )
    if table_aabb.shape != (6,) or not np.all(np.isfinite(table_aabb)):
        raise RuntimeError(f"invalid table AABB: {table_aabb}")
    table_min = table_aabb[:3]
    table_max = table_aabb[3:]
    table_top_z_m = float(table_max[2])
    target_extent = target_aabb[3:] - target_aabb[:3]
    runtime_target_checks = {
        "aabb_extent_is_positive": bool(np.all(target_extent > 1e-4)),
        "footprint_is_on_table": bool(
            target_aabb[0] >= table_min[0]
            and target_aabb[3] <= table_max[0]
            and target_aabb[1] >= table_min[1]
            and target_aabb[4] <= table_max[1]
        ),
        "bottom_is_not_below_table": bool(target_aabb[2] >= table_top_z_m - 0.01),
        "bottom_is_near_table_after_settling": bool(
            target_aabb[2] <= table_top_z_m + 0.03
        ),
    }
    target_asset["table_aabb_world_m"] = table_aabb.astype(float).tolist()
    target_asset["settled_aabb_world_m"] = target_aabb.astype(float).tolist()
    target_asset["settled_extent_m"] = target_extent.astype(float).tolist()
    target_asset["automatic_checks"] = runtime_target_checks

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
        camera_name=args.camera_name,
        points_camera=points_camera,
        points_world=points_world,
    )
    saved = capture.save(args.output)

    if scene_usd is None:
        scene_layout_report = LAYOUT.validation_report()
        scene_layout_report["scene_source"] = {
            "kind": "generated_legacy_smoke_scene",
            "scene_usd": None,
        }
        scene_layout_report["runtime_target"] = target_asset
        scene_layout_report["automatic_checks"]["runtime_target_is_valid"] = bool(
            all(runtime_target_checks.values())
        )
        scene_layout_report["status"] = (
            "success"
            if all(scene_layout_report["automatic_checks"].values())
            else "failure"
        )
    else:
        scene_layout_report = {
            "status": "success" if all(runtime_target_checks.values()) else "failure",
            "reference": {
                "scene_loading": "Isaac Sim 5.1 core experimental stage.open_stage",
                "asset_composition": "OpenUSD referenced physics-ready target",
            },
            "scene_source": {
                "kind": "authored_usd_scene",
                "scene_usd": str(scene_usd),
                "panda_prim": args.panda_prim,
                "table_prim": args.table_prim,
                "target_prim": target_prim_path,
                "camera_prim": args.camera_prim,
            },
            "runtime_target": target_asset,
            "automatic_checks": {
                "saved_scene_opened": True,
                "target_is_physics_ready": bool(all(physics_apis.values())),
                "runtime_target_is_valid": bool(all(runtime_target_checks.values())),
            },
        }
        scene_layout_report["status"] = (
            "success"
            if all(scene_layout_report["automatic_checks"].values())
            else "failure"
        )
    scene_layout_path = saved / "scene_layout.json"
    scene_layout_path.write_text(
        json.dumps(scene_layout_report, indent=2) + "\n", encoding="utf-8"
    )
    if scene_layout_report["status"] != "success":
        raise RuntimeError(f"tabletop scene validation failed; inspect {scene_layout_path}")

    # cuRobo's official RobotSegmenter requires the measured articulation state
    # and the camera pose relative to the robot base. Save the actual Isaac
    # state rather than borrowing a default pose from a separate demo.
    robot_base_position, robot_base_orientation = panda.get_world_pose()
    robot_state = RobotStateCapture(
        joint_names=tuple(str(name) for name in panda.dof_names),
        joint_positions=np.asarray(panda.get_joint_positions(), dtype=np.float64),
        T_world_robot_base=matrix_from_pose(robot_base_position, robot_base_orientation),
        prim_path=args.panda_prim,
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
    if args.sam3_part_prompt and not args.sam3_prompt:
        raise ValueError("--sam3-part-prompt requires --sam3-prompt for the whole object")
    segmentation_prompt_directories = None
    if args.sam3_prompt:
        from panda_handover.segmentation import (
            infer_sam3_prompts,
            prompt_slug,
            save_prompt_overlap_report,
            save_segmentation_artifacts,
        )

        segmentation_directory = args.sam3_output or (args.output / "sam3")
        prompts = [args.sam3_prompt, *args.sam3_part_prompt]
        predictions = infer_sam3_prompts(
            rgb,
            prompts,
            model_id=args.sam3_model_id,
            device=args.sam3_device,
            dtype=args.sam3_dtype,
            score_threshold=args.sam3_score_threshold,
            mask_threshold=args.sam3_mask_threshold,
            local_files_only=not args.sam3_allow_download,
        )
        segmentation_prompt_directories = {}
        segmentation_reports = {}
        for index, prompt in enumerate(prompts):
            prompt_directory = (
                segmentation_directory
                if index == 0
                else segmentation_directory / "parts" / prompt_slug(prompt)
            )
            segmentation_prompt_directories[prompt] = str(prompt_directory)
            segmentation_reports[prompt] = save_segmentation_artifacts(
                prompt_directory,
                rgb=rgb,
                depth_m=depth_m,
                points_camera=points_camera,
                points_world=points_world,
                prediction=predictions[prompt],
                prompt=prompt,
                model_id=args.sam3_model_id,
                score_threshold=args.sam3_score_threshold,
                mask_threshold=args.sam3_mask_threshold,
            )
        overlap_report = save_prompt_overlap_report(
            segmentation_directory,
            predictions,
            tuple(rgb.shape[:2]),
        )
        if not all(
            report["automatic_checks_passed"] for report in segmentation_reports.values()
        ):
            raise RuntimeError(
                "SAM3 returned no mask with valid 3D points for at least one prompt; inspect "
                f"{segmentation_directory} before changing prompts or thresholds"
            )
        if len(prompts) > 1 and not overlap_report["automatic_checks_passed"]:
            raise RuntimeError(f"SAM3 multi-prompt gate failed; inspect {segmentation_directory}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "capture_directory": str(saved),
                "scene_layout": str(scene_layout_path),
                "scene_usd": str(scene_usd) if scene_usd is not None else None,
                "target_prim": target_prim_path,
                "segmentation_directory": (
                    str(segmentation_directory) if segmentation_directory is not None else None
                ),
                "segmentation_prompt_directories": segmentation_prompt_directories,
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
    print(f"saved validated tabletop layout to {scene_layout_path}", flush=True)
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
                "scene_usd": str(args.scene_usd) if args.scene_usd is not None else None,
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
