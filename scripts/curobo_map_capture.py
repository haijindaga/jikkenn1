#!/usr/bin/env python3
"""Build cuRobo's official TSDF/ESDF map from a saved Isaac RGB-D frame.

Run this in an isolated cuRobo v0.8 environment, with the GraspGenX server and
Isaac Sim stopped. This stage writes inspection artifacts only; it never plans
or executes a robot trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


CUROBO_REF_VALIDATED_BY_GRASPGENX = "057a96ffb1088531535f9915154f9d0dabd62428"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--esdf-voxel-size", type=float, default=0.01)
    parser.add_argument("--extent", type=float, nargs=3, default=(1.6, 1.6, 1.6))
    parser.add_argument("--grid-center", type=float, nargs=3, default=(0.5, 0.0, 0.75))
    parser.add_argument("--robot-distance-threshold", type=float, default=0.05)
    return parser.parse_args()


def save_previews(output: Path, robot_mask: np.ndarray, depth: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    Image.fromarray(robot_mask.astype(np.uint8) * 255, mode="L").save(
        output / "robot_mask.png"
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1e-3
        preview[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255, 0, 255).astype(
            np.uint8
        )
    Image.fromarray(255 - preview, mode="L").save(output / "mapping_depth.png")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.curobo_bridge import validate_mapping_inputs

    depth = np.load(args.capture / "depth_m.npy").astype(np.float32, copy=False)
    rgb = np.load(args.capture / "rgb.npy").astype(np.uint8, copy=False)
    intrinsics = np.load(args.capture / "intrinsics.npy").astype(np.float32, copy=False)
    target_mask = np.load(args.segmentation / "union_mask.npy").astype(bool, copy=False)
    T_robot_camera = np.load(args.capture / "T_robot_base_camera.npy").astype(
        np.float32, copy=False
    )
    joint_positions = np.load(args.capture / "panda_joint_positions.npy").astype(
        np.float32, copy=False
    )
    robot_report = json.loads(
        (args.capture / "robot_state.json").read_text(encoding="utf-8")
    )
    joint_names = tuple(robot_report["joint_names"])
    validate_mapping_inputs(depth, rgb, intrinsics, target_mask, T_robot_camera)
    if joint_positions.shape != (len(joint_names),):
        raise ValueError(
            f"saved Panda state has {len(joint_names)} names but {joint_positions.shape} positions"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo mapping requires CUDA, but torch.cuda.is_available() is false")
    from curobo import __version__ as curobo_version
    from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
    from curobo.types import CameraObservation, JointState, Pose

    device = torch.device(args.device)
    depth_tensor = torch.from_numpy(depth).to(device=device, dtype=torch.float32).unsqueeze(0)
    rgb_tensor = torch.from_numpy(rgb).to(device=device).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics).to(device=device).unsqueeze(0)
    camera_pose = Pose.from_matrix(
        torch.from_numpy(T_robot_camera).to(device=device, dtype=torch.float32).unsqueeze(0)
    )
    observation = CameraObservation(
        name="isaac_camera_0",
        depth_image=depth_tensor,
        rgb_image=rgb_tensor,
        intrinsics=intrinsics_tensor,
        pose=camera_pose,
        depth_to_meter=1.0,
    )
    state = JointState.from_position(
        torch.from_numpy(joint_positions).to(device=device).unsqueeze(0),
        joint_names=list(joint_names),
    )

    # Official cuRobo robot-sphere segmentation. No Isaac semantic labels or
    # simulator-only geometry are used.
    loaded_segmenter = RobotSegmenter.from_robot_file(
        args.robot,
        distance_threshold=args.robot_distance_threshold,
        use_cuda_graph=False,
    )
    # cuRobo v0.8's public constructor defaults to bfloat16 while its tensor
    # preflight accepts float16 or float32. Select the accuracy-oriented,
    # officially supported float32 path without patching cuRobo itself.
    segmenter = RobotSegmenter(
        loaded_segmenter.kinematics,
        distance_threshold=args.robot_distance_threshold,
        use_cuda_graph=False,
        ops_dtype=torch.float32,
    )
    robot_mask_tensor, depth_without_robot = segmenter.get_robot_mask(observation, state)
    target_mask_tensor = torch.from_numpy(target_mask).to(device=device).unsqueeze(0)
    mapping_depth = torch.where(target_mask_tensor, 0.0, depth_without_robot)

    # This is cuRobo's official depth filter. Reapply both masks afterwards so
    # smoothing can never fill target/robot pixels back into the obstacle map.
    depth_filter = FilterDepth(
        image_shape=depth.shape,
        depth_minimum_distance=0.05,
        depth_maximum_distance=3.0,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
        device=args.device,
    )
    mapping_depth, _ = depth_filter(mapping_depth)
    exclusion_mask = torch.logical_or(robot_mask_tensor, target_mask_tensor)
    mapping_depth = torch.where(exclusion_mask, 0.0, mapping_depth)

    mapper_cfg = MapperCfg(
        extent_meters_xyz=tuple(args.extent),
        voxel_size=args.voxel_size,
        esdf_voxel_size=args.esdf_voxel_size,
        grid_center=torch.tensor(args.grid_center, device=device, dtype=torch.float32),
        truncation_distance=args.voxel_size * 6.0,
        minimum_tsdf_weight=0.1,
        depth_minimum_distance=0.05,
        depth_maximum_distance=3.0,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        enable_static=False,
        num_cameras=1,
        image_height=depth.shape[0],
        image_width=depth.shape[1],
        device=args.device,
    )
    mapper = Mapper(mapper_cfg)
    map_observation = CameraObservation(
        name="isaac_camera_0_without_robot_or_target",
        depth_image=mapping_depth,
        rgb_image=rgb_tensor,
        intrinsics=intrinsics_tensor,
        pose=camera_pose,
        depth_to_meter=1.0,
    )
    mapper.integrate(map_observation)
    voxel_grid = mapper.compute_esdf()
    occupied = mapper.extract_occupied_voxels(surface_only=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    robot_mask = robot_mask_tensor[0].detach().cpu().numpy().astype(bool)
    mapping_depth_np = mapping_depth[0].detach().cpu().numpy().astype(np.float32)
    occupied_points = occupied.centers.detach().cpu().numpy().astype(np.float32)
    occupied_colors = occupied.colors_uint8().detach().cpu().numpy().astype(np.uint8)
    esdf = voxel_grid.feature_tensor.detach().cpu().numpy()
    np.save(output / "robot_mask.npy", robot_mask)
    np.save(output / "mapping_depth_m.npy", mapping_depth_np)
    np.save(output / "occupied_points_robot_base.npy", occupied_points)
    np.save(output / "occupied_colors.npy", occupied_colors)
    np.save(output / "esdf_features.npy", esdf)
    save_previews(output, robot_mask, mapping_depth_np)

    finite_esdf = np.isfinite(esdf)
    automatic_checks = {
        "robot_pixels_removed": int(robot_mask.sum()) > 0,
        "target_pixels_removed": int(target_mask.sum()) > 0,
        "mapping_depth_pixels": int((mapping_depth_np > 0.0).sum()) > 0,
        "occupied_surface_voxels": int(occupied_points.shape[0]) > 0,
        "esdf_has_finite_values": bool(finite_esdf.any()),
    }
    report = {
        "status": "success" if all(automatic_checks.values()) else "failed_checks",
        "reference": {
            "curobo_version": str(curobo_version),
            "graspgenx_validated_curobo_commit": CUROBO_REF_VALIDATED_BY_GRASPGENX,
            "apis": ["RobotSegmenter", "FilterDepth", "Mapper.compute_esdf"],
        },
        "frames": {
            "camera_pose": "T_robot_base_camera",
            "map": "franka robot base",
        },
        "parameters": {
            "voxel_size_m": args.voxel_size,
            "esdf_voxel_size_m": args.esdf_voxel_size,
            "extent_m": list(args.extent),
            "grid_center_robot_base_m": list(args.grid_center),
            "robot_distance_threshold_m": args.robot_distance_threshold,
            "robot_segmentation_ops_dtype": "float32",
            "minimum_tsdf_weight": 0.1,
            "input_frames": 1,
        },
        "counts": {
            "valid_depth_before": int((np.isfinite(depth) & (depth > 0.0)).sum()),
            "robot_mask_pixels": int(robot_mask.sum()),
            "target_mask_pixels": int(target_mask.sum()),
            "mapping_depth_pixels": int((mapping_depth_np > 0.0).sum()),
            "occupied_surface_voxels": int(occupied_points.shape[0]),
        },
        "esdf": {
            "shape": list(esdf.shape),
            "dims_m": list(voxel_grid.dims),
            "pose_xyzw_or_wxyz_as_curobo": list(voxel_grid.pose),
            "voxel_size_m": float(voxel_grid.voxel_size),
            "finite_fraction": float(finite_esdf.mean()),
            "min_m": float(esdf[finite_esdf].min()) if finite_esdf.any() else None,
            "max_m": float(esdf[finite_esdf].max()) if finite_esdf.any() else None,
        },
        "automatic_checks": automatic_checks,
        "safe_to_plan": False,
        "next_gate": (
            "Visually inspect robot_mask.png, mapping_depth.png, and occupied_points_robot_base.npy "
            "before enabling cuRobo plan_grasp"
        ),
        "unknown_environment_contract": {
            "isaac_semantic_labels_used": False,
            "isaac_ground_truth_obstacle_geometry_used": False,
            "target_removed_with_sam3_mask": True,
            "robot_removed_with_curobo_kinematics": True,
        },
    }
    report_path = output / "esdf_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"cuRobo {curobo_version}")
    print(f"robot mask pixels: {robot_mask.sum()}")
    print(f"mapping depth pixels: {(mapping_depth_np > 0.0).sum()}")
    print(f"occupied surface voxels: {occupied_points.shape[0]}")
    print(f"saved: {report_path}")
    if not all(automatic_checks.values()):
        raise RuntimeError(f"automatic ESDF checks failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
