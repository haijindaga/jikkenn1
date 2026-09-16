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
    parser.add_argument(
        "--capture",
        type=Path,
        nargs="+",
        required=True,
        help="One or more calibrated camera directories",
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        nargs="+",
        required=True,
        help="Matching SAM3 directories, in the same order as --capture",
    )
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


def pair_capture_inputs(
    captures: list[Path], segmentations: list[Path]
) -> list[tuple[Path, Path]]:
    """Pair calibrated views with their masks without silently truncating either list."""
    if len(captures) != len(segmentations):
        raise ValueError(
            "--capture and --segmentation must contain the same number of paths; "
            f"got {len(captures)} and {len(segmentations)}"
        )
    if not captures:
        raise ValueError("at least one capture/segmentation pair is required")
    return list(zip(captures, segmentations))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.curobo_bridge import (
        extent_covers_requested,
        validate_mapping_inputs,
    )

    input_pairs = pair_capture_inputs(args.capture, args.segmentation)

    views = []
    image_shape = None
    reference_world_robot_base = None
    for capture_path, segmentation_path in input_pairs:
        depth = np.load(capture_path / "depth_m.npy").astype(np.float32, copy=False)
        rgb = np.load(capture_path / "rgb.npy").astype(np.uint8, copy=False)
        intrinsics = np.load(capture_path / "intrinsics.npy").astype(
            np.float32, copy=False
        )
        target_mask = np.load(segmentation_path / "union_mask.npy").astype(
            bool, copy=False
        )
        T_robot_camera = np.load(capture_path / "T_robot_base_camera.npy").astype(
            np.float32, copy=False
        )
        T_world_robot_base = np.load(capture_path / "T_world_robot_base.npy").astype(
            np.float64, copy=False
        )
        joint_positions = np.load(capture_path / "panda_joint_positions.npy").astype(
            np.float32, copy=False
        )
        robot_report = json.loads(
            (capture_path / "robot_state.json").read_text(encoding="utf-8")
        )
        joint_names = tuple(robot_report["joint_names"])
        validate_mapping_inputs(depth, rgb, intrinsics, target_mask, T_robot_camera)
        if joint_positions.shape != (len(joint_names),):
            raise ValueError(
                f"{capture_path}: saved Panda state has {len(joint_names)} names "
                f"but {joint_positions.shape} positions"
            )
        if image_shape is None:
            image_shape = depth.shape
        elif depth.shape != image_shape:
            raise ValueError(
                f"all views must have one image shape; {capture_path} has {depth.shape}, "
                f"expected {image_shape}"
            )
        if reference_world_robot_base is None:
            reference_world_robot_base = T_world_robot_base
        elif not np.allclose(
            T_world_robot_base, reference_world_robot_base, atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                f"{capture_path}: Panda base moved between captures; "
                "multi-view fusion requires one fixed robot-base map frame"
            )
        views.append(
            {
                "capture": capture_path,
                "segmentation": segmentation_path,
                "depth": depth,
                "rgb": rgb,
                "intrinsics": intrinsics,
                "target_mask": target_mask,
                "T_robot_camera": T_robot_camera,
                "T_world_robot_base": T_world_robot_base,
                "joint_positions": joint_positions,
                "joint_names": joint_names,
            }
        )

    if image_shape is None:
        raise RuntimeError("validated capture list unexpectedly produced no views")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo mapping requires CUDA, but torch.cuda.is_available() is false")
    from curobo import __version__ as curobo_version
    from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
    from curobo.types import CameraObservation, JointState, Pose

    device = torch.device(args.device)

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
    depth_filter = FilterDepth(
        image_shape=image_shape,
        depth_minimum_distance=0.05,
        depth_maximum_distance=3.0,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
        device=args.device,
    )
    mapper_cfg = MapperCfg(
        extent_meters_xyz=tuple(args.extent),
        voxel_size=args.voxel_size,
        esdf_voxel_size=args.esdf_voxel_size,
        # cuRobo exposes a separate ESDF extent. State it explicitly so the
        # collision grid cannot silently use a smaller default than the TSDF.
        extent_esdf_meters_xyz=tuple(args.extent),
        grid_center=torch.tensor(args.grid_center, device=device, dtype=torch.float32),
        truncation_distance=args.voxel_size * 6.0,
        minimum_tsdf_weight=0.1,
        depth_minimum_distance=0.05,
        depth_maximum_distance=3.0,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        enable_static=False,
        # Views are integrated sequentially. Keeping one camera slot follows
        # Mapper's official ``for obs in observations: integrate(obs)`` path
        # and avoids allocating an N-camera scratch buffer on an 8 GB GPU.
        num_cameras=1,
        image_height=image_shape[0],
        image_width=image_shape[1],
        device=args.device,
    )
    mapper = Mapper(mapper_cfg)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    views_output = output / "views"
    views_output.mkdir(exist_ok=True)
    robot_masks = []
    mapping_depths = []
    view_reports = []

    for view_index, view in enumerate(views):
        depth_tensor = torch.from_numpy(view["depth"]).to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)
        rgb_tensor = torch.from_numpy(view["rgb"]).to(device=device).unsqueeze(0)
        intrinsics_tensor = torch.from_numpy(view["intrinsics"]).to(
            device=device
        ).unsqueeze(0)
        camera_pose = Pose.from_matrix(
            torch.from_numpy(view["T_robot_camera"])
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
        )
        observation = CameraObservation(
            name=f"isaac_camera_{view_index}",
            depth_image=depth_tensor,
            rgb_image=rgb_tensor,
            intrinsics=intrinsics_tensor,
            pose=camera_pose,
            depth_to_meter=1.0,
        )
        state = JointState.from_position(
            torch.from_numpy(view["joint_positions"]).to(device=device).unsqueeze(0),
            joint_names=list(view["joint_names"]),
        )

        robot_mask_tensor, depth_without_robot = segmenter.get_robot_mask(
            observation, state
        )
        target_mask_tensor = torch.from_numpy(view["target_mask"]).to(
            device=device
        ).unsqueeze(0)
        mapping_depth = torch.where(target_mask_tensor, 0.0, depth_without_robot)

        # This is cuRobo's official depth filter. Reapply both masks afterwards
        # so smoothing can never fill target/robot pixels back into the map.
        mapping_depth, _ = depth_filter(mapping_depth)
        exclusion_mask = torch.logical_or(robot_mask_tensor, target_mask_tensor)
        mapping_depth = torch.where(exclusion_mask, 0.0, mapping_depth)
        map_observation = CameraObservation(
            name=f"isaac_camera_{view_index}_without_robot_or_target",
            depth_image=mapping_depth,
            rgb_image=rgb_tensor,
            intrinsics=intrinsics_tensor,
            pose=camera_pose,
            depth_to_meter=1.0,
        )
        mapper.integrate(map_observation)

        robot_mask = robot_mask_tensor[0].detach().cpu().numpy().astype(bool)
        mapping_depth_np = mapping_depth[0].detach().cpu().numpy().astype(np.float32)
        robot_masks.append(robot_mask)
        mapping_depths.append(mapping_depth_np)
        view_directory = views_output / f"camera_{view_index}"
        view_directory.mkdir(exist_ok=True)
        np.save(view_directory / "robot_mask.npy", robot_mask)
        np.save(view_directory / "mapping_depth_m.npy", mapping_depth_np)
        save_previews(view_directory, robot_mask, mapping_depth_np)
        view_reports.append(
            {
                "index": view_index,
                "capture": str(view["capture"]),
                "segmentation": str(view["segmentation"]),
                "valid_depth_before": int(
                    (np.isfinite(view["depth"]) & (view["depth"] > 0.0)).sum()
                ),
                "robot_mask_pixels": int(robot_mask.sum()),
                "target_mask_pixels": int(view["target_mask"].sum()),
                "mapping_depth_pixels": int((mapping_depth_np > 0.0).sum()),
            }
        )

    voxel_grid = mapper.compute_esdf()
    occupied = mapper.extract_occupied_voxels(surface_only=True)

    occupied_points = occupied.centers.detach().cpu().numpy().astype(np.float32)
    occupied_colors = occupied.colors_uint8().detach().cpu().numpy().astype(np.uint8)
    esdf = voxel_grid.feature_tensor.detach().cpu().numpy()
    actual_esdf_extent = np.asarray(voxel_grid.dims, dtype=np.float64)
    esdf_covers_requested_extent = extent_covers_requested(
        actual_esdf_extent, args.extent
    )
    # Keep the established HxW shape for a one-view run. Multi-view outputs
    # are NxHxW; individual HxW arrays are always available under views/.
    robot_masks_array = np.stack(robot_masks)
    mapping_depths_array = np.stack(mapping_depths)
    np.save(
        output / "robot_mask.npy",
        robot_masks_array[0] if len(views) == 1 else robot_masks_array,
    )
    np.save(
        output / "mapping_depth_m.npy",
        mapping_depths_array[0] if len(views) == 1 else mapping_depths_array,
    )
    np.save(output / "occupied_points_robot_base.npy", occupied_points)
    np.save(output / "occupied_colors.npy", occupied_colors)
    np.save(output / "esdf_features.npy", esdf)
    finite_esdf = np.isfinite(esdf)
    automatic_checks = {
        "robot_pixels_removed_in_every_view": all(
            report["robot_mask_pixels"] > 0 for report in view_reports
        ),
        "target_pixels_removed_in_every_view": all(
            report["target_mask_pixels"] > 0 for report in view_reports
        ),
        "mapping_depth_pixels_in_every_view": all(
            report["mapping_depth_pixels"] > 0 for report in view_reports
        ),
        "occupied_surface_voxels": int(occupied_points.shape[0]) > 0,
        "esdf_has_finite_values": bool(finite_esdf.any()),
        "esdf_covers_requested_extent": esdf_covers_requested_extent,
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
            "extent_esdf_m": list(args.extent),
            "grid_center_robot_base_m": list(args.grid_center),
            "robot_distance_threshold_m": args.robot_distance_threshold,
            "robot_segmentation_ops_dtype": "float32",
            "minimum_tsdf_weight": 0.1,
            "input_frames": len(views),
            "integration_mode": "sequential_single_camera_slot",
        },
        "counts": {
            "valid_depth_before_total": sum(
                report["valid_depth_before"] for report in view_reports
            ),
            "robot_mask_pixels_total": sum(
                report["robot_mask_pixels"] for report in view_reports
            ),
            "target_mask_pixels_total": sum(
                report["target_mask_pixels"] for report in view_reports
            ),
            "mapping_depth_pixels_total": sum(
                report["mapping_depth_pixels"] for report in view_reports
            ),
            "occupied_surface_voxels": int(occupied_points.shape[0]),
        },
        "views": view_reports,
        "esdf": {
            "shape": list(esdf.shape),
            "dims_m": actual_esdf_extent.tolist(),
            "requested_dims_m": list(args.extent),
            "pose_xyzw_or_wxyz_as_curobo": list(voxel_grid.pose),
            "voxel_size_m": float(voxel_grid.voxel_size),
            "finite_fraction": float(finite_esdf.mean()),
            "min_m": float(esdf[finite_esdf].min()) if finite_esdf.any() else None,
            "max_m": float(esdf[finite_esdf].max()) if finite_esdf.any() else None,
        },
        "automatic_checks": automatic_checks,
        "safe_to_plan": False,
        "next_gate": (
            "Visually inspect views/camera_*/robot_mask.png, "
            "views/camera_*/mapping_depth.png, and occupied_points_robot_base.npy "
            "before evaluating observed-space coverage"
        ),
        "unknown_environment_contract": {
            "isaac_semantic_labels_used": False,
            "isaac_ground_truth_obstacle_geometry_used": False,
            "target_removed_with_sam3_mask": True,
            "robot_removed_with_curobo_kinematics": True,
            "unobserved_space_proven_occupied": False,
            "multi_view_reduces_unobserved_space": len(views) > 1,
        },
    }
    report_path = output / "esdf_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"cuRobo {curobo_version}")
    print(f"integrated views: {len(views)}")
    print(
        "robot mask pixels: "
        + ", ".join(str(report["robot_mask_pixels"]) for report in view_reports)
    )
    print(
        "mapping depth pixels: "
        + ", ".join(str(report["mapping_depth_pixels"]) for report in view_reports)
    )
    print(f"occupied surface voxels: {occupied_points.shape[0]}")
    print(f"saved: {report_path}")
    if not all(automatic_checks.values()):
        raise RuntimeError(f"automatic ESDF checks failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
