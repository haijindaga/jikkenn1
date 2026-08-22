"""Frame-explicit preparation and persistence for GraspGenX candidates.

Grasp generation itself stays in NVIDIA's official GraspGenX server.  This
module only validates the saved Isaac/SAM3 arrays, preserves their camera
frame, and converts returned poses into Isaac world and Panda tool frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# Copied as data (not code) from GraspGenX end2end/robots/franka_panda.yaml.
# GraspGenX: +X closing axis. Panda panda_hand: +Y closing axis.
T_GRASP_PANDA_HAND = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def prepare_scene_point_cloud(
    points_camera: np.ndarray, union_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare GraspGenX ``infer_scene_pc`` inputs without reprojection.

    Non-finite Isaac point-map pixels remain non-finite in the organized point
    cloud (the official server ignores them).  Their instance labels are set
    to background so the count reported here exactly matches points presented
    as instance 1 to the server.
    """
    points = np.asarray(points_camera)
    mask = np.asarray(union_mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_camera must have shape (H, W, 3), got {points.shape}")
    if mask.shape != points.shape[:2]:
        raise ValueError(
            f"union_mask shape {mask.shape} does not match point map {points.shape[:2]}"
        )

    valid = np.all(np.isfinite(points), axis=2)
    instance_pixels = mask & valid
    instance_mask = instance_pixels.astype(np.int32)
    return points.astype(np.float32, copy=False), instance_mask, int(instance_pixels.sum())


def split_target_from_scene(
    points_camera: np.ndarray,
    union_mask: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one organized Isaac point map into target and non-target points.

    This is the same caller-side contract used by GraspGenX's official scene
    collision demo: the target pixels are removed before checking the gripper
    mesh against the rest of the scene.  No depth reprojection is performed.
    """
    points = np.asarray(points_camera)
    mask = np.asarray(union_mask, dtype=bool)
    colors = np.asarray(rgb)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_camera must have shape (H, W, 3), got {points.shape}")
    if mask.shape != points.shape[:2]:
        raise ValueError(
            f"union_mask shape {mask.shape} does not match point map {points.shape[:2]}"
        )
    if colors.shape != points.shape:
        raise ValueError(f"rgb must have shape {points.shape}, got {colors.shape}")

    valid = np.all(np.isfinite(points), axis=2)
    target = valid & mask
    surrounding = valid & ~mask
    return (
        points[surrounding].astype(np.float32, copy=False),
        colors[surrounding].astype(np.uint8, copy=False),
        points[target].astype(np.float32, copy=False),
        colors[target].astype(np.uint8, copy=False),
    )


def transform_grasp_poses(
    grasps_camera: np.ndarray, T_world_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical GraspGenX poses to world and Panda tool poses."""
    grasps = np.asarray(grasps_camera, dtype=np.float64)
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"grasps_camera must have shape (N, 4, 4), got {grasps.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_world_camera must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(grasps)) or not np.all(np.isfinite(transform)):
        raise ValueError("grasp poses and T_world_camera must be finite")

    grasps_world = np.einsum("ij,njk->nik", transform, grasps)
    panda_hand_world = np.einsum("nij,jk->nik", grasps_world, T_GRASP_PANDA_HAND)
    return grasps_world.astype(np.float32), panda_hand_world.astype(np.float32)


def pose_quality(poses: np.ndarray) -> dict[str, float | bool]:
    """Report rigid-transform residuals without silently repairing poses."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {values.shape}")
    if values.shape[0] == 0:
        return {
            "finite": True,
            "max_rotation_orthogonality_error": 0.0,
            "max_rotation_determinant_error": 0.0,
            "max_homogeneous_row_error": 0.0,
        }
    rotations = values[:, :3, :3]
    orthogonality = np.matmul(np.transpose(rotations, (0, 2, 1)), rotations)
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "max_rotation_orthogonality_error": float(
            np.max(np.abs(orthogonality - np.eye(3)))
        ),
        "max_rotation_determinant_error": float(
            np.max(np.abs(np.linalg.det(rotations) - 1.0))
        ),
        "max_homogeneous_row_error": float(
            np.max(np.abs(values[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
        ),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def save_grasp_candidates(
    output: str | Path,
    *,
    grasps_camera: np.ndarray,
    scores: np.ndarray,
    branch_tags: list[str],
    T_world_camera: np.ndarray,
    input_point_count: int,
    parameters: dict[str, Any],
    server_health: dict[str, Any],
    server_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save raw and transformed candidates plus an explicit safety report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grasps_camera = np.asarray(grasps_camera, dtype=np.float32).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != grasps_camera.shape[0]:
        raise ValueError("scores and grasps_camera have different lengths")
    if branch_tags and len(branch_tags) != grasps_camera.shape[0]:
        raise ValueError("branch_tags and grasps_camera have different lengths")

    grasps_world, panda_hand_world = transform_grasp_poses(
        grasps_camera, T_world_camera
    )
    np.save(output / "grasps_camera.npy", grasps_camera)
    np.save(output / "scores.npy", scores)
    np.save(output / "grasps_world.npy", grasps_world)
    np.save(output / "panda_hand_world.npy", panda_hand_world)
    np.save(output / "T_grasp_panda_hand.npy", T_GRASP_PANDA_HAND)
    (output / "branch_tags.json").write_text(
        json.dumps(list(branch_tags), indent=2) + "\n", encoding="utf-8"
    )

    report: dict[str, Any] = {
        "status": "success" if grasps_camera.shape[0] > 0 else "no_candidates",
        "reference": {
            "implementation": "NVIDIA GraspGenX official ZMQ infer_scene_pc",
            "url": "https://github.com/NVlabs/GraspGenX/tree/main/client-server",
            "panda_frame_offset": "GraspGenX end2end/robots/franka_panda.yaml",
        },
        "input": {
            "frame": "opencv_optical_x_right_y_down_z_forward",
            "instance_id": 1,
            "valid_instance_points": int(input_point_count),
        },
        "parameters": _json_compatible(parameters),
        "server": {
            "health": _json_compatible(server_health),
            "metadata": _json_compatible(server_metadata),
        },
        "candidates": {
            "count": int(grasps_camera.shape[0]),
            "score_min": float(scores.min()) if scores.size else None,
            "score_max": float(scores.max()) if scores.size else None,
            "camera_pose_quality": pose_quality(grasps_camera),
            "world_pose_quality": pose_quality(grasps_world),
            "panda_hand_pose_quality": pose_quality(panda_hand_world),
        },
        "frames": {
            "grasps_camera.npy": "T_camera_graspgenx_grasp",
            "grasps_world.npy": "T_world_graspgenx_grasp",
            "panda_hand_world.npy": "T_world_panda_hand = T_world_graspgenx_grasp @ T_grasp_panda_hand",
        },
        "safety": {
            "reachability_checked": False,
            "collision_checked": False,
            "trajectory_planned": False,
            "safe_to_execute": False,
            "manual_review_required": True,
        },
    }
    (output / "graspgenx_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def save_collision_filter_results(
    output: str | Path,
    *,
    grasps_camera: np.ndarray,
    scores: np.ndarray,
    branch_tags: list[str],
    collision_free_mask: np.ndarray,
    T_world_camera: np.ndarray,
    collision_scene_camera: np.ndarray,
    scene_point_count_before_downsampling: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Persist the official point-cloud collision filter's exact inputs/results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grasps = np.asarray(grasps_camera, dtype=np.float32).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = np.asarray(collision_free_mask, dtype=bool).reshape(-1)
    collision_scene = np.asarray(collision_scene_camera, dtype=np.float32)
    if scores.shape[0] != grasps.shape[0] or keep.shape[0] != grasps.shape[0]:
        raise ValueError("grasps, scores and collision_free_mask must have equal lengths")
    if len(branch_tags) != grasps.shape[0]:
        raise ValueError("branch_tags and grasps_camera have different lengths")
    if collision_scene.ndim != 2 or collision_scene.shape[1] != 3:
        raise ValueError(
            f"collision_scene_camera must have shape (N, 3), got {collision_scene.shape}"
        )
    if not np.all(np.isfinite(collision_scene)):
        raise ValueError("collision_scene_camera must contain only finite points")

    kept_indices = np.flatnonzero(keep).astype(np.int32)
    filtered_camera = grasps[keep]
    filtered_scores = scores[keep]
    filtered_tags = [tag for tag, accepted in zip(branch_tags, keep) if accepted]
    filtered_world, filtered_panda_hand = transform_grasp_poses(
        filtered_camera, T_world_camera
    )
    best_filtered_index = (
        int(np.argmax(filtered_scores)) if filtered_scores.size else None
    )
    best_original_index = (
        int(kept_indices[best_filtered_index])
        if best_filtered_index is not None
        else None
    )

    np.save(output / "collision_free_mask.npy", keep)
    np.save(output / "kept_candidate_indices.npy", kept_indices)
    np.save(output / "collision_scene_camera.npy", collision_scene)
    np.save(output / "grasps_camera.npy", filtered_camera)
    np.save(output / "scores.npy", filtered_scores)
    np.save(output / "grasps_world.npy", filtered_world)
    np.save(output / "panda_hand_world.npy", filtered_panda_hand)
    (output / "branch_tags.json").write_text(
        json.dumps(filtered_tags, indent=2) + "\n", encoding="utf-8"
    )

    report: dict[str, Any] = {
        "status": "success" if kept_indices.size else "no_collision_free_candidates",
        "reference": {
            "implementation": "NVIDIA GraspGenX official filter_colliding_grasps",
            "demo": "GraspGenX scripts/demo_scene_pc.py",
            "url": "https://github.com/NVlabs/GraspGenX/blob/main/scripts/demo_scene_pc.py",
        },
        "frame": "opencv_optical_x_right_y_down_z_forward",
        "parameters": _json_compatible(parameters),
        "scene": {
            "target_pixels_removed": True,
            "points_before_downsampling": int(scene_point_count_before_downsampling),
            "points_used": int(collision_scene.shape[0]),
        },
        "candidates": {
            "before": int(grasps.shape[0]),
            "collision_free": int(kept_indices.size),
            "rejected_as_colliding": int(grasps.shape[0] - kept_indices.size),
            "best_kept_original_index": best_original_index,
            "best_score": (
                float(filtered_scores[best_filtered_index])
                if best_filtered_index is not None
                else None
            ),
            "camera_pose_quality": pose_quality(filtered_camera),
            "world_pose_quality": pose_quality(filtered_world),
            "panda_hand_pose_quality": pose_quality(filtered_panda_hand),
        },
        "safety": {
            "static_gripper_pose_vs_observed_scene_checked": True,
            "checker": "simple point-cloud nearest-distance filter",
            "approach_sweep_checked": False,
            "unobserved_space_checked": False,
            "robot_reachability_checked": False,
            "robot_self_collision_checked": False,
            "trajectory_planned": False,
            "safe_to_execute": False,
            "manual_review_required": True,
        },
    }
    (output / "collision_filter_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
