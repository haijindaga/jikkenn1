#!/usr/bin/env python3
"""Probe the official AnyGrasp SDK on a saved calibrated RGB-D capture.

This deliberately stops at candidate generation and visualization.  AnyGrasp's
GraspNet frame is not silently treated as the Panda hand frame; execution stays
disabled until that robot-specific transform has been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official AnyGrasp detector on an existing RGB-D capture, "
            "steered by an element-aligned SAM3 mask."
        )
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anygrasp-root",
        type=Path,
        help="Optional official anygrasp_sdk checkout used for provenance",
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--min-region-points", type=int, default=100)
    parser.add_argument("--max-gripper-width", type=float, default=0.08)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--dense-grasp", action="store_true")
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Open the official Open3D-style candidate visualization",
    )
    args = parser.parse_args()
    if args.topk <= 0 or args.min_region_points <= 0:
        parser.error("--topk and --min-region-points must be positive")
    if not 0.0 < args.max_gripper_width <= 0.1:
        parser.error("--max-gripper-width must be in (0, 0.1]")
    if args.gripper_height <= 0.0:
        parser.error("--gripper-height must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path | None) -> str | None:
    if root is None or not root.is_dir():
        return None
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def prepare_anygrasp_inputs(
    points_camera: np.ndarray,
    rgb: np.ndarray,
    union_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the organized capture while preserving mask-point alignment."""
    points = np.asarray(points_camera)
    colors = np.asarray(rgb)
    mask = np.asarray(union_mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_camera must have shape (H,W,3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"rgb must have shape {points.shape}, got {colors.shape}")
    if mask.shape != points.shape[:2]:
        raise ValueError(
            f"union_mask shape {mask.shape} does not match point map {points.shape[:2]}"
        )
    valid = np.all(np.isfinite(points), axis=2)
    return (
        points[valid].astype(np.float32, copy=False),
        colors[valid].astype(np.float32, copy=False) / 255.0,
        mask[valid].astype(bool, copy=False),
        valid,
    )


def _extract_grasp_arrays(grasp_group: Any) -> dict[str, np.ndarray]:
    count = len(grasp_group)
    rotations = np.asarray(grasp_group.rotation_matrices, dtype=np.float32).reshape(
        count, 3, 3
    )
    translations = np.asarray(grasp_group.translations, dtype=np.float32).reshape(
        count, 3
    )
    depths = np.asarray(grasp_group.depths, dtype=np.float32).reshape(count)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    tips = translations + depths[:, None] * rotations[:, :, 0]
    return {
        "grasps_camera_graspnet": poses,
        "scores": np.asarray(grasp_group.scores, dtype=np.float32).reshape(count),
        "widths": np.asarray(grasp_group.widths, dtype=np.float32).reshape(count),
        "heights": np.asarray(grasp_group.heights, dtype=np.float32).reshape(count),
        "depths": depths,
        "translations": translations,
        "rotation_matrices": rotations,
        "gripper_tips_camera": tips.astype(np.float32),
    }


def _visualize(points: np.ndarray, colors: np.ndarray, grasp_group: Any) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    display_transform = np.diag([1.0, 1.0, -1.0, 1.0])
    cloud.transform(display_transform)
    grippers = grasp_group.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(display_transform)
    o3d.visualization.draw_geometries([*grippers, cloud])


def _write_report(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    capture = args.capture.expanduser().resolve()
    segmentation = args.segmentation.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AnyGrasp checkpoint does not exist: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output already contains files: {output}; use a new --output name"
        )
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "anygrasp_check.json"

    points, colors, region_steering, valid_pixels = prepare_anygrasp_inputs(
        np.load(capture / "points_camera.npy", allow_pickle=False),
        np.load(capture / "rgb.npy", allow_pickle=False),
        np.load(segmentation / "union_mask.npy", allow_pickle=False),
    )
    region_count = int(region_steering.sum())
    if region_count < args.min_region_points:
        raise RuntimeError(
            f"SAM3 region has {region_count} valid 3D points, below "
            f"--min-region-points={args.min_region_points}"
        )
    np.save(output / "valid_pixel_mask.npy", valid_pixels)
    np.save(output / "region_steering.npy", region_steering)

    try:
        import gsnet

        detector = gsnet.create_detector(
            argparse.Namespace(
                checkpoint_path=str(checkpoint),
                max_gripper_width=args.max_gripper_width,
                gripper_height=args.gripper_height,
            )
        )
        if detector is None:
            raise RuntimeError(
                "AnyGrasp detector initialization failed; validate the machine-bound "
                "official SDK license and checkpoint"
            )
        options = {
            "dense_grasp": bool(args.dense_grasp),
            "collision_detection": True,
            "region_steering": region_steering,
            "approach_steering": None,
            "approach_thresh": float(np.pi),
        }
        started = time.monotonic()
        grasps = detector.get_grasp(points, options)
        elapsed_s = time.monotonic() - started
        if grasps is not None and len(grasps) > 0:
            if not args.dense_grasp:
                nms_result = grasps.nms()
                if nms_result is not None:
                    grasps = nms_result
            sorted_result = grasps.sort_by_score()
            if sorted_result is not None:
                grasps = sorted_result
            grasps = grasps[: min(args.topk, len(grasps))]

        count = 0 if grasps is None else len(grasps)
        arrays: dict[str, np.ndarray] = {}
        if count:
            arrays = _extract_grasp_arrays(grasps)
            for name, values in arrays.items():
                np.save(output / f"{name}.npy", values)

        report = {
            "status": "success" if count else "no_candidates",
            "reference": {
                "implementation": "official graspnet/anygrasp_sdk create_detector + get_grasp",
                "repository": "https://github.com/graspnet/anygrasp_sdk",
                "usage": "https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/USAGE.md",
                "sdk_commit": _git_commit(
                    args.anygrasp_root.expanduser().resolve()
                    if args.anygrasp_root is not None
                    else None
                ),
                "gsnet_module": str(Path(gsnet.__file__).resolve()),
            },
            "inputs": {
                "capture": str(capture),
                "segmentation": str(segmentation),
                "frame": "opencv_optical_x_right_y_down_z_forward",
                "scene_point_count": int(points.shape[0]),
                "region_steering_point_count": region_count,
            },
            "parameters": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "topk": args.topk,
                "max_gripper_width_m": args.max_gripper_width,
                "gripper_height_m": args.gripper_height,
                "dense_grasp": bool(args.dense_grasp),
                "collision_detection": True,
                "region_steering": True,
                "inference_elapsed_s": elapsed_s,
            },
            "candidates": {
                "count": count,
                "score_min": float(arrays["scores"].min()) if count else None,
                "score_max": float(arrays["scores"].max()) if count else None,
                "pose_origin": "AnyGrasp grasp center, not Panda hand origin",
                "approach_axis": "+X of each GraspNet rotation matrix",
                "closing_axis": "+Y of each GraspNet rotation matrix",
            },
            "safety": {
                "official_anygrasp_collision_detection_enabled": True,
                "panda_hand_frame_transform_validated": False,
                "curobo_planned": False,
                "trajectory_executed": False,
                "safe_to_execute": False,
                "manual_candidate_review_required": True,
            },
            "next_gate": (
                "Inspect official Open3D candidates, then validate the GraspNet-to-Panda "
                "hand transform before connecting these outputs to cuRobo."
            ),
        }
        _write_report(report_path, report)
        print(f"valid scene points: {points.shape[0]}", flush=True)
        print(f"SAM3 region points: {region_count}", flush=True)
        print(f"AnyGrasp candidates: {count}", flush=True)
        print(f"saved: {report_path}", flush=True)
        if args.vis and count:
            _visualize(points, colors, grasps[: min(20, count)])
        return 0 if count else 2
    except Exception as exc:
        if not report_path.exists():
            _write_report(
                report_path,
                {
                    "status": "failed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "inputs": {
                        "capture": str(capture),
                        "segmentation": str(segmentation),
                        "region_steering_point_count": region_count,
                    },
                    "safety": {
                        "trajectory_executed": False,
                        "safe_to_execute": False,
                    },
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
