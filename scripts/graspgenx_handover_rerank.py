#!/usr/bin/env python3
"""Prefer grasps that leave the segmented human receive region unencumbered.

Run this in the official GraspGenX environment.  It deliberately reuses the
official Franka collision mesh and point-cloud collision checker.  The check
is a conservative contact-space clearance proxy, not a model of the human or
ContactHandover's multi-view visibility/occlusion score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--receive-segmentation", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receive-clearance", type=float, default=0.015)
    parser.add_argument("--num-collision-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.receive_clearance) or args.receive_clearance <= 0.0:
        raise ValueError("--receive-clearance must be positive and finite")
    if args.num_collision_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch counts must be positive")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    try:
        import trimesh
        from graspgenx.utils.collision_filter import filter_colliding_grasps
        from graspgenx.x_grippers import resolve_gripper_info
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with the official GraspGenX uv environment."
        ) from exc

    from panda_handover.grasp_candidates import pose_quality, transform_grasp_poses
    from panda_handover.handover import receive_clear_score_order

    source_report = json.loads(
        (args.candidates / "collision_filter_check.json").read_text(encoding="utf-8")
    )
    if source_report.get("status") != "success":
        raise ValueError("source candidates did not pass static collision filtering")
    receive_report = json.loads(
        (args.receive_segmentation / "segmentation_check.json").read_text(
            encoding="utf-8"
        )
    )
    if receive_report.get("automatic_checks_passed") is not True:
        raise ValueError("receive-part segmentation did not pass its automatic checks")

    grasps = np.load(args.candidates / "grasps_camera.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    scores = np.load(args.candidates / "scores.npy", allow_pickle=False).astype(
        np.float32, copy=False
    ).reshape(-1)
    source_indices = np.load(
        args.candidates / "kept_candidate_indices.npy", allow_pickle=False
    ).reshape(-1)
    branch_tags = json.loads(
        (args.candidates / "branch_tags.json").read_text(encoding="utf-8")
    )
    receive_points = np.load(
        args.receive_segmentation / "points_camera.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    T_world_camera = np.load(
        args.capture / "T_world_camera.npy", allow_pickle=False
    )
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) == 0:
        raise ValueError("filtered grasps must have non-empty shape (N,4,4)")
    if not (
        len(scores) == len(grasps) == len(source_indices) == len(branch_tags)
    ):
        raise ValueError("candidate arrays and provenance have mismatched lengths")
    if receive_points.ndim != 2 or receive_points.shape[1] != 3:
        raise ValueError("receive-part points_camera.npy must have shape (N,3)")
    if len(receive_points) < 20 or not np.isfinite(receive_points).all():
        raise ValueError("receive-part point cloud is too small or non-finite")

    np.random.seed(args.random_seed)
    gripper = resolve_gripper_info("franka_panda")
    surface_points, _ = trimesh.sample.sample_surface(
        gripper.collision_mesh, args.num_collision_samples
    )
    started = time.monotonic()
    receive_clear_mask = np.asarray(
        filter_colliding_grasps(
            scene_pc=receive_points,
            grasp_poses=grasps,
            collision_threshold=args.receive_clearance,
            gripper_surface_points=np.asarray(surface_points, dtype=np.float32),
            batch_size=args.batch_size,
            device=None if args.device == "auto" else args.device,
        ),
        dtype=bool,
    ).reshape(-1)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if len(receive_clear_mask) != len(grasps):
        raise RuntimeError("official collision checker returned a mismatched mask")

    # Lexicographic policy: first enforce receive-region clearance, then retain
    # the learned GraspGenX ranking.  No hand-tuned weighted score is invented.
    order = receive_clear_score_order(scores, receive_clear_mask)
    ordered_grasps = grasps[order]
    ordered_scores = scores[order]
    ordered_source_indices = source_indices[order].astype(np.int32, copy=False)
    ordered_tags = [branch_tags[int(index)] for index in order]
    ordered_world, ordered_hands = transform_grasp_poses(
        ordered_grasps, T_world_camera
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "receive_clear_mask.npy", receive_clear_mask)
    np.save(output / "handover_order_in_static_filtered.npy", order.astype(np.int32))
    np.save(output / "kept_candidate_indices.npy", ordered_source_indices)
    np.save(output / "grasps_camera.npy", ordered_grasps)
    np.save(output / "scores.npy", ordered_scores)
    np.save(output / "grasps_world.npy", ordered_world)
    np.save(output / "panda_hand_world.npy", ordered_hands)
    np.save(output / "receive_part_points_camera.npy", receive_points)
    (output / "branch_tags.json").write_text(
        json.dumps(ordered_tags, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "success" if len(order) else "no_receive_clear_candidates",
        "reference": {
            "candidate_reranking_precedent": "ContactHandover (IROS 2024)",
            "candidate_reranking_url": "https://arxiv.org/abs/2404.01402",
            "task_oriented_grasp_precedent": (
                "The Grasp Strategy of a Robot Passer Influences Performance "
                "and Quality of Robot-to-Human Object Handover"
            ),
            "checker": "NVIDIA GraspGenX official filter_colliding_grasps",
            "gripper_geometry": "NVIDIA GraspGenX official franka_panda collision mesh",
        },
        "inputs": {
            "capture": str(args.capture),
            "receive_segmentation": str(args.receive_segmentation),
            "static_filtered_candidates": str(args.candidates),
            "receive_part_point_count": int(len(receive_points)),
        },
        "policy": {
            "ordering": (
                "hard gate on receive-part clearance, then original GraspGenX "
                "score descending"
            ),
            "weighted_score_added": False,
            "receive_clearance_m": float(args.receive_clearance),
            "clearance_interpretation": (
                "contact-space encumbrance proxy: sampled gripper collision surface "
                "must remain farther than the threshold from observed receive-part points"
            ),
            "true_human_contact_occlusion_or_visibility_model": False,
            "human_body_model_present": False,
            "random_seed": args.random_seed,
            "num_collision_samples": args.num_collision_samples,
            "batch_size": args.batch_size,
            "device": args.device,
            "elapsed_ms": elapsed_ms,
        },
        "candidates": {
            "before": int(len(grasps)),
            "receive_clear": int(len(order)),
            "rejected_as_receive_region_encumbered": int(len(grasps) - len(order)),
            "ordered_source_candidate_indices": ordered_source_indices.tolist(),
            "best_source_candidate_index": (
                int(ordered_source_indices[0]) if len(order) else None
            ),
            "best_graspgenx_score": (
                float(ordered_scores[0]) if len(order) else None
            ),
            "camera_pose_quality": pose_quality(ordered_grasps),
            "world_pose_quality": pose_quality(ordered_world),
            "panda_hand_pose_quality": pose_quality(ordered_hands),
        },
        "next_gate": (
            "For each surviving grasp, generate affordance-aligned handover roll "
            "variants and require cuRobo reachability with the whole object attached."
        ),
        "safety": {
            "simulation_only": True,
            "transport_trajectory_checked": False,
            "human_collision_checked": False,
            "safe_for_real_robot_execution": False,
            "manual_review_required": True,
        },
    }
    report_path = output / "handover_rerank_check.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"handover receive-region clearance: {len(order)}/{len(grasps)} candidates",
        flush=True,
    )
    print(f"saved: {report_path}", flush=True)
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
