#!/usr/bin/env python3
"""Send an Isaac/SAM3 capture to NVIDIA's official GraspGenX server.

Run this script with GraspGenX's own ``uv`` environment, not env_isaaclab.
The two projects intentionally remain separate because their Torch pins differ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/graspgenx_smoke"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--planner", default="graspmoe", choices=("graspmoe", "diffusion"))
    parser.add_argument("--min-object-points", type=int, default=100)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--topk", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    try:
        from graspgenx.serving.types import SweepVolumeParams
        from graspgenx.serving.zmq_client import GraspGenXClient
    except ImportError as exc:
        raise RuntimeError(
            "GraspGenX is not importable. Run this script from the official "
            "GraspGenX checkout with `uv run python /path/to/this/script ...`."
        ) from exc

    from panda_handover.grasp_candidates import (
        prepare_scene_point_cloud,
        save_grasp_candidates,
    )

    points_camera = np.load(args.capture / "points_camera.npy")
    T_world_camera = np.load(args.capture / "T_world_camera.npy")
    union_mask = np.load(args.segmentation / "union_mask.npy")
    point_cloud, instance_mask, instance_count = prepare_scene_point_cloud(
        points_camera, union_mask
    )
    if instance_count < args.min_object_points:
        raise RuntimeError(
            f"SAM3 has {instance_count} valid 3D points, below "
            f"--min-object-points={args.min_object_points}"
        )

    sweep_params = SweepVolumeParams.from_gripper_config("franka_panda")
    started = time.monotonic()
    with GraspGenXClient(
        host=args.host, port=args.port, timeout_ms=args.timeout_ms
    ) as client:
        health = client.health()
        metadata = client.server_metadata
        if health.get("status") != "ok":
            raise RuntimeError(f"GraspGenX health check failed: {health}")
        if "infer_scene_pc" not in metadata.get("actions", []):
            raise RuntimeError(
                "The connected GraspGenX server does not advertise infer_scene_pc; "
                "update the official GraspGenX checkout."
            )
        results = client.infer_scene_pc(
            point_cloud=point_cloud,
            instance_mask=instance_mask,
            sweep_volume_params=sweep_params,
            planner=args.planner,
            min_object_points=args.min_object_points,
            num_grasps=args.num_grasps,
            grasp_threshold=args.grasp_threshold,
            topk_num_grasps=args.topk,
            return_branch_tags=True,
        )
    elapsed_ms = (time.monotonic() - started) * 1000.0

    empty_grasps = np.empty((0, 4, 4), dtype=np.float32)
    empty_scores = np.empty((0,), dtype=np.float32)
    grasps, scores, tags = results.get(1, (empty_grasps, empty_scores, []))
    parameters = {
        "planner": args.planner,
        "gripper": "franka_panda",
        "min_object_points": args.min_object_points,
        "num_grasps": args.num_grasps,
        "grasp_threshold": args.grasp_threshold,
        "topk_num_grasps": args.topk,
        "round_trip_ms": elapsed_ms,
        "sweep_volume_params": sweep_params.to_dict(),
    }
    report = save_grasp_candidates(
        args.output,
        grasps_camera=grasps,
        scores=scores,
        branch_tags=tags,
        T_world_camera=T_world_camera,
        input_point_count=instance_count,
        parameters=parameters,
        server_health=health,
        server_metadata=metadata,
    )
    print(f"valid instance points: {instance_count}", flush=True)
    print(f"GraspGenX candidates: {report['candidates']['count']}", flush=True)
    print(f"saved: {args.output / 'graspgenx_check.json'}", flush=True)
    if report["status"] != "success":
        print("No candidates were returned; no motion command was produced.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
