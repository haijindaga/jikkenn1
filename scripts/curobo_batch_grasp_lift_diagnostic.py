#!/usr/bin/env python3
"""Batch-diagnose score-ordered grasp/lift feasibility with pinned cuRobo.

This is an opt-in simulation-only diagnostic.  It deliberately does not emit
replayable trajectories and does not replace the reviewed sequential planner.
It answers the first equivalence question: which candidates pass cuRobo's
official BatchMotionPlanner grasp/lift stages when the robot, world, and CUDA
buffers are initialized once and retained across candidate chunks?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--pregrasp-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--grasp-depth-offset-m", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help=(
            "GPU resource cap only; candidates remain in the same score order "
            "and no candidate-dependent threshold is changed"
        ),
    )
    parser.add_argument(
        "--maximum-candidates",
        type=int,
        help="Optional score-ordered diagnostic prefix; omitted evaluates all",
    )
    return parser.parse_args()


def _cpu_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_reviewed_inputs(
    pregrasp_plan: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    report_path = pregrasp_plan / "pregrasp_plan_check.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report.get("automatic_checks", {})
    safety = report.get("safety", {})
    parameters = report.get("parameters", {})
    if report.get("status") != "success" or not checks or not all(checks.values()):
        raise ValueError("pre-grasp plan did not pass its automatic gate")
    if parameters.get("scene_backend") != "observed_pointcloud_mesh":
        raise ValueError("batch diagnostic requires the reviewed observed mesh")
    if safety.get("simulation_only") is not True:
        raise ValueError("batch diagnostic is restricted to simulation-only input")
    transforms = np.load(
        pregrasp_plan / "grasp_transforms_robot_base.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    source_indices = np.load(
        pregrasp_plan / "source_candidate_indices.npy", allow_pickle=False
    ).reshape(-1)
    scores = np.load(
        pregrasp_plan / "candidate_scores.npy", allow_pickle=False
    ).reshape(-1)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("grasp transforms must have shape (N,4,4)")
    if len(transforms) == 0 or not (
        len(transforms) == len(source_indices) == len(scores)
    ):
        raise ValueError("pre-grasp candidate arrays are empty or mismatched")
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("source candidate indices must be unique")
    return report, transforms, source_indices, scores


def _mask(result: Any, field: str, active_count: int) -> list[bool]:
    from panda_handover.batch_planning import reduce_batch_success

    value = getattr(result, field, None) if result is not None else None
    if value is None:
        return [False] * active_count
    return reduce_batch_success(_cpu_numpy(value), active_count)


def main() -> int:
    process_started = time.monotonic()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.maximum_candidates is not None and args.maximum_candidates <= 0:
        raise ValueError("--maximum-candidates must be positive")
    if not np.isfinite(args.lift_offset) or args.lift_offset <= 0.0:
        raise ValueError("--lift-offset must be positive and finite")
    if not np.isfinite(args.grasp_depth_offset_m) or args.grasp_depth_offset_m < 0.0:
        raise ValueError("--grasp-depth-offset-m must be finite and non-negative")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output already contains files: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "apply_curobo_voxel_round_fix.py"),
            "--check-only",
        ],
        check=True,
    )

    from panda_handover.batch_planning import (
        candidate_chunks,
        pad_candidate_indices,
    )
    from panda_handover.curobo_bridge import select_named_joint_positions
    from panda_handover.curobo_planning import (
        load_singleview_observed_pointcloud,
        offset_grasp_depth_tool_z,
        rotation_matrix_to_quaternion_wxyz,
    )

    pregrasp_report, transforms, source_indices, scores = _load_reviewed_inputs(
        args.pregrasp_plan
    )
    if args.maximum_candidates is not None:
        stop = min(args.maximum_candidates, len(transforms))
        transforms = transforms[:stop]
        source_indices = source_indices[:stop]
        scores = scores[:stop]
    original_transforms = transforms.copy()
    transforms = offset_grasp_depth_tool_z(
        transforms, args.grasp_depth_offset_m
    ).astype(np.float32, copy=False)

    prepared_map_value = pregrasp_report.get("inputs", {}).get("prepared_map")
    if not isinstance(prepared_map_value, str) or not prepared_map_value:
        raise ValueError("source pre-grasp report has no prepared_map provenance")
    observed_scene = load_singleview_observed_pointcloud(
        Path(prepared_map_value), args.capture
    )
    robot_report = json.loads(
        (args.capture / "robot_state.json").read_text(encoding="utf-8")
    )
    captured_names = tuple(str(name) for name in robot_report["joint_names"])
    captured_positions = np.load(
        args.capture / "panda_joint_positions.npy", allow_pickle=False
    ).astype(np.float32, copy=False)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo batch diagnostic requires CUDA")
    from curobo._src.geom.types import Mesh, SceneCfg
    from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    scene_mesh = Mesh.from_pointcloud(
        observed_scene.points_robot_base_m,
        pitch=observed_scene.voxel_size_m,
        name="observed_scene_without_robot_or_target",
    )
    configured_batch_size = min(args.batch_size, len(transforms))
    planner_initialization_started = time.monotonic()
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=SceneCfg(mesh=[scene_mesh]),
        device_cfg=device_cfg,
        max_batch_size=configured_batch_size,
        multi_env=False,
        max_goalset=1,
    )
    planner = BatchMotionPlanner(planner_cfg)
    planner_initialization_elapsed = time.monotonic() - planner_initialization_started
    planning_started = time.monotonic()
    candidates: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    try:
        if planner.tool_frames != ["panda_hand"]:
            raise RuntimeError(
                f"reviewed Franka tool frame changed: {planner.tool_frames}"
            )
        contact_links = list(
            planner.kinematics.config.kinematics_config.grasp_contact_link_names
            or ()
        )
        if not contact_links:
            raise RuntimeError("reviewed Franka config has no grasp contact links")
        planner.warmup(enable_graph=False, num_warmup_iterations=1)
        start_positions = select_named_joint_positions(
            captured_names, captured_positions, planner.joint_names
        ).astype(np.float32, copy=False)

        for batch_rank, index_range in enumerate(
            candidate_chunks(len(transforms), configured_batch_size)
        ):
            indices = np.asarray(list(index_range), dtype=np.int64)
            active_count = len(indices)
            # The pinned cuRobo goal registry allocates a fixed leading batch
            # dimension during warmup.  Despite max_batch_size's API wording,
            # changing that dimension on the final partial chunk fails in
            # JointState.copy_.  Duplicate the last real candidate solely to
            # preserve tensor shape, then discard all padded result rows.
            padded_indices = np.asarray(
                pad_candidate_indices(indices.tolist(), configured_batch_size),
                dtype=np.int64,
            )
            batch_transforms = transforms[padded_indices]
            quaternions = rotation_matrix_to_quaternion_wxyz(
                batch_transforms[:, :3, :3]
            ).astype(np.float32, copy=False)
            goals = GoalToolPose(
                tool_frames=planner.tool_frames,
                position=torch.from_numpy(batch_transforms[:, :3, 3]).to(
                    device_cfg.device
                )[:, None, None, None, :],
                quaternion=torch.from_numpy(quaternions).to(device_cfg.device)[
                    :, None, None, None, :
                ],
            )
            current_state = JointState.from_position(
                torch.from_numpy(start_positions).to(device_cfg.device)[None, :].repeat(
                    configured_batch_size, 1
                ),
                joint_names=planner.joint_names,
            )

            planner.reset_seed()
            preflight_started = time.monotonic()
            preflight = planner.plan_pose(
                goal_tool_poses=goals,
                current_state=current_state,
                max_attempts=args.max_attempts,
                success_ratio=1.0,
            )
            preflight_elapsed = time.monotonic() - preflight_started
            preflight_success = _mask(preflight, "success", active_count)

            planner.reset_seed()
            grasp_started = time.monotonic()
            result = planner.plan_grasp(
                grasp_poses=goals,
                current_state=current_state,
                grasp_approach_axis="z",
                grasp_approach_offset=-float(
                    pregrasp_report["parameters"]["approach_offset_m"]
                ),
                grasp_approach_in_tool_frame=True,
                grasp_lift_axis="z",
                grasp_lift_offset=args.lift_offset,
                grasp_lift_in_tool_frame=False,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
            )
            grasp_elapsed = time.monotonic() - grasp_started
            goalset_success = _mask(
                getattr(result, "goalset_result", None), "success", active_count
            )
            approach_success = _mask(result, "approach_success", active_count)
            grasp_success = _mask(result, "grasp_success", active_count)
            lift_success = _mask(result, "lift_success", active_count)
            overall_success = _mask(result, "success", active_count)
            batch_reports.append(
                {
                    "batch_rank": batch_rank,
                    "original_goalset_ranks": indices.astype(int).tolist(),
                    "active_candidate_count": active_count,
                    "configured_batch_size": configured_batch_size,
                    "padding_count": configured_batch_size - active_count,
                    "padding_policy": (
                        "duplicate final real candidate; discard padded results"
                    ),
                    "preflight_wall_time_s": preflight_elapsed,
                    "grasp_lift_wall_time_s": grasp_elapsed,
                    "planner_status": str(getattr(result, "status", "no result")),
                }
            )
            for local_rank, original_rank in enumerate(indices):
                candidates.append(
                    {
                        "original_goalset_rank": int(original_rank),
                        "source_candidate_index": int(source_indices[original_rank]),
                        "graspgenx_score": float(scores[original_rank]),
                        "strict_issue_663_preflight_success": preflight_success[
                            local_rank
                        ],
                        "goalset_success": goalset_success[local_rank],
                        "approach_success": approach_success[local_rank],
                        "grasp_success": grasp_success[local_rank],
                        "lift_success": lift_success[local_rank],
                        "batch_grasp_lift_success": overall_success[local_rank],
                        "batch_prefilter_eligible": bool(
                            preflight_success[local_rank]
                            and overall_success[local_rank]
                        ),
                    }
                )
    finally:
        planner.destroy()

    candidates.sort(key=lambda item: item["original_goalset_rank"])
    report = {
        "status": "success",
        "purpose": "opt-in cuRobo batch grasp/lift feasibility diagnostic",
        "reference": {
            "curobo_commit": CUROBO_COMMIT,
            "planner": "curobo._src.motion.motion_planner_batch.BatchMotionPlanner",
            "official_source": (
                "curobo/_src/motion/motion_planner_batch.py"
            ),
        },
        "inputs": {
            "capture": str(args.capture),
            "pregrasp_plan": str(args.pregrasp_plan),
            "prepared_map": prepared_map_value,
            "robot": args.robot,
        },
        "parameters": {
            "candidate_count": len(transforms),
            "requested_batch_size": args.batch_size,
            "configured_batch_size": configured_batch_size,
            "max_attempts_for_issue_663_preflight": args.max_attempts,
            "batch_plan_grasp_attempts": 1,
            "approach_offset_m": float(
                pregrasp_report["parameters"]["approach_offset_m"]
            ),
            "lift_offset_m": args.lift_offset,
            "grasp_depth_offset_m": args.grasp_depth_offset_m,
            "candidate_specific_parameter_tuning": False,
            "shared_collision_world": True,
            "planner_reused_across_chunks": True,
            "partial_final_batch_policy": (
                "duplicate final real candidate to fixed cuRobo batch size; "
                "discard padded result rows"
            ),
        },
        "timing": {
            "process_wall_time_s": time.monotonic() - process_started,
            "planner_initialization_wall_time_s": planner_initialization_elapsed,
            "warmup_and_planning_wall_time_s": time.monotonic() - planning_started,
            "batches": batch_reports,
        },
        "candidates": candidates,
        "summary": {
            "strict_preflight_success_count": sum(
                item["strict_issue_663_preflight_success"] for item in candidates
            ),
            "batch_grasp_lift_success_count": sum(
                item["batch_grasp_lift_success"] for item in candidates
            ),
            "batch_prefilter_eligible_count": sum(
                item["batch_prefilter_eligible"] for item in candidates
            ),
        },
        "safety": {
            "diagnosis_only": True,
            "replayable_trajectory_saved": False,
            "trajectory_executed": False,
            "sequential_pipeline_replaced": False,
            "safe_for_real_robot_execution": False,
        },
        "limitations": [
            "BatchMotionPlanner performs one grasp-planning pass per chunk, while the reviewed sequential planner has different retry behavior.",
            "Reviewed support-contact fallback is not applied by this diagnostic.",
            "Attached-object handover transport and post-validation are not yet batched.",
        ],
        "next_gate": (
            "Compare this report with a sequential trial manifest before using "
            "batch results to filter or select any candidate."
        ),
    }
    np.save(
        args.output / "original_grasp_transforms_robot_base.npy",
        original_transforms,
        allow_pickle=False,
    )
    np.save(
        args.output / "depth_adjusted_grasp_transforms_robot_base.npy",
        transforms,
        allow_pickle=False,
    )
    report_path = args.output / "batch_grasp_lift_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
