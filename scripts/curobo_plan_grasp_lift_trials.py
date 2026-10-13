#!/usr/bin/env python3
"""Create separate cuRobo grasp/lift plans for several score-ordered candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.vlm_parts import load_handover_parts_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--pregrasp-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-prefilter-report",
        type=Path,
        help=(
            "Optional diagnosis-only GPU batch report. Only candidates that "
            "passed both its strict pose preflight and grasp/lift gate receive "
            "the unchanged complete sequential handover planner."
        ),
    )
    parser.add_argument("--max-physical-trials", type=int, default=5)
    parser.add_argument(
        "--maximum-candidates",
        type=int,
        help=(
            "Optional fixed prefix of the pregrasp score order to evaluate. "
            "Omit to preserve the reviewed all-candidate automatic-handover search."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument(
        "--grasp-depth-offset-m",
        type=float,
        default=0.0,
        help="Apply one common tool +Z depth offset to every candidate",
    )
    parser.add_argument(
        "--handover-goal-position-robot-base-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--handover-goal-quaternion-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument(
        "--handover-receiver-position-robot-base-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--handover-region-json", type=Path)
    parser.add_argument(
        "--handover-region-samples-per-axis", type=int, default=3
    )
    parser.add_argument(
        "--handover-human-direction-robot-base",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument(
        "--handover-height-policy",
        choices=("receiver-position", "preserve-lift-end"),
        default="receiver-position",
    )
    parser.add_argument("--transport-orientation-policy-report", type=Path)
    parser.add_argument(
        "--diagnostic-gravity-tilt-tolerance-deg",
        type=float,
        help="Pass one fixed simulation-only gravity-tilt tolerance to every trial",
    )
    parser.add_argument(
        "--semantic-orientation-cost",
        action="store_true",
        help=(
            "Apply one common paper-inspired cuRobo orientation soft-cost policy "
            "to every candidate. For preserve_gravity_tilt, exhaust the eligible "
            "pool and select lexicographically by receive-direction error, maximum "
            "transport deviation, then original score rank."
        ),
    )
    parser.add_argument(
        "--exclude-source-candidate-index",
        type=int,
        action="append",
        default=[],
        help=(
            "Skip a source candidate already physically evaluated; may be "
            "repeated without changing the remaining score order"
        ),
    )
    parser.add_argument(
        "--allow-reviewed-support-contact-preflight", action="store_true"
    )
    args = parser.parse_args()
    if args.max_physical_trials <= 0:
        parser.error("--max-physical-trials must be positive")
    if args.maximum_candidates is not None and args.maximum_candidates <= 0:
        parser.error("--maximum-candidates must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if (
        args.handover_region_samples_per_axis <= 0
        or args.handover_region_samples_per_axis % 2 == 0
    ):
        parser.error("--handover-region-samples-per-axis must be positive and odd")
    if not np.isfinite(args.grasp_depth_offset_m) or args.grasp_depth_offset_m < 0.0:
        parser.error("--grasp-depth-offset-m must be finite and non-negative")
    if args.diagnostic_gravity_tilt_tolerance_deg is not None and (
        not np.isfinite(args.diagnostic_gravity_tilt_tolerance_deg)
        or not 0.0 < args.diagnostic_gravity_tilt_tolerance_deg <= 90.0
    ):
        parser.error(
            "--diagnostic-gravity-tilt-tolerance-deg must be in (0, 90]"
        )
    if any(value < 0 for value in args.exclude_source_candidate_index):
        parser.error("--exclude-source-candidate-index must be non-negative")
    if len(set(args.exclude_source_candidate_index)) != len(
        args.exclude_source_candidate_index
    ):
        parser.error("excluded source candidate indices must be unique")
    if (
        args.handover_goal_quaternion_wxyz is not None
        and args.handover_goal_position_robot_base_m is None
    ):
        parser.error("handover orientation requires a handover position")
    point_handover = args.handover_receiver_position_robot_base_m is not None
    region_handover = args.handover_region_json is not None
    if point_handover != (args.handover_human_direction_robot_base is not None):
        parser.error(
            "automatic handover requires both receiver position and human direction"
        )
    if region_handover and (
        point_handover or args.handover_human_direction_robot_base is not None
    ):
        parser.error(
            "--handover-region-json cannot be combined with point handover arguments"
        )
    if (
        (point_handover or region_handover)
        and args.handover_goal_position_robot_base_m is not None
    ):
        parser.error(
            "automatic affordance-aware handover cannot be combined with a fixed "
            "panda_hand goal"
        )
    if (
        args.handover_height_policy != "receiver-position"
        and not point_handover
    ):
        parser.error("--handover-height-policy requires automatic point handover")
    if (
        args.semantic_orientation_cost
        and not (point_handover or region_handover)
    ):
        parser.error("--semantic-orientation-cost requires automatic handover")
    return args


def read_plan_status(directory: Path) -> tuple[str, str | None]:
    report_names = (
        "grasp_lift_plan_check.json",
        "grasp_preflight_failure.json",
        "grasp_lift_failure.json",
        "handover_transport_failure.json",
        "handover_orientation_constraint_failure.json",
    )
    for name in report_names:
        path = directory / name
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            return str(report.get("status", "unknown")), str(path)
    return "missing_report", None


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def optional_mapping(value: object) -> dict:
    """Treat absent optional report sections as empty mappings."""

    return value if isinstance(value, dict) else {}


def _finite_metric(value: object, *, name: str, source_index: int) -> float:
    if value is None or not np.isfinite(value):
        raise RuntimeError(
            f"candidate {source_index} succeeded without a finite {name}"
        )
    return float(value)


def load_transport_orientation_policy(path: Path) -> str:
    """Load the validated policy using the pipeline's canonical VLM schema."""

    return load_handover_parts_report(path).transport_orientation_policy


def load_batch_prefilter_sources(
    path: Path,
    *,
    pregrasp_plan: Path,
    grasp_depth_offset_m: float,
) -> list[int]:
    """Validate batch provenance and return its score-ordered eligible sources."""

    from panda_handover.batch_planning import batch_prefilter_eligible_sources

    report = json.loads(path.read_text(encoding="utf-8"))
    inputs = optional_mapping(report.get("inputs"))
    reported_pregrasp = inputs.get("pregrasp_plan")
    if not isinstance(reported_pregrasp, str) or (
        Path(reported_pregrasp).expanduser().resolve()
        != pregrasp_plan.expanduser().resolve()
    ):
        raise ValueError("batch prefilter pregrasp provenance does not match")
    parameters = optional_mapping(report.get("parameters"))
    reported_depth = parameters.get("grasp_depth_offset_m")
    if reported_depth is None or not np.isclose(
        float(reported_depth), grasp_depth_offset_m, atol=1e-9, rtol=0.0
    ):
        raise ValueError("batch prefilter grasp-depth policy does not match")
    return batch_prefilter_eligible_sources(report)


def select_physical_trial_attempts(
    attempts: list[dict],
    *,
    limit: int,
    automatic_handover: bool,
    semantic_orientation_cost: bool,
) -> list[dict]:
    """Select planner-feasible trials with a fixed, candidate-independent rule."""

    feasible = [attempt for attempt in attempts if attempt["planner_feasible"]]
    if automatic_handover:
        return sorted(
            feasible,
            key=lambda attempt: (
                attempt["handover_horizontal_priority_rank"],
                attempt["receive_direction_alignment_error_rad"],
                (
                    attempt["maximum_transport_orientation_deviation_rad"]
                    if attempt["maximum_transport_orientation_deviation_rad"]
                    is not None
                    else 0.0
                ),
                attempt["original_goalset_rank"],
            ),
        )[:limit]
    if semantic_orientation_cost:
        return sorted(
            feasible,
            key=lambda attempt: (
                attempt["semantic_orientation_cost"],
                attempt["original_goalset_rank"],
            ),
        )[:limit]
    return sorted(
        feasible, key=lambda attempt: attempt["original_goalset_rank"]
    )[:limit]


def main() -> int:
    args = parse_args()
    source_indices = np.load(
        args.pregrasp_plan / "source_candidate_indices.npy", allow_pickle=False
    ).reshape(-1)
    scores = np.load(
        args.pregrasp_plan / "candidate_scores.npy", allow_pickle=False
    ).reshape(-1)
    if len(source_indices) == 0 or len(source_indices) != len(scores):
        raise ValueError("pre-grasp candidate indices and scores are empty or mismatched")
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("pre-grasp source candidate indices must be unique")
    all_source_indices = source_indices
    total_candidate_pool_count = int(len(source_indices))
    original_goalset_ranks = np.arange(total_candidate_pool_count, dtype=np.int64)
    if args.maximum_candidates is not None:
        search_stop = min(args.maximum_candidates, total_candidate_pool_count)
        source_indices = source_indices[:search_stop]
        scores = scores[:search_stop]
        original_goalset_ranks = original_goalset_ranks[:search_stop]
    evaluated_prefix_count = int(len(source_indices))
    candidate_pool_truncated = evaluated_prefix_count < total_candidate_pool_count
    batch_prefilter_eligible_count = None
    batch_prefilter_rejected_count = None
    if args.batch_prefilter_report is not None:
        eligible_sources = load_batch_prefilter_sources(
            args.batch_prefilter_report,
            pregrasp_plan=args.pregrasp_plan,
            grasp_depth_offset_m=args.grasp_depth_offset_m,
        )
        report_sources = set(eligible_sources)
        batch_report = json.loads(
            args.batch_prefilter_report.read_text(encoding="utf-8")
        )
        covered_sources = [
            int(item["source_candidate_index"])
            for item in sorted(
                batch_report["candidates"],
                key=lambda item: int(item["original_goalset_rank"]),
            )
        ]
        expected_sources = [int(value) for value in source_indices]
        if covered_sources != expected_sources:
            raise ValueError(
                "batch prefilter candidate order/coverage does not match the "
                "requested score-ordered planning scope"
            )
        selected = np.asarray(
            [int(value) in report_sources for value in source_indices], dtype=bool
        )
        batch_prefilter_eligible_count = int(np.count_nonzero(selected))
        batch_prefilter_rejected_count = int(len(selected) - np.count_nonzero(selected))
        source_indices = source_indices[selected]
        scores = scores[selected]
        original_goalset_ranks = original_goalset_ranks[selected]

    transport_orientation_policy = None
    if args.transport_orientation_policy_report is not None:
        transport_orientation_policy = load_transport_orientation_policy(
            args.transport_orientation_policy_report
        )
    automatic_handover = bool(
        args.handover_receiver_position_robot_base_m is not None
        or args.handover_region_json is not None
    )
    constrained_orientation_selection = bool(
        automatic_handover
        and transport_orientation_policy
        in {"preserve_gravity_tilt", "keep_grasp_orientation"}
    )

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"trial output already contains files: {output}; use a new output name"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "grasp_lift_trial_plans.json"
    manifest = {
        "status": "planning_in_progress",
        "inputs": {
            "capture": str(args.capture),
            "segmentation": str(args.segmentation),
            "pregrasp_plan": str(args.pregrasp_plan),
            "batch_prefilter_report": (
                str(args.batch_prefilter_report)
                if args.batch_prefilter_report is not None
                else None
            ),
        },
        "policy": {
            "candidate_order": "pregrasp score order",
            "maximum_physical_trials": args.max_physical_trials,
            "candidate_specific_parameter_tuning": False,
            "grasp_depth_offset_m_for_every_candidate": args.grasp_depth_offset_m,
            "grasp_depth_axis": "GraspGenX canonical tool +Z",
            "excluded_previously_evaluated_source_candidate_indices": (
                args.exclude_source_candidate_index
            ),
            "continue_after_planning_rejection": True,
            "handover_transport_requested": bool(
                args.handover_goal_position_robot_base_m is not None
                or automatic_handover
            ),
            "handover_goal_position_robot_base_m": (
                args.handover_goal_position_robot_base_m
            ),
            "handover_goal_quaternion_wxyz": args.handover_goal_quaternion_wxyz,
            "handover_receiver_position_robot_base_m": (
                args.handover_receiver_position_robot_base_m
            ),
            "handover_region_json": (
                str(args.handover_region_json)
                if args.handover_region_json is not None
                else None
            ),
            "handover_region_samples_per_axis": (
                args.handover_region_samples_per_axis
                if automatic_handover
                else None
            ),
            "handover_human_direction_robot_base": (
                args.handover_human_direction_robot_base
            ),
            "handover_height_policy": args.handover_height_policy,
            "handover_orientation_policy": (
                "from transport-orientation-policy report"
                if automatic_handover
                and args.transport_orientation_policy_report is not None
                else "affordance-axis alignment with fixed roll variants"
                if automatic_handover
                else "manual panda_hand goal"
                if args.handover_goal_position_robot_base_m is not None
                else None
            ),
            "transport_orientation_policy_report": (
                str(args.transport_orientation_policy_report)
                if args.transport_orientation_policy_report is not None
                else None
            ),
            "transport_orientation_policy": transport_orientation_policy,
            "diagnostic_gravity_tilt_tolerance_deg_for_every_candidate": (
                args.diagnostic_gravity_tilt_tolerance_deg
            ),
            "semantic_orientation_cost_for_every_candidate": (
                args.semantic_orientation_cost
            ),
            "successful_plan_order": (
                "lexicographic handover-region horizontal priority, receive-"
                "direction error, constrained transport-orientation deviation "
                "when applicable, then original score rank"
                if automatic_handover
                else "ascending semantic orientation cost, then original score rank"
                if args.semantic_orientation_cost
                else "original score rank"
            ),
            "joint_handover_candidate_selection": automatic_handover,
            "direction_aware_candidate_selection": automatic_handover,
            "constrained_orientation_candidate_selection": (
                constrained_orientation_selection
            ),
            "candidate_search_scope": (
                f"fixed score-ordered prefix of {evaluated_prefix_count} candidates"
                if args.maximum_candidates is not None
                else "all eligible pregrasp candidates"
                if automatic_handover
                else "until the requested physical-trial count is available"
            ),
            "maximum_candidates": args.maximum_candidates,
            "candidate_prefix_changes_constraints_or_thresholds": False,
            "selection_optimality_scope": (
                "evaluated score-ordered prefix only"
                if candidate_pool_truncated
                else "complete pregrasp candidate pool"
            ),
            "selection_uses_weighted_sum": False,
            "selection_uses_candidate_specific_thresholds": False,
            "batch_prefilter_enabled": args.batch_prefilter_report is not None,
            "batch_prefilter_policy": (
                "strict Issue #663 pose preflight AND official batch grasp/lift "
                "success; complete handover constraints remain sequential"
                if args.batch_prefilter_report is not None
                else None
            ),
        },
        "candidate_pool_count": total_candidate_pool_count,
        "evaluated_candidate_prefix_count": evaluated_prefix_count,
        "candidate_pool_truncated": candidate_pool_truncated,
        "batch_prefilter_eligible_count": batch_prefilter_eligible_count,
        "batch_prefilter_rejected_count": batch_prefilter_rejected_count,
        "attempts": [],
        "successful_plan_directories": [],
    }
    write_manifest(manifest_path, manifest)

    planner_script = Path(__file__).with_name("curobo_plan_grasp_lift.py")
    planner_feasible_plans = 0
    for original_rank, source_index, score in zip(
        original_goalset_ranks, source_indices, scores, strict=True
    ):
        if not automatic_handover and planner_feasible_plans >= args.max_physical_trials:
            break
        original_rank = int(original_rank)
        source_index = int(source_index)
        if source_index in args.exclude_source_candidate_index:
            continue
        trial_directory = output / f"candidate_{source_index:03d}"
        command = [
            sys.executable,
            str(planner_script),
            "--capture",
            str(args.capture),
            "--segmentation",
            str(args.segmentation),
            "--pregrasp-plan",
            str(args.pregrasp_plan),
            "--output",
            str(trial_directory),
            "--source-candidate-index",
            str(source_index),
            "--robot",
            args.robot,
            "--device",
            args.device,
            "--lift-offset",
            str(args.lift_offset),
            "--grasp-depth-offset-m",
            str(args.grasp_depth_offset_m),
            "--max-attempts",
            str(args.max_attempts),
        ]
        if args.allow_reviewed_support_contact_preflight:
            command.append("--allow-reviewed-support-contact-preflight")
        if args.handover_goal_position_robot_base_m is not None:
            command.extend(
                [
                    "--handover-goal-position-robot-base-m",
                    *[
                        str(value)
                        for value in args.handover_goal_position_robot_base_m
                    ],
                ]
            )
        if args.handover_goal_quaternion_wxyz is not None:
            command.extend(
                [
                    "--handover-goal-quaternion-wxyz",
                    *[str(value) for value in args.handover_goal_quaternion_wxyz],
                ]
            )
        if args.handover_receiver_position_robot_base_m is not None:
            command.extend(
                [
                    "--handover-receiver-position-robot-base-m",
                    *[
                        str(value)
                        for value in args.handover_receiver_position_robot_base_m
                    ],
                    "--handover-human-direction-robot-base",
                    *[
                        str(value)
                        for value in args.handover_human_direction_robot_base
                    ],
                    "--handover-height-policy",
                    args.handover_height_policy,
                ]
            )
        elif args.handover_region_json is not None:
            command.extend(
                [
                    "--handover-region-json",
                    str(args.handover_region_json),
                    "--handover-region-samples-per-axis",
                    str(args.handover_region_samples_per_axis),
                ]
            )
        if args.transport_orientation_policy_report is not None:
            command.extend(
                [
                    "--transport-orientation-policy-report",
                    str(args.transport_orientation_policy_report),
                ]
            )
        if args.diagnostic_gravity_tilt_tolerance_deg is not None:
            command.extend(
                [
                    "--diagnostic-gravity-tilt-tolerance-deg",
                    str(args.diagnostic_gravity_tilt_tolerance_deg),
                ]
            )
        if args.semantic_orientation_cost:
            command.append("--semantic-orientation-cost")
        print(
            f"=== planning candidate {source_index} "
            f"(score rank {original_rank}, score {float(score):.6f}) ===",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        plan_status, report_path = read_plan_status(trial_directory)
        accepted = completed.returncode == 0 and plan_status == "success"
        semantic_cost = None
        direction_error_rad = None
        direction_error_deg = None
        horizontal_priority_rank = None
        maximum_transport_orientation_deviation_rad = None
        if accepted and report_path is not None:
            plan_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            result_report = optional_mapping(plan_report.get("result"))
            transport_report = optional_mapping(result_report.get("transport"))
            orientation_validation = optional_mapping(
                transport_report.get("orientation_validation")
            )
            soft_cost_total = optional_mapping(
                orientation_validation.get("soft_cost")
            ).get("total")
            if soft_cost_total is not None:
                semantic_cost = _finite_metric(
                    soft_cost_total,
                    name="semantic orientation cost",
                    source_index=source_index,
                )
            if automatic_handover:
                goal_report = optional_mapping(transport_report.get("goal"))
                selected_direction = optional_mapping(
                    goal_report.get("selected_handover_goal_metadata")
                    or goal_report.get("selected_receive_direction_priority")
                )
                direction_error_rad = _finite_metric(
                    selected_direction.get("alignment_error_rad"),
                    name="receive-direction alignment error",
                    source_index=source_index,
                )
                direction_error_deg = _finite_metric(
                    selected_direction.get("alignment_error_deg"),
                    name="receive-direction alignment error in degrees",
                    source_index=source_index,
                )
                horizontal_priority_rank = int(
                    selected_direction["horizontal_priority_rank"]
                )
                maximum_transport_deviation = orientation_validation.get(
                    "maximum_transport_deviation_rad"
                )
                if maximum_transport_deviation is not None:
                    maximum_transport_orientation_deviation_rad = _finite_metric(
                        maximum_transport_deviation,
                        name="maximum transport orientation deviation",
                        source_index=source_index,
                    )
        attempt = {
            "original_goalset_rank": original_rank,
            "source_candidate_index": source_index,
            "graspgenx_score": float(score),
            "output": str(trial_directory),
            "return_code": completed.returncode,
            "plan_status": plan_status,
            "report": report_path,
            "planner_feasible": accepted,
            "available_for_physical_trial": False,
            "semantic_orientation_cost": semantic_cost,
            "handover_horizontal_priority_rank": horizontal_priority_rank,
            "receive_direction_alignment_error_rad": direction_error_rad,
            "receive_direction_alignment_error_deg": direction_error_deg,
            "maximum_transport_orientation_deviation_rad": (
                maximum_transport_orientation_deviation_rad
            ),
        }
        manifest["attempts"].append(attempt)
        if accepted:
            planner_feasible_plans += 1
        write_manifest(manifest_path, manifest)
        if plan_status == "missing_report":
            manifest["status"] = "planning_error"
            write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"candidate {source_index} exited without a planner report; "
                "refusing to classify an infrastructure error as a rejected grasp"
            )

    selected_attempts = select_physical_trial_attempts(
        manifest["attempts"],
        limit=args.max_physical_trials,
        automatic_handover=automatic_handover,
        semantic_orientation_cost=args.semantic_orientation_cost,
    )
    selected_outputs = {attempt["output"] for attempt in selected_attempts}
    for attempt in manifest["attempts"]:
        attempt["available_for_physical_trial"] = (
            attempt["output"] in selected_outputs
        )
    manifest["successful_plan_directories"] = [
        attempt["output"] for attempt in selected_attempts
    ]
    manifest["status"] = "plans_ready" if selected_attempts else "no_plans_ready"
    manifest["planner_feasible_plan_count"] = planner_feasible_plans
    manifest["successful_plan_count"] = len(selected_attempts)
    manifest["selected_physical_trial_count"] = len(selected_attempts)
    eligible_candidate_count = int(
        np.count_nonzero(
            ~np.isin(source_indices, args.exclude_source_candidate_index)
        )
    )
    total_eligible_candidate_count = int(
        total_candidate_pool_count
        - np.count_nonzero(
            np.isin(
                all_source_indices,
                args.exclude_source_candidate_index,
            )
        )
    )
    manifest["eligible_candidate_pool_count"] = total_eligible_candidate_count
    manifest["eligible_candidate_prefix_count"] = eligible_candidate_count
    manifest["evaluated_candidate_prefix_exhausted"] = (
        len(manifest["attempts"]) == eligible_candidate_count
    )
    manifest["candidate_pool_exhausted"] = (
        not candidate_pool_truncated
        and len(manifest["attempts"]) == total_eligible_candidate_count
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved: {manifest_path}", flush=True)
    return 0 if selected_attempts else 2


if __name__ == "__main__":
    raise SystemExit(main())
