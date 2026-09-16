#!/usr/bin/env python3
"""Replay candidate plans until one configured grasp-retention trial succeeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan-trials", type=Path, required=True)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-physical-trials", type=int, default=5)
    parser.add_argument(
        "--finger-drive-preset",
        choices=("authored-usd", "isaaclab-franka"),
        default="isaaclab-franka",
    )
    parser.add_argument("--finger-drive-scale", type=float, default=1.0)
    parser.add_argument(
        "--fingertip-friction-coefficient",
        type=float,
        help=(
            "Simulation-only static/dynamic fingertip friction override passed "
            "unchanged to every candidate replay"
        ),
    )
    parser.add_argument(
        "--grasp-retention-mode",
        choices=(
            "physics",
            "physx-auto-attachment",
            "rigid-attachment",
            "surface-gripper-attachment",
            "kinematic-pose-lock",
        ),
        default="physics",
    )
    parser.add_argument("--solver-position-iterations", type=int)
    parser.add_argument("--solver-velocity-iterations", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--replay-physics", choices=("default", "cpu"), default="default",
        help="Replay-only CPU PhysX + Fabric OFF diagnostic",
    )
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.simulation_only:
        parser.error("--simulation-only is required")
    if args.max_physical_trials <= 0:
        parser.error("--max-physical-trials must be positive")
    if not math.isfinite(args.finger_drive_scale) or args.finger_drive_scale <= 0.0:
        parser.error("--finger-drive-scale must be positive and finite")
    if args.finger_drive_scale != 1.0 and args.finger_drive_preset != "isaaclab-franka":
        parser.error("scaled diagnostics require --finger-drive-preset isaaclab-franka")
    if args.fingertip_friction_coefficient is not None and (
        not math.isfinite(args.fingertip_friction_coefficient)
        or args.fingertip_friction_coefficient < 0.0
    ):
        parser.error("--fingertip-friction-coefficient must be finite and non-negative")
    if (
        args.fingertip_friction_coefficient is not None
        and args.grasp_retention_mode != "physics"
    ):
        parser.error(
            "--fingertip-friction-coefficient is meaningful only with "
            "--grasp-retention-mode physics"
        )
    if (
        args.fingertip_friction_coefficient is not None
        and args.finger_drive_scale != 1.0
    ):
        parser.error(
            "do not combine fingertip friction and finger-drive diagnostics in "
            "one controlled run"
        )
    solver_iteration_values = (
        args.solver_position_iterations,
        args.solver_velocity_iterations,
    )
    if (solver_iteration_values[0] is None) != (solver_iteration_values[1] is None):
        parser.error(
            "--solver-position-iterations and --solver-velocity-iterations must "
            "be supplied together"
        )
    if args.solver_position_iterations is not None:
        if not 1 <= args.solver_position_iterations <= 255:
            parser.error("--solver-position-iterations must be in 1..255")
        if not 0 <= args.solver_velocity_iterations <= 255:
            parser.error("--solver-velocity-iterations must be in 0..255")
        if args.grasp_retention_mode != "rigid-attachment":
            parser.error(
                "solver-iteration diagnostics currently require "
                "--grasp-retention-mode rigid-attachment"
            )
    return args


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    plan_manifest_path = args.plan_trials / "grasp_lift_trial_plans.json"
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    plan_attempts = [
        attempt
        for attempt in plan_manifest.get("attempts", [])
        if attempt.get("available_for_physical_trial") is True
    ][: args.max_physical_trials]
    if not plan_attempts:
        raise RuntimeError("the planning manifest contains no executable trial plans")

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"replay output already contains files: {output}; use a new output name"
        )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "grasp_lift_physical_trials.json"
    summary = {
        "status": "replay_in_progress",
        "inputs": {
            "capture": str(args.capture),
            "plan_trials": str(args.plan_trials),
            "scene_usd": str(args.scene_usd),
        },
        "policy": {
            "replay_physics": args.replay_physics,
            "maximum_physical_trials": args.max_physical_trials,
            "stop_at_first_success": True,
            "grasp_retention_mode": args.grasp_retention_mode,
            "rigid_attachment_means_grasp_success_is_assumed": bool(
                args.grasp_retention_mode == "rigid-attachment"
            ),
            "physx_auto_attachment_means_grasp_success_is_assumed": bool(
                args.grasp_retention_mode == "physx-auto-attachment"
            ),
            "surface_gripper_attachment_means_grasp_success_is_assumed": bool(
                args.grasp_retention_mode == "surface-gripper-attachment"
            ),
            "kinematic_pose_lock_means_grasp_success_is_assumed": bool(
                args.grasp_retention_mode == "kinematic-pose-lock"
            ),
            "finger_drive_preset_for_every_candidate": args.finger_drive_preset,
            "finger_drive_diagnostic_scale_for_every_candidate": args.finger_drive_scale,
            "fingertip_friction_coefficient_for_every_candidate": (
                args.fingertip_friction_coefficient
            ),
            "fingertip_friction_policy": (
                "same static and dynamic coefficient; PhysX max combine mode; "
                "target and table materials unchanged; simulation diagnostic only"
            ),
            "finger_drive_scaling_policy": (
                "max_force and stiffness multiplied by scale; damping multiplied "
                "by sqrt(scale); simulation diagnostic only"
            ),
            "hardware_force_calibrated": False,
            "candidate_specific_parameter_tuning": False,
            "solver_iteration_override_for_every_candidate": (
                {
                    "position": args.solver_position_iterations,
                    "velocity": args.solver_velocity_iterations,
                    "diagnostic_only": True,
                }
                if args.solver_position_iterations is not None
                else None
            ),
        },
        "attempts": [],
        "selected_success": None,
    }
    write_summary(summary_path, summary)

    replay_script = Path(__file__).with_name("isaac_replay_grasp_lift.py")
    for trial_number, plan_attempt in enumerate(plan_attempts, start=1):
        source_index = int(plan_attempt["source_candidate_index"])
        trial_output = output / f"trial_{trial_number:02d}_candidate_{source_index:03d}"
        command = [
            sys.executable,
            str(replay_script),
            "--capture",
            str(args.capture),
            "--plan",
            str(plan_attempt["output"]),
            "--scene-usd",
            str(args.scene_usd),
            "--output",
            str(trial_output),
            "--finger-drive-preset",
            args.finger_drive_preset,
            "--finger-drive-scale",
            str(args.finger_drive_scale),
            "--grasp-retention-mode",
            args.grasp_retention_mode,
            "--replay-physics",
            args.replay_physics,
            "--simulation-only",
        ]
        if args.fingertip_friction_coefficient is not None:
            command.extend(
                [
                    "--fingertip-friction-coefficient",
                    str(args.fingertip_friction_coefficient),
                ]
            )
        if args.solver_position_iterations is not None:
            command.extend(
                [
                    "--solver-position-iterations",
                    str(args.solver_position_iterations),
                    "--solver-velocity-iterations",
                    str(args.solver_velocity_iterations),
                ]
            )
        if args.headless:
            command.append("--headless")
        print(
            f"=== replay trial {trial_number}/{len(plan_attempts)}: "
            f"candidate {source_index} ===",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        report_path = trial_output / "grasp_lift_replay_check.json"
        attachment_gate_failure_path = (
            trial_output / "fixed_joint_attachment_gate_failure.json"
        )
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            effective_report_path = report_path
        elif attachment_gate_failure_path.exists():
            report = json.loads(
                attachment_gate_failure_path.read_text(encoding="utf-8")
            )
            effective_report_path = attachment_gate_failure_path
        else:
            report = {"status": "missing_report"}
            effective_report_path = None
        status = str(report.get("status", "unknown"))
        physical_pick = report.get("physical_object", {}).get(
            "physical_pick_observed"
        )
        execution_success = bool(
            report.get("execution", {}).get(
                "success_observed", physical_pick is True
            )
        )
        attempt = {
            "trial_number": trial_number,
            "source_candidate_index": source_index,
            "graspgenx_score": plan_attempt.get("graspgenx_score"),
            "plan": plan_attempt["output"],
            "output": str(trial_output),
            "return_code": completed.returncode,
            "replay_status": status,
            "physical_pick_observed": physical_pick,
            "execution_success_observed": execution_success,
            "execution_evidence_kind": report.get("execution", {}).get(
                "evidence_kind"
            ),
            "report": (
                str(effective_report_path)
                if effective_report_path is not None
                else None
            ),
        }
        summary["attempts"].append(attempt)
        if completed.returncode == 0 and status == "success" and execution_success:
            summary["status"] = "success"
            summary["selected_success"] = attempt
            write_summary(summary_path, summary)
            print(
                f"{args.grasp_retention_mode} replay succeeded with candidate "
                f"{source_index}",
                flush=True,
            )
            print(f"saved: {summary_path}", flush=True)
            return 0
        expected_failure_statuses = (
            {"physical_pick_not_observed"}
            if args.grasp_retention_mode == "physics"
            else {
                "assumed_grasp_execution_failed",
                "attachment_pose_not_retained",
                "fixed_joint_attachment_gate_failed",
            }
        )
        if status not in expected_failure_statuses:
            summary["status"] = "replay_error"
            write_summary(summary_path, summary)
            raise RuntimeError(
                f"candidate {source_index} ended with non-physical failure status "
                f"{status!r}; refusing to hide it by trying another candidate"
            )
        write_summary(summary_path, summary)

    summary["status"] = (
        "physical_pick_not_observed"
        if args.grasp_retention_mode == "physics"
        else "assumed_grasp_execution_failed"
    )
    write_summary(summary_path, summary)
    print("all available candidate replays failed to retain the object", flush=True)
    print(f"saved: {summary_path}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
