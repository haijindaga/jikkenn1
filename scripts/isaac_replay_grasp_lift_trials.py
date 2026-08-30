#!/usr/bin/env python3
"""Replay candidate-specific grasp/lift plans until one physical pick succeeds."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.simulation_only:
        parser.error("--simulation-only is required")
    if args.max_physical_trials <= 0:
        parser.error("--max-physical-trials must be positive")
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
            "maximum_physical_trials": args.max_physical_trials,
            "stop_at_first_physical_pick": True,
            "finger_drive_preset_for_every_candidate": args.finger_drive_preset,
            "candidate_specific_parameter_tuning": False,
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
            "--simulation-only",
        ]
        if args.headless:
            command.append("--headless")
        print(
            f"=== physical trial {trial_number}/{len(plan_attempts)}: "
            f"candidate {source_index} ===",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        report_path = trial_output / "grasp_lift_replay_check.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {"status": "missing_report"}
        )
        status = str(report.get("status", "unknown"))
        physical_pick = bool(
            report.get("physical_object", {}).get("physical_pick_observed", False)
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
            "report": str(report_path) if report_path.exists() else None,
        }
        summary["attempts"].append(attempt)
        if completed.returncode == 0 and status == "success" and physical_pick:
            summary["status"] = "success"
            summary["selected_success"] = attempt
            write_summary(summary_path, summary)
            print(f"physical pick succeeded with candidate {source_index}", flush=True)
            print(f"saved: {summary_path}", flush=True)
            return 0
        if status != "physical_pick_not_observed":
            summary["status"] = "replay_error"
            write_summary(summary_path, summary)
            raise RuntimeError(
                f"candidate {source_index} ended with non-physical failure status "
                f"{status!r}; refusing to hide it by trying another candidate"
            )
        write_summary(summary_path, summary)

    summary["status"] = "physical_pick_not_observed"
    write_summary(summary_path, summary)
    print("all available candidate replays failed to retain the object", flush=True)
    print(f"saved: {summary_path}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
