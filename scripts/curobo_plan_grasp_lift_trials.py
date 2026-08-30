#!/usr/bin/env python3
"""Create separate cuRobo grasp/lift plans for several score-ordered candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--pregrasp-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-physical-trials", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument(
        "--allow-reviewed-support-contact-preflight", action="store_true"
    )
    args = parser.parse_args()
    if args.max_physical_trials <= 0:
        parser.error("--max-physical-trials must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    return args


def read_plan_status(directory: Path) -> tuple[str, str | None]:
    report_names = (
        "grasp_lift_plan_check.json",
        "grasp_preflight_failure.json",
        "grasp_lift_failure.json",
    )
    for name in report_names:
        path = directory / name
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            return str(report.get("status", "unknown")), str(path)
    return "missing_report", None


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
        },
        "policy": {
            "candidate_order": "pregrasp score order",
            "maximum_physical_trials": args.max_physical_trials,
            "candidate_specific_parameter_tuning": False,
            "continue_after_planning_rejection": True,
        },
        "candidate_pool_count": int(len(source_indices)),
        "attempts": [],
        "successful_plan_directories": [],
    }
    write_manifest(manifest_path, manifest)

    planner_script = Path(__file__).with_name("curobo_plan_grasp_lift.py")
    successful_plans = 0
    for original_rank, (source_index, score) in enumerate(
        zip(source_indices, scores, strict=True)
    ):
        if successful_plans >= args.max_physical_trials:
            break
        source_index = int(source_index)
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
            "--max-attempts",
            str(args.max_attempts),
        ]
        if args.allow_reviewed_support_contact_preflight:
            command.append("--allow-reviewed-support-contact-preflight")
        print(
            f"=== planning candidate {source_index} "
            f"(score rank {original_rank}, score {float(score):.6f}) ===",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        plan_status, report_path = read_plan_status(trial_directory)
        accepted = completed.returncode == 0 and plan_status == "success"
        attempt = {
            "original_goalset_rank": original_rank,
            "source_candidate_index": source_index,
            "graspgenx_score": float(score),
            "output": str(trial_directory),
            "return_code": completed.returncode,
            "plan_status": plan_status,
            "report": report_path,
            "available_for_physical_trial": accepted,
        }
        manifest["attempts"].append(attempt)
        if accepted:
            successful_plans += 1
            manifest["successful_plan_directories"].append(str(trial_directory))
        write_manifest(manifest_path, manifest)
        if plan_status == "missing_report":
            manifest["status"] = "planning_error"
            write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"candidate {source_index} exited without a planner report; "
                "refusing to classify an infrastructure error as a rejected grasp"
            )

    manifest["status"] = (
        "plans_ready" if manifest["successful_plan_directories"] else "no_plans_ready"
    )
    manifest["successful_plan_count"] = successful_plans
    manifest["candidate_pool_exhausted"] = len(manifest["attempts"]) == len(
        source_indices
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved: {manifest_path}", flush=True)
    return 0 if successful_plans else 2


if __name__ == "__main__":
    raise SystemExit(main())
