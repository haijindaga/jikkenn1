#!/usr/bin/env python3
"""Compare two Isaac grasp/lift replay reports without rerunning simulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPORT_NAME = "grasp_lift_replay_check.json"


def report_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_dir():
        path = path / REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"replay report does not exist: {path}")
    return path


def finite_number(value: object, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def first_configured_drive(report: dict) -> dict:
    drives = report.get("physical_parameters", {}).get("finger_joint_drives", [])
    return next((item for item in drives if item.get("found") is True), {})


def effective_drive_value(drive: dict, name: str) -> object:
    return drive.get(f"{name}_after", drive.get(name))


def summary(path: Path, report: dict) -> dict:
    physical = report.get("physical_object", {})
    retention = report.get("retention_diagnostics", {})
    parameters = report.get("physical_parameters", {})
    drive = first_configured_drive(report)
    return {
        "report": str(path),
        "status": report.get("status"),
        "capture": report.get("inputs", {}).get("capture"),
        "plan": report.get("inputs", {}).get("plan"),
        "scene_usd": report.get("inputs", {}).get("scene_usd"),
        "target_effective_mass_kg": parameters.get("target_effective_mass_kg"),
        "finger_drive_preset": parameters.get("finger_drive_preset", "legacy-or-unknown"),
        "finger_drive": {
            "max_force": effective_drive_value(drive, "max_force"),
            "stiffness": effective_drive_value(drive, "stiffness"),
            "damping": effective_drive_value(drive, "damping"),
        },
        "physical_pick_observed": physical.get("physical_pick_observed"),
        "peak_object_lift_m": retention.get("peak_object_lift_m"),
        "lift_after_hold_m": physical.get("lift_after_hold_m"),
        "lift_lost_from_peak_to_final_m": retention.get(
            "lift_lost_from_peak_to_final_m"
        ),
        "finger_gap_at_peak_lift_m": retention.get("finger_gap_at_peak_lift_m"),
        "finger_gap_at_final_hold_m": retention.get("finger_gap_at_final_hold_m"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = report_path(args.baseline)
    candidate_path = report_path(args.candidate)
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = summary(baseline_path, baseline_report)
    candidate = summary(candidate_path, candidate_report)

    controlled_fields = ("capture", "plan", "scene_usd", "target_effective_mass_kg")
    mismatches = {
        field: {"baseline": baseline[field], "candidate": candidate[field]}
        for field in controlled_fields
        if baseline[field] != candidate[field]
    }
    metric_fields = (
        "peak_object_lift_m",
        "lift_after_hold_m",
        "lift_lost_from_peak_to_final_m",
        "finger_gap_at_peak_lift_m",
        "finger_gap_at_final_hold_m",
    )
    deltas = {
        field: finite_number(candidate[field], field=f"candidate.{field}")
        - finite_number(baseline[field], field=f"baseline.{field}")
        for field in metric_fields
    }
    result = {
        "status": "success",
        "comparison_is_controlled": not mismatches,
        "controlled_input_mismatches": mismatches,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "interpretation": {
            "candidate_physical_pick_observed": candidate["physical_pick_observed"],
            "positive_held_lift_delta_means_better_retention": True,
            "negative_lift_loss_delta_means_less_slip_or_drop": True,
            "planning_success_is_not_reclassified_by_this_comparison": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
