#!/usr/bin/env python3
"""Run the existing SAM3 multi-prompt implementation on a saved RGB-D capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.segmentation import (
    infer_sam3_prompts,
    save_prompt_overlap_report,
    save_segmentation_artifacts,
)
from panda_handover.vlm_parts import load_handover_parts_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Manual whole-object SAM3 prompt")
    source.add_argument("--vlm-result", type=Path)
    parser.add_argument("--grasp-part-prompt")
    parser.add_argument("--receive-part-prompt")
    parser.add_argument("--sam3-model-id", default="facebook/sam3")
    parser.add_argument("--sam3-score-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-mask-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--sam3-allow-download", action="store_true")
    return parser.parse_args()


def resolve_prompts(args: argparse.Namespace) -> dict[str, str]:
    if args.vlm_result is not None:
        if args.grasp_part_prompt or args.receive_part_prompt:
            raise ValueError("manual part prompts cannot be combined with --vlm-result")
        return load_handover_parts_report(args.vlm_result).to_dict()
    object_prompt = str(args.prompt).strip()
    if not object_prompt:
        raise ValueError("manual object prompt must not be empty")
    prompts = {"object": object_prompt}
    if bool(args.grasp_part_prompt) != bool(args.receive_part_prompt):
        raise ValueError("manual grasp and receive part prompts must be supplied together")
    if args.grasp_part_prompt:
        prompts["grasp_part"] = args.grasp_part_prompt.strip()
        prompts["receive_part"] = args.receive_part_prompt.strip()
    if any(not prompt for prompt in prompts.values()):
        raise ValueError("SAM3 prompts must not be empty")
    if len(set(prompts.values())) != len(prompts):
        raise ValueError("SAM3 prompts must be unique")
    return prompts


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "run_status.json"
    try:
        role_prompts = resolve_prompts(args)
        rgb = np.load(args.capture / "rgb.npy")
        depth_m = np.load(args.capture / "depth_m.npy")
        points_camera = np.load(args.capture / "points_camera.npy")
        points_world = np.load(args.capture / "points_world.npy")
        predictions = infer_sam3_prompts(
            rgb,
            list(role_prompts.values()),
            model_id=args.sam3_model_id,
            device=args.sam3_device,
            dtype=args.sam3_dtype,
            score_threshold=args.sam3_score_threshold,
            mask_threshold=args.sam3_mask_threshold,
            local_files_only=not args.sam3_allow_download,
        )
        reports = {}
        directories = {}
        for role, prompt in role_prompts.items():
            directory = args.output if role == "object" else args.output / "parts" / role
            directories[role] = str(directory.resolve())
            reports[role] = save_segmentation_artifacts(
                directory,
                rgb=rgb,
                depth_m=depth_m,
                points_camera=points_camera,
                points_world=points_world,
                prediction=predictions[prompt],
                prompt=prompt,
                model_id=args.sam3_model_id,
                score_threshold=args.sam3_score_threshold,
                mask_threshold=args.sam3_mask_threshold,
            )
        overlap = save_prompt_overlap_report(
            args.output, predictions, tuple(rgb.shape[:2])
        )
        passed = all(report["automatic_checks_passed"] for report in reports.values())
        passed = passed and overlap["automatic_checks_passed"]
        report = {
            "status": "success" if passed else "failed",
            "inputs": {
                "capture": str(args.capture.resolve()),
                "vlm_result": str(args.vlm_result.resolve()) if args.vlm_result else None,
            },
            "prompts": role_prompts,
            "prompt_directories": directories,
            "automatic_checks_passed": passed,
            "manual_review_required": True,
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "automatic_checks_passed": False,
            "manual_review_required": True,
        }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {report_path}")
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
