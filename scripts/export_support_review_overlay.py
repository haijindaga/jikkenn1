#!/usr/bin/env python3
"""Export segmented 3-D parts for the dependency-free Windows scene editor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import numpy as np

from panda_handover.support_fixture import (
    build_support_review_overlay,
    masked_points_world,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grasp-mask-role",
        choices=("grasp_part", "receive_part"),
        default="grasp_part",
    )
    parser.add_argument(
        "--support-mask-role",
        choices=("grasp_part", "receive_part"),
        default="receive_part",
    )
    parser.add_argument("--grasp-part-label", required=True)
    parser.add_argument("--support-part-label", required=True)
    parser.add_argument("--max-points-per-part", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.capture = args.capture.expanduser().resolve()
    args.segmentation = args.segmentation.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.grasp_mask_role == args.support_mask_role:
        parser.error("--grasp-mask-role and --support-mask-role must differ")
    if args.max_points_per_part < 3:
        parser.error("--max-points-per-part must be at least three")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite intentionally")
    return args


def main() -> int:
    args = parse_args()
    points_path = args.capture / "points_camera.npy"
    transform_path = args.capture / "T_world_camera.npy"
    grasp_mask_path = (
        args.segmentation / "parts" / args.grasp_mask_role / "union_mask.npy"
    )
    support_mask_path = (
        args.segmentation / "parts" / args.support_mask_role / "union_mask.npy"
    )
    for path in (
        points_path,
        transform_path,
        grasp_mask_path,
        support_mask_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    points_camera = np.load(points_path, allow_pickle=False)
    transform = np.load(transform_path, allow_pickle=False)
    grasp_points = masked_points_world(
        points_camera,
        np.load(grasp_mask_path, allow_pickle=False),
        transform,
    )
    support_points = masked_points_world(
        points_camera,
        np.load(support_mask_path, allow_pickle=False),
        transform,
    )
    overlay = build_support_review_overlay(
        grasp_points,
        support_points,
        grasp_part_label=args.grasp_part_label,
        support_part_label=args.support_part_label,
        max_points_per_part=args.max_points_per_part,
    )
    overlay["inputs"] = {
        "capture": str(args.capture),
        "segmentation": str(args.segmentation),
        "grasp_mask_role": args.grasp_mask_role,
        "support_mask_role": args.support_mask_role,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(overlay, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(args.output),
                "grasp_source_points": len(grasp_points),
                "support_source_points": len(support_points),
                "display_points_per_part_maximum": args.max_points_per_part,
                "acceptance_uses_complete_points": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
