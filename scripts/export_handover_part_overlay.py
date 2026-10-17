#!/usr/bin/env python3
"""Combine saved grasp/receive SAM3 masks on one RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.segmentation import save_handover_part_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.segmentation
    rgb = np.load(args.capture / "rgb.npy")
    grasp_mask = np.load(args.segmentation / "parts" / "grasp_part" / "union_mask.npy")
    receive_mask = np.load(args.segmentation / "parts" / "receive_part" / "union_mask.npy")
    report = save_handover_part_overlay(
        output,
        rgb=rgb,
        grasp_part_mask=grasp_mask,
        receive_part_mask=receive_mask,
        alpha=args.alpha,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
