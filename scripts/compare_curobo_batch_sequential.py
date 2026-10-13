#!/usr/bin/env python3
"""Compare batch grasp/lift feasibility with a sequential trial manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--sequential-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from panda_handover.batch_planning import compare_batch_to_sequential

    batch = json.loads(args.batch_report.read_text(encoding="utf-8"))
    sequential = json.loads(args.sequential_manifest.read_text(encoding="utf-8"))
    if batch.get("status") != "success":
        raise ValueError("batch diagnostic did not succeed")
    if sequential.get("status") not in {"plans_ready", "no_plans_ready"}:
        raise ValueError("sequential manifest is not complete")
    report = compare_batch_to_sequential(
        batch.get("candidates", []), sequential.get("attempts", [])
    )
    report["inputs"] = {
        "batch_report": str(args.batch_report),
        "sequential_manifest": str(args.sequential_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {args.output}")
    return 0 if report["status"] != "feasibility_mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
