#!/usr/bin/env python3
"""Discover object/grasp/receive prompts with Ollama structured output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.vlm_parts import discover_handover_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "vlm_part_discovery.json"
    try:
        parts, metadata = discover_handover_parts(
            args.capture / "rgb.png",
            target_object=args.target_object,
            model=args.model,
            base_url=args.ollama_url,
            timeout_s=args.timeout_s,
        )
        report = {
            "status": "success",
            "inputs": {
                "capture": str(args.capture.resolve()),
                "target_object": args.target_object.strip(),
            },
            "parameters": {
                "model": args.model,
                "ollama_url": args.ollama_url,
                "timeout_s": args.timeout_s,
            },
            "result": parts.to_dict(),
            "provenance": metadata,
            "automatic_checks_passed": True,
            "manual_review_required": True,
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "inputs": {
                "capture": str(args.capture.resolve()),
                "target_object": args.target_object.strip(),
            },
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "automatic_checks_passed": False,
            "manual_review_required": True,
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"VLM part discovery failed: {exc}")
        print(f"saved: {report_path}")
        return 2
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(parts.to_dict(), indent=2, ensure_ascii=False))
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
