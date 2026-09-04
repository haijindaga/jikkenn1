#!/usr/bin/env python3
"""Generate and open the standard HTML report for an existing output tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from panda_handover.result_report import generate_result_report, open_result_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    root = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / "pipeline_status.json"
    )
    report = generate_result_report(root, manifest_path=manifest)
    print(f"results: {report}")
    if not args.no_open and not open_result_report(report):
        print("the browser did not confirm opening the report; use the path above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
