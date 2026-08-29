#!/usr/bin/env python3
"""Verify that local baseline USDs are unchanged RoboLab assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "robolab_baseline_assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing the upstream-style assets/objects tree",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--asset",
        action="append",
        dest="asset_names",
        help="Asset name to verify; repeat for several. Defaults to every manifest asset.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    names = args.asset_names or sorted(assets)
    unknown = sorted(set(names) - set(assets))
    if unknown:
        parser.error(f"unknown manifest assets: {', '.join(unknown)}")

    checks = []
    all_passed = True
    for name in names:
        record = assets[name]
        usd_path = (args.asset_root / record["path"]).resolve()
        license_path = (args.asset_root / record["license_path"]).resolve()
        actual_hash = sha256(usd_path) if usd_path.is_file() else None
        passed = actual_hash == record["sha256"] and license_path.is_file()
        all_passed = all_passed and passed
        checks.append(
            {
                "asset": name,
                "usd_path": str(usd_path),
                "usd_exists": usd_path.is_file(),
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "usd_matches_pinned_robolab_asset": actual_hash == record["sha256"],
                "license": record["license"],
                "license_path": str(license_path),
                "license_exists": license_path.is_file(),
                "passed": passed,
            }
        )

    result = {
        "status": "success" if all_passed else "verification_failed",
        "source_repository": manifest["source_repository"],
        "source_commit": manifest["source_commit"],
        "checks": checks,
        "automatic_checks_passed": all_passed,
    }
    print(json.dumps(result, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
