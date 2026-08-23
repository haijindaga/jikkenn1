#!/usr/bin/env python3
"""Apply the exact cuRobo Issue #699 fix to one pinned source checkout.

The script refuses unknown commits, dirty source trees, and unexpected file
contents. It is intentionally not a general-purpose patcher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys


CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"
TARGET_RELATIVE = Path("curobo/_src/geom/data/data_voxel.py")
ORIGINAL_SHA256 = "d2343d5453bf93164bae25611e8ad9e45f83f36dc1da8d3b2a2a748da38f7a31"
PATCHED_SHA256 = "bab10d99e555fe722f2c3d893425ea9126978238ee43b9f6c0e250875c10e004"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="cuRobo Git checkout root; defaults to the imported curobo package",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the exact patched state without changing files",
    )
    return parser.parse_args()


def run_git(source: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def discover_source() -> Path:
    spec = importlib.util.find_spec("curobo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("curobo is not importable; pass --source explicitly")
    package = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package.parent


def sha256(path: Path) -> str:
    # Git may materialize CRLF on Windows; the reviewed source and Ubuntu
    # checkout use LF. Hash normalized bytes so provenance is cross-platform.
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def expected_patch_file() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "curobo"
        / "057a96f-voxel-grid-round.patch"
    )


def verify_patched_state(source: Path, target: Path) -> None:
    actual_hash = sha256(target)
    if actual_hash != PATCHED_SHA256:
        raise RuntimeError(
            "cuRobo voxel source is not the reviewed patched file; "
            f"expected {PATCHED_SHA256}, found {actual_hash}"
        )
    status = run_git(source, "status", "--porcelain", "--untracked-files=no")
    # run_git strips surrounding whitespace, including porcelain's leading
    # unstaged-status column.
    expected_status = f"M {TARGET_RELATIVE.as_posix()}"
    if status != expected_status:
        raise RuntimeError(
            "the pinned cuRobo checkout has unexpected tracked changes; "
            f"expected only {expected_status!r}, found {status!r}"
        )
    run_git(source, "diff", "--check", "--", TARGET_RELATIVE.as_posix())


def main() -> int:
    args = parse_args()
    source = (args.source or discover_source()).resolve()
    target = source / TARGET_RELATIVE
    if not (source / ".git").exists() or not target.is_file():
        raise RuntimeError(f"not a cuRobo source checkout: {source}")

    commit = run_git(source, "rev-parse", "HEAD")
    if commit != CUROBO_COMMIT:
        raise RuntimeError(
            "refusing to patch an unreviewed cuRobo commit; "
            f"expected {CUROBO_COMMIT}, found {commit}"
        )

    actual_hash = sha256(target)
    if actual_hash == PATCHED_SHA256:
        verify_patched_state(source, target)
        print(f"cuRobo Issue #699 fix already verified at {source}")
        return 0
    if args.check_only:
        raise RuntimeError("cuRobo Issue #699 fix is not applied")
    if actual_hash != ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to modify unexpected cuRobo source contents; "
            f"expected {ORIGINAL_SHA256}, found {actual_hash}"
        )
    status = run_git(source, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "refusing to patch a dirty cuRobo checkout; tracked changes are:\n" + status
        )

    patch = expected_patch_file()
    if not patch.is_file():
        raise RuntimeError(f"reviewed patch is missing: {patch}")
    run_git(source, "apply", "--check", str(patch))
    run_git(source, "apply", str(patch))
    verify_patched_state(source, target)
    print("Applied cuRobo Issue #699 voxel-dimension rounding fix")
    print(f"source: {source}")
    print(f"commit: {CUROBO_COMMIT}")
    print(f"patched sha256: {PATCHED_SHA256}")
    print(
        "revert with: git -C "
        + repr(str(source))
        + " restore -- "
        + TARGET_RELATIVE.as_posix()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
