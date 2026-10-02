#!/usr/bin/env python3
"""Compose a visual receiver over an existing scene without launching Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT


DEFAULT_EVIDENCE = (
    repo_root / "config" / "receiver_assets" / "isaac_sim_5_1_1x_neo.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usd_number(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("USD transform values must be finite")
    return format(value, ".17g")


def build_overlay_text(
    *,
    base_reference: str,
    receiver_url: str,
    translation: tuple[float, float, float],
    yaw_deg: float,
) -> str:
    tx, ty, tz = (usd_number(value) for value in translation)
    yaw = usd_number(yaw_deg)
    return f'''#usda 1.0
(
    defaultPrim = "World"
    subLayers = [
        @{base_reference}@
    ]
)

over "World"
{{
    def Xform "Receiver" (
        customData = {{
            string assetName = "1x-neo"
            string role = "visual-only handover receiver"
            bool sourceAssetModified = false
        }}
    )
    {{
        double3 xformOp:translate = ({tx}, {ty}, {tz})
        float3 xformOp:rotateXYZ = (0, 0, {yaw})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

        def Xform "Asset" (
            prepend references = @{receiver_url}@
        )
        {{
        }}
    }}
}}
'''


def create_overlay(
    *,
    base_scene: Path,
    output: Path,
    evidence_path: Path,
    receiver_root_xy: tuple[float, float],
    receiver_yaw_deg: float,
    overwrite: bool,
) -> dict:
    base_scene = base_scene.expanduser().resolve()
    output = output.expanduser().resolve()
    evidence_path = evidence_path.expanduser().resolve()

    if not base_scene.is_file():
        raise FileNotFoundError(f"base scene does not exist: {base_scene}")
    if base_scene.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("base scene must be a USD file")
    if output.suffix.lower() != ".usda":
        raise ValueError("receiver overlay output must end in .usda")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; use a new name or pass --overwrite"
        )
    if not evidence_path.is_file():
        raise FileNotFoundError(f"receiver evidence does not exist: {evidence_path}")
    if not all(math.isfinite(float(value)) for value in receiver_root_xy):
        raise ValueError("receiver root XY values must be finite")
    if not math.isfinite(float(receiver_yaw_deg)):
        raise ValueError("receiver yaw must be finite")

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "measured":
        raise ValueError("receiver evidence must have status=measured")
    if evidence.get("asset_name") != "1x-neo":
        raise ValueError("receiver evidence is not for 1X NEO")
    measurement = evidence.get("measurement", {})
    root_to_lowest_z_m = float(measurement["root_to_lowest_z_m"])
    if not math.isfinite(root_to_lowest_z_m) or root_to_lowest_z_m >= 0.0:
        raise ValueError("receiver root-to-lowest-Z measurement is invalid")

    floor_z_m = float(DEFAULT_TABLETOP_LAYOUT.ground_z_m)
    receiver_root_z_m = floor_z_m - root_to_lowest_z_m
    translation = (
        float(receiver_root_xy[0]),
        float(receiver_root_xy[1]),
        receiver_root_z_m,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    base_reference = os.path.relpath(base_scene, start=output.parent).replace(
        os.sep, "/"
    )
    overlay_text = build_overlay_text(
        base_reference=base_reference,
        receiver_url=str(evidence["usd_url"]),
        translation=translation,
        yaw_deg=float(receiver_yaw_deg),
    )
    output.write_text(overlay_text, encoding="utf-8", newline="\n")

    checks = {
        "base_scene_exists": base_scene.is_file(),
        "receiver_evidence_is_measured": evidence["status"] == "measured",
        "computed_lowest_z_matches_floor": math.isclose(
            receiver_root_z_m + root_to_lowest_z_m,
            floor_z_m,
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
        "receiver_root_is_behind_robot": translation[0] < 0.0,
        "source_asset_is_not_modified": True,
        "base_scene_is_not_modified": True,
    }
    report = {
        "status": "success" if all(checks.values()) else "failure",
        "purpose": "visual-only 1X NEO receiver overlay",
        "output": str(output),
        "composition": {
            "method": "OpenUSD subLayer plus reference",
            "base_scene": str(base_scene),
            "base_scene_sha256": sha256_file(base_scene),
            "base_reference": base_reference,
            "receiver_evidence": str(evidence_path),
            "receiver_evidence_sha256": sha256_file(evidence_path),
            "receiver_usd": evidence["usd_url"],
            "source_asset_modified": False,
            "base_scene_modified": False,
            "physics_apis_modified": False,
            "asset_rescaled": False,
        },
        "placement": {
            "receiver_root_translation_world_m": list(translation),
            "receiver_yaw_world_z_deg": float(receiver_yaw_deg),
            "floor_z_m": floor_z_m,
            "measured_root_to_lowest_z_m": root_to_lowest_z_m,
            "root_z_formula": "floor_z_m - measured_root_to_lowest_z_m",
            "orientation_basis": (
                "Isaac Sim world convention uses +X forward; visual review of "
                "the asset-local facing direction is still required"
            ),
        },
        "safety": {
            "presentation_only": True,
            "timeline_must_remain_stopped": True,
            "human_collision_model_present": False,
            "safe_for_physical_replay": False,
        },
        "automatic_checks": checks,
    }
    report_path = output.with_suffix(output.suffix + ".check.json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if report["status"] != "success":
        raise RuntimeError(f"receiver overlay validation failed: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver-evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--receiver-root-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(-1.15, 0.0),
    )
    parser.add_argument("--receiver-yaw-deg", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = create_overlay(
        base_scene=args.base_scene,
        output=args.output,
        evidence_path=args.receiver_evidence,
        receiver_root_xy=tuple(args.receiver_root_xy),
        receiver_yaw_deg=args.receiver_yaw_deg,
        overwrite=args.overwrite,
    )
    output = Path(report["output"])
    print(f"saved receiver overlay: {output}")
    print(f"saved validation report: {output.with_suffix(output.suffix + '.check.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
