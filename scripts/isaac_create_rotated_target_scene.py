#!/usr/bin/env python3
"""Create a new USD scene with only the target yaw changed.

The source USD is opened but never saved.  The target is rotated about its own
world-space origin around world +Z, then the composed stage is exported to a
new path together with a machine-readable validation report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=180.0,
        help="target rotation about world +Z (default: 180 degrees)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_scene = args.base_scene.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.base_scene.is_file():
        parser.error(f"base scene does not exist: {args.base_scene}")
    if args.output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        parser.error("--output must end in .usd, .usda, or .usdc")
    if args.output == args.base_scene:
        parser.error("--output must differ from --base-scene")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite intentionally")
    if not args.target_prim.startswith("/") or args.target_prim == "/":
        parser.error("--target-prim must be an absolute non-root prim path")
    if not math.isfinite(args.yaw_deg):
        parser.error("--yaw-deg must be finite")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


args = parse_args()

from isaacsim import SimulationApp


# Kit is started only to register Isaac Sim's packaged Pixar USD plugins.  No
# timeline or physics step is advanced by this scene-authoring utility.
simulation_app = SimulationApp(
    {
        "headless": True,
        "disable_viewport_updates": True,
    }
)

import numpy as np
from pxr import Gf, Usd, UsdGeom

from panda_handover.scene_rotation import (
    local_pose_from_world_poses,
    rotate_world_pose_about_own_origin,
)


def _gf_to_column_transform(matrix: Gf.Matrix4d) -> np.ndarray:
    """Convert USD's row-vector matrix representation to column convention."""

    return np.asarray(matrix, dtype=np.float64).T.copy()


def _column_transform_to_gf(matrix: np.ndarray) -> Gf.Matrix4d:
    """Convert a column-vector homogeneous transform to ``Gf.Matrix4d``."""

    row_matrix = np.asarray(matrix, dtype=np.float64).T
    return Gf.Matrix4d(*row_matrix.reshape(-1).tolist())


try:
    source_hash_before = _sha256(args.base_scene)
    stage = Usd.Stage.Open(str(args.base_scene))
    print(f"open_stage returned: {stage is not None}", flush=True)
    if stage is None:
        raise RuntimeError(f"USD stage did not open: {args.base_scene}")
    stage_up_axis = str(UsdGeom.GetStageUpAxis(stage))
    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    print(
        "stage metadata: "
        f"up_axis={stage_up_axis}, meters_per_unit={stage_meters_per_unit:g}",
        flush=True,
    )

    # Author only into the opened root layer in memory.  That layer is exported
    # to a new filename and is deliberately never saved back to the source.
    stage.SetEditTarget(stage.GetRootLayer())
    target_prim = stage.GetPrimAtPath(args.target_prim)
    if not target_prim.IsValid() or not target_prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"target must be an existing Xformable: {args.target_prim}")
    target_xform = UsdGeom.Xformable(target_prim)
    if target_xform.TransformMightBeTimeVarying():
        raise RuntimeError("target transform is time-varying; refusing a default-only edit")

    time = Usd.TimeCode.Default()
    parent_prim = target_prim.GetParent()
    parent_world = _gf_to_column_transform(
        UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(time)
    )
    target_world_before = _gf_to_column_transform(
        target_xform.ComputeLocalToWorldTransform(time)
    )
    target_world_expected = rotate_world_pose_about_own_origin(
        target_world_before,
        yaw_degrees=args.yaw_deg,
    )
    target_local_after = local_pose_from_world_poses(
        parent_world,
        target_world_expected,
    )

    reset_xform_stack = target_xform.GetResetXformStack()
    matrix_op = target_xform.MakeMatrixXform()
    if not matrix_op.Set(_column_transform_to_gf(target_local_after), time):
        raise RuntimeError("failed to author target transform matrix")
    target_xform.SetResetXformStack(reset_xform_stack)

    target_world_after = _gf_to_column_transform(
        target_xform.ComputeLocalToWorldTransform(time)
    )
    checks = {
        "target_origin_translation_unchanged": bool(
            np.allclose(
                target_world_after[:3, 3],
                target_world_before[:3, 3],
                atol=1e-8,
                rtol=0.0,
            )
        ),
        "target_world_transform_matches_requested_yaw": bool(
            np.allclose(
                target_world_after,
                target_world_expected,
                atol=1e-8,
                rtol=0.0,
            )
        ),
        "target_transform_remains_affine": bool(
            np.allclose(
                target_world_after[3],
                [0.0, 0.0, 0.0, 1.0],
                atol=1e-9,
                rtol=0.0,
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "target-yaw scene validation failed: "
            + json.dumps(checks, sort_keys=True)
        )

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        world_prim.SetCustomDataByKey(
            "panda_handover:scene_variant", "target-world-z-yaw"
        )
        world_prim.SetCustomDataByKey(
            "panda_handover:target_yaw_degrees", float(args.yaw_deg)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(args.output)):
        raise RuntimeError(f"failed to save rotated scene: {args.output}")
    source_hash_after = _sha256(args.base_scene)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source scene changed while saving rotated variant")

    saved_stage = Usd.Stage.Open(str(args.output))
    if saved_stage is None:
        raise RuntimeError(f"saved USD stage did not reopen: {args.output}")
    saved_target = saved_stage.GetPrimAtPath(args.target_prim)
    if not saved_target.IsValid():
        raise RuntimeError("saved USD is missing the target prim")
    saved_target_world = _gf_to_column_transform(
        UsdGeom.Xformable(saved_target).ComputeLocalToWorldTransform(time)
    )
    checks["saved_scene_reopens_with_expected_target_transform"] = bool(
        np.allclose(
            saved_target_world,
            target_world_expected,
            atol=1e-8,
            rtol=0.0,
        )
    )
    checks["source_scene_hash_unchanged"] = source_hash_after == source_hash_before
    if not all(checks.values()):
        raise RuntimeError(
            "saved target-yaw scene validation failed: "
            + json.dumps(checks, sort_keys=True)
        )

    report = {
        "status": "success",
        "purpose": "new scene variant with only the target world-Z yaw changed",
        "source_scene": str(args.base_scene),
        "source_scene_sha256": source_hash_before,
        "scene_usd": str(args.output),
        "source_stage_metadata": {
            "up_axis": stage_up_axis,
            "meters_per_unit": stage_meters_per_unit,
            "used_to_choose_rotation_axis": False,
        },
        "authored": {
            "target_prim": args.target_prim,
            "yaw_axis": "world +Z",
            "yaw_degrees": float(args.yaw_deg),
            "pivot": "target world-space origin",
            "T_world_target_before": target_world_before.tolist(),
            "T_world_target_after": saved_target_world.tolist(),
        },
        "automatic_checks": checks,
        "source_scene_modified": False,
        "candidate_or_planner_parameters_changed": False,
        "physics_or_robot_parameters_changed": False,
        "manual_review_required": True,
        "next_gate": (
            "Open the new scene in the capture view and confirm that the intended "
            "mug side faces the camera before running a fresh pipeline output."
        ),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".check.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved: {args.output}", flush=True)
    print(f"saved: {report_path}", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
