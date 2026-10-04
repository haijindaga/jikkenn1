#!/usr/bin/env python3
"""Create a diagnostic scene supported below a non-grasp object part.

The source scene is copied to a new USD. A fixed cuboid is placed below a
support point derived from the saved receive-part segmentation, while the
complete target is translated upward. The support footprint must remain
outside the saved grasp-part projection. Historical hammer option names remain
accepted for reproducibility.
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
    parser.add_argument("--reference-capture", type=Path, required=True)
    parser.add_argument("--reference-segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument("--table-prim", default="/World/Table")
    parser.add_argument("--support-prim", default="/World/HandleSupport")
    parser.add_argument(
        "--target-clearance-m",
        "--head-clearance-m",
        dest="target_clearance_m",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--support-width-m",
        type=float,
        default=0.03,
        help="rail width, or maximum square width for part-mask-patch",
    )
    parser.add_argument(
        "--grasp-part-projection-margin-m",
        "--head-projection-margin-m",
        dest="grasp_part_projection_margin_m",
        type=float,
        default=0.005,
    )
    parser.add_argument("--object-label", default="hammer")
    parser.add_argument("--grasp-part-label", default="hammer head")
    parser.add_argument("--support-part-label", default="hammer handle")
    parser.add_argument(
        "--grasp-mask-role",
        choices=("grasp_part", "receive_part"),
        default="grasp_part",
        help="saved SAM3 part role used as the grasp-clear region",
    )
    parser.add_argument(
        "--support-mask-role",
        choices=("grasp_part", "receive_part"),
        default="receive_part",
        help="saved SAM3 part role used to locate the support fixture",
    )
    parser.add_argument(
        "--support-geometry-policy",
        choices=("handle-axis-rail", "part-mask-patch"),
        default="handle-axis-rail",
        help="geometry rule used to place support inside the saved support part",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_scene = args.base_scene.expanduser().resolve()
    args.reference_capture = args.reference_capture.expanduser().resolve()
    args.reference_segmentation = args.reference_segmentation.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.base_scene.is_file():
        parser.error(f"base scene does not exist: {args.base_scene}")
    if args.output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        parser.error("--output must end in .usd, .usda, or .usdc")
    if args.output == args.base_scene:
        parser.error("--output must differ from --base-scene")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite intentionally")
    for value, label in (
        (args.target_clearance_m, "--target-clearance-m"),
        (args.support_width_m, "--support-width-m"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{label} must be positive and finite")
    if (
        not math.isfinite(args.grasp_part_projection_margin_m)
        or args.grasp_part_projection_margin_m < 0.0
    ):
        parser.error(
            "--grasp-part-projection-margin-m must be finite and non-negative"
        )
    for value, label in (
        (args.object_label, "--object-label"),
        (args.grasp_part_label, "--grasp-part-label"),
        (args.support_part_label, "--support-part-label"),
    ):
        if not value.strip():
            parser.error(f"{label} must not be empty")
    if args.grasp_mask_role == args.support_mask_role:
        parser.error("--grasp-mask-role and --support-mask-role must differ")
    for value, label in (
        (args.target_prim, "--target-prim"),
        (args.table_prim, "--table-prim"),
        (args.support_prim, "--support-prim"),
    ):
        if not value.startswith("/") or value == "/":
            parser.error(f"{label} must be an absolute non-root prim path")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


args = parse_args()

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.utils import stage as stage_utils
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from panda_handover.support_fixture import (
        masked_points_world,
        plan_handle_support,
        plan_part_support_patch,
    )

    required_capture_files = (
        "points_camera.npy",
        "T_world_camera.npy",
    )
    for name in required_capture_files:
        if not (args.reference_capture / name).is_file():
            raise FileNotFoundError(args.reference_capture / name)
    head_mask_path = (
        args.reference_segmentation
        / "parts"
        / args.grasp_mask_role
        / "union_mask.npy"
    )
    handle_mask_path = (
        args.reference_segmentation
        / "parts"
        / args.support_mask_role
        / "union_mask.npy"
    )
    for path in (head_mask_path, handle_mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    points_camera = np.load(
        args.reference_capture / "points_camera.npy", allow_pickle=False
    )
    T_world_camera = np.load(
        args.reference_capture / "T_world_camera.npy", allow_pickle=False
    )
    grasp_points_world = masked_points_world(
        points_camera,
        np.load(head_mask_path, allow_pickle=False),
        T_world_camera,
    )
    support_points_world = masked_points_world(
        points_camera,
        np.load(handle_mask_path, allow_pickle=False),
        T_world_camera,
    )
    if args.support_geometry_policy == "handle-axis-rail":
        support_plan = plan_handle_support(
            grasp_points_world,
            support_points_world,
            support_width_m=args.support_width_m,
            head_projection_margin_m=args.grasp_part_projection_margin_m,
            target_lift_m=args.target_clearance_m,
        )
    else:
        support_plan = plan_part_support_patch(
            grasp_points_world,
            support_points_world,
            support_width_m=args.support_width_m,
            grasp_projection_margin_m=args.grasp_part_projection_margin_m,
            target_lift_m=args.target_clearance_m,
        )

    base_hash_before = _sha256(args.base_scene)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    result = context.open_stage(str(args.base_scene))
    print(f"open_stage returned: {result}", flush=True)
    for _ in range(120):
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"USD stage did not open: {args.base_scene}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("diagnostic support scene requires a Z-up stage")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise RuntimeError("diagnostic support scene requires metre stage units")

    target_prim = stage.GetPrimAtPath(args.target_prim)
    table_prim = stage.GetPrimAtPath(args.table_prim)
    if not target_prim.IsValid() or not target_prim.IsA(UsdGeom.Xform):
        raise RuntimeError(f"target must be an existing Xform: {args.target_prim}")
    if not table_prim.IsValid():
        raise RuntimeError(f"table prim does not exist: {args.table_prim}")
    if stage.GetPrimAtPath(args.support_prim).IsValid():
        raise RuntimeError(f"support prim already exists: {args.support_prim}")

    parent_prim = target_prim.GetParent()
    parent_world = np.asarray(
        UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ),
        dtype=np.float64,
    )
    if not np.allclose(parent_world, np.eye(4), atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "target parent is transformed; refusing to treat world-Z lift as local-Z"
        )

    bbox_cache = create_bbox_cache()
    table_aabb = np.asarray(
        compute_aabb(bbox_cache, args.table_prim, include_children=True),
        dtype=np.float64,
    )
    target_aabb_before = np.asarray(
        compute_aabb(bbox_cache, args.target_prim, include_children=True),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(table_aabb)) or not np.all(np.isfinite(target_aabb_before)):
        raise RuntimeError("table or target has invalid bounds")
    table_top_z_m = float(table_aabb[5])

    target_xform = UsdGeom.XformCommonAPI(target_prim)
    translation, _, _, _, _ = target_xform.GetXformVectors(Usd.TimeCode.Default())
    target_translation_before = np.asarray(translation, dtype=np.float64)
    target_translation_after = target_translation_before.copy()
    target_translation_after[2] += args.target_clearance_m
    if not target_xform.SetTranslate(Gf.Vec3d(*target_translation_after.tolist())):
        raise RuntimeError("failed to translate target wrapper")

    support_height_m = args.target_clearance_m
    support_center = (
        support_plan.support_center_xy_world_m[0],
        support_plan.support_center_xy_world_m[1],
        table_top_z_m + 0.5 * support_height_m,
    )
    support = UsdGeom.Cube.Define(stage, args.support_prim)
    support.CreateSizeAttr(1.0)
    support.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.65, 0.95)])
    support_xform = UsdGeom.XformCommonAPI(support.GetPrim())
    support_xform.SetTranslate(Gf.Vec3d(*support_center))
    support_xform.SetScale(
        Gf.Vec3f(
            support_plan.support_length_m,
            support_plan.support_width_m,
            support_height_m,
        )
    )
    support_xform.SetRotate(
        Gf.Vec3f(0.0, 0.0, support_plan.support_yaw_deg),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    world_prim = stage.GetPrimAtPath("/World")
    world_prim.SetCustomDataByKey(
        "panda_handover:diagnostic_scene", "support-part-supported-grasp-clear"
    )
    world_prim.SetCustomDataByKey(
        "panda_handover:diagnostic_support_prim", args.support_prim
    )
    simulation_app.update()

    bbox_cache = create_bbox_cache()
    target_aabb_after = np.asarray(
        compute_aabb(bbox_cache, args.target_prim, include_children=True),
        dtype=np.float64,
    )
    support_aabb = np.asarray(
        compute_aabb(bbox_cache, args.support_prim, include_children=True),
        dtype=np.float64,
    )
    checks = {
        "target_raised_by_requested_amount": bool(
            np.isclose(
                target_aabb_after[2] - target_aabb_before[2],
                args.target_clearance_m,
                atol=1e-5,
                rtol=0.0,
            )
        ),
        "support_starts_at_tabletop": bool(
            np.isclose(support_aabb[2], table_top_z_m, atol=1e-5, rtol=0.0)
        ),
        "support_top_matches_target_lift": bool(
            np.isclose(
                support_aabb[5],
                table_top_z_m + args.target_clearance_m,
                atol=1e-5,
                rtol=0.0,
            )
        ),
        "support_footprint_excludes_saved_head_projection": bool(
            support_plan.minimum_head_to_support_footprint_distance_m
            > args.grasp_part_projection_margin_m
        ),
        "support_is_static_collision_geometry": bool(
            support.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            and not support.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "diagnostic support scene validation failed: "
            + json.dumps(checks, sort_keys=True)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not stage_utils.save_stage(str(args.output)):
        raise RuntimeError(f"failed to save diagnostic scene: {args.output}")
    base_hash_after = _sha256(args.base_scene)
    if base_hash_after != base_hash_before:
        raise RuntimeError("source scene changed while saving diagnostic variant")

    report = {
        "status": "success",
        "purpose": (
            "controlled simulation-only test of whether tabletop clearance "
            f"limits {args.grasp_part_label} grasp insertion"
        ),
        "source_scene": str(args.base_scene),
        "source_scene_sha256": base_hash_before,
        "scene_usd": str(args.output),
        "reference_capture": str(args.reference_capture),
        "reference_segmentation": str(args.reference_segmentation),
        "part_roles": {
            "object": args.object_label,
            "grasp_part": args.grasp_part_label,
            "grasp_mask_role": args.grasp_mask_role,
            "support_part": (
                f"{args.support_part_label} used only to locate support"
            ),
            "support_mask_role": args.support_mask_role,
            "support_geometry_policy": args.support_geometry_policy,
            "geometry_implementation_names": {
                "head_points": "grasp_part",
                "handle_points": "support_part",
            },
        },
        "support_plan": support_plan.as_dict(),
        "authored": {
            "target_prim": args.target_prim,
            "target_translation_before_m": target_translation_before.tolist(),
            "target_translation_after_m": target_translation_after.tolist(),
            "target_aabb_before_m": target_aabb_before.tolist(),
            "target_aabb_after_m": target_aabb_after.tolist(),
            "table_top_z_m": table_top_z_m,
            "support_prim": args.support_prim,
            "support_center_world_m": list(support_center),
            "support_size_m": [
                support_plan.support_length_m,
                support_plan.support_width_m,
                support_height_m,
            ],
            "requested_support_width_m": args.support_width_m,
            "support_yaw_deg": support_plan.support_yaw_deg,
            "support_aabb_world_m": support_aabb.tolist(),
        },
        "automatic_checks": checks,
        "source_scene_modified": False,
        "candidate_or_planner_parameters_changed": False,
        "manual_review_required": True,
        "next_gate": (
            "Open the scene without playing physics, confirm the block is below "
            f"the {args.support_part_label} and not the {args.grasp_part_label}, "
            "then run a fresh capture and pipeline. The post-settle grasp-part "
            "clearance must be visually reviewed."
        ),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".check.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved: {args.output}", flush=True)
    print(f"saved: {report_path}", flush=True)
except BaseException:
    # Isaac Sim shutdown can hide a pending Python exception on some builds.
    # Emit it before closing so a failed diagnostic never looks successful.
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
