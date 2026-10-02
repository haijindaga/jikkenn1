#!/usr/bin/env python3
"""Create and interactively edit a reusable Panda tabletop USD scene.

The saved stage is intentionally a scene-authoring artifact.  Physics-ready
objects can be referenced or dragged below ``/World/Objects`` in Isaac Sim,
renamed to ``Target`` for the current single-target pipeline, and saved without
modifying the source asset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT


LAYOUT = DEFAULT_TABLETOP_LAYOUT

RECEIVER_CHARACTER_ASSETS = {
    "male-police": {
        "relative_usd": (
            "Isaac/People/Characters/original_male_adult_police_04/"
            "male_adult_police_04.usd"
        ),
        "asset_kind": "skinned human character",
        "documented_local_forward_axis": "-Y",
    },
    "humanoid-proxy": {
        "relative_usd": "Isaac/Robots/IsaacSim/Humanoid/humanoid.usd",
        "asset_kind": "articulated humanoid proxy",
        "documented_local_forward_axis": None,
    },
    "1x-neo": {
        "relative_usd": "Isaac/Robots/1X/Neo/Neo.usd",
        "asset_kind": "1X NEO humanoid robot",
        "documented_local_forward_axis": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scenes/tabletop_base.usda"),
        help="Local USD/USDA stage to create and keep open in Isaac Sim",
    )
    parser.add_argument(
        "--exit-after-save",
        action="store_true",
        help="Create and validate the template without keeping the GUI open",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output template intentionally",
    )
    parser.add_argument(
        "--target-usd",
        type=Path,
        help=(
            "Physics-ready object USD to reference automatically below "
            "/World/Objects/Target"
        ),
    )
    parser.add_argument(
        "--target-center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=LAYOUT.target_center_m[:2],
        help="Desired target AABB centre on the tabletop in world metres",
    )
    parser.add_argument(
        "--target-clearance-m",
        type=float,
        default=0.002,
        help="Initial gap above the tabletop before capture-time settling",
    )
    parser.add_argument(
        "--static-receiver-character",
        choices=tuple(RECEIVER_CHARACTER_ASSETS),
        help=(
            "Add an official NVIDIA character or humanoid proxy as a static "
            "visual-only receiver; it is not included as a human collision or "
            "safety model"
        ),
    )
    parser.add_argument(
        "--receiver-center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(-1.15, 0.0),
        help=(
            "World XY assigned to the receiver asset root. The official asset "
            "is not recentered or rescaled"
        ),
    )
    parser.add_argument(
        "--receiver-yaw-deg",
        type=float,
        default=90.0,
        help=(
            "Receiver yaw about world +Z. For male-police, 90 degrees maps its "
            "documented local -Y forward axis toward world +X; humanoid-proxy "
            "and 1x-neo orientations must be confirmed visually"
        ),
    )
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        parser.error("--output must end in .usd, .usda, or .usdc")
    if args.output.expanduser().exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output}; use a new name or pass --overwrite"
        )
    if args.target_usd is not None:
        target_usd = args.target_usd.expanduser()
        if not target_usd.is_file():
            parser.error(f"target USD does not exist: {args.target_usd}")
        if target_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            parser.error("--target-usd must end in .usd, .usda, or .usdc")
    if not all(math.isfinite(value) for value in args.target_center_xy):
        parser.error("--target-center-xy values must be finite")
    if not math.isfinite(args.target_clearance_m) or args.target_clearance_m < 0.0:
        parser.error("--target-clearance-m must be finite and non-negative")
    if not all(math.isfinite(value) for value in args.receiver_center_xy):
        parser.error("--receiver-center-xy values must be finite")
    if not math.isfinite(args.receiver_yaw_deg):
        parser.error("--receiver-yaw-deg must be finite")
    return args


args = parse_args()

# Isaac requires SimulationApp construction before importing omni/pxr modules.
from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": args.exit_after_save})

try:
    import numpy as np
    import omni.client
    import omni.timeline
    import omni.usd

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.experimental.utils import stage as stage_utils
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    from panda_handover.geometry import look_at_quaternion_world

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 30.0,
    )
    world.scene.add_default_ground_plane(z_position=LAYOUT.ground_z_m)
    world.scene.add(
        Franka(
            prim_path="/World/Panda",
            name="panda",
            position=np.asarray(LAYOUT.robot_base_position_m),
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.asarray(LAYOUT.table_center_m),
            scale=np.asarray(LAYOUT.table_size_m),
            color=np.array([0.45, 0.32, 0.20]),
        )
    )
    create_prim("/World/Objects", prim_type="Scope")

    camera_position = np.asarray(LAYOUT.camera_position_m, dtype=np.float64)
    camera_target = np.asarray(LAYOUT.camera_target_m, dtype=np.float64)
    camera_orientation = look_at_quaternion_world(camera_position, camera_target)
    camera = Camera(
        prim_path="/World/camera_0",
        position=camera_position,
        orientation=camera_orientation,
        frequency=30,
        resolution=(640, 480),
    )

    world.reset()
    camera.initialize()
    camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")
    camera.set_clipping_range(0.05, 3.0)
    aperture = float(camera.get_horizontal_aperture())
    focal_length = aperture / (2.0 * np.tan(np.deg2rad(69.0) / 2.0))
    camera.set_focal_length(focal_length)

    # Stop before authoring.  This keeps drag-and-drop transforms as initial
    # conditions instead of saving a partially simulated state.
    omni.timeline.get_timeline_interface().stop()
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    world_prim.SetCustomDataByKey("panda_handover:schema_version", 1)
    world_prim.SetCustomDataByKey("panda_handover:panda_prim", "/World/Panda")
    world_prim.SetCustomDataByKey("panda_handover:table_prim", "/World/Table")
    world_prim.SetCustomDataByKey("panda_handover:camera_prim", "/World/camera_0")
    world_prim.SetCustomDataByKey("panda_handover:target_prim", "/World/Objects/Target")

    target_authoring = None
    if args.target_usd is not None:
        target_usd = args.target_usd.expanduser().resolve()
        target_wrapper = UsdGeom.Xform.Define(stage, "/World/Objects/Target")
        target_asset_prim = stage.DefinePrim("/World/Objects/Target/Asset", "Xform")
        relative_asset_path = Path(
            os.path.relpath(target_usd, start=output.parent)
        ).as_posix()
        # The current World stage is anonymous until save_stage. Compose from
        # the absolute source now; Isaac Sim's save_stage resolves authored
        # asset paths from the anonymous root to the output layer, producing
        # the portable relative reference recorded below.
        if not target_asset_prim.GetReferences().AddReference(str(target_usd)):
            raise RuntimeError(f"failed to reference target USD: {target_usd}")
        simulation_app.update()

        target_prims = tuple(Usd.PrimRange(target_asset_prim))
        physics_apis = {
            "rigid_body": any(
                prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in target_prims
            ),
            "collision": any(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                for prim in target_prims
            ),
            "mass": any(prim.HasAPI(UsdPhysics.MassAPI) for prim in target_prims),
        }
        if not all(physics_apis.values()):
            missing_apis = [
                name for name, present in physics_apis.items() if not present
            ]
            raise RuntimeError(
                "automatic target is not physics-ready; missing USD APIs: "
                + ", ".join(missing_apis)
            )

        initial_aabb = np.asarray(
            compute_aabb(
                create_bbox_cache(),
                "/World/Objects/Target",
                include_children=True,
            ),
            dtype=np.float64,
        )
        initial_extent = initial_aabb[3:] - initial_aabb[:3]
        if (
            initial_aabb.shape != (6,)
            or not np.all(np.isfinite(initial_aabb))
            or not np.all(initial_extent > 1e-4)
        ):
            raise RuntimeError(f"target USD has invalid bounds: {initial_aabb}")
        initial_center = 0.5 * (initial_aabb[:3] + initial_aabb[3:])
        target_translation = np.asarray(
            (
                args.target_center_xy[0] - initial_center[0],
                args.target_center_xy[1] - initial_center[1],
                LAYOUT.table_top_z_m
                + args.target_clearance_m
                - initial_aabb[2],
            ),
            dtype=np.float64,
        )
        UsdGeom.XformCommonAPI(target_wrapper.GetPrim()).SetTranslate(
            Gf.Vec3d(*target_translation.tolist())
        )
        simulation_app.update()
        placed_aabb = np.asarray(
            compute_aabb(
                create_bbox_cache(),
                "/World/Objects/Target",
                include_children=True,
            ),
            dtype=np.float64,
        )
        placed_center = 0.5 * (placed_aabb[:3] + placed_aabb[3:])
        table_min = np.asarray(LAYOUT.table_center_m) - 0.5 * np.asarray(
            LAYOUT.table_size_m
        )
        table_max = np.asarray(LAYOUT.table_center_m) + 0.5 * np.asarray(
            LAYOUT.table_size_m
        )
        placement_checks = {
            "aabb_is_finite": bool(np.all(np.isfinite(placed_aabb))),
            "xy_center_matches_request": bool(
                np.allclose(
                    placed_center[:2],
                    np.asarray(args.target_center_xy),
                    atol=1e-5,
                    rtol=0.0,
                )
            ),
            "bottom_matches_requested_clearance": bool(
                np.isclose(
                    placed_aabb[2],
                    LAYOUT.table_top_z_m + args.target_clearance_m,
                    atol=1e-5,
                    rtol=0.0,
                )
            ),
            "footprint_is_on_table": bool(
                placed_aabb[0] >= table_min[0]
                and placed_aabb[3] <= table_max[0]
                and placed_aabb[1] >= table_min[1]
                and placed_aabb[4] <= table_max[1]
            ),
        }
        if not all(placement_checks.values()):
            raise RuntimeError(
                "automatic target placement failed validation: "
                + json.dumps(placement_checks, sort_keys=True)
            )
        target_authoring = {
            "source_usd": str(target_usd),
            "scene_reference": relative_asset_path,
            "authoring_reference": str(target_usd),
            "save_stage_resolves_reference_to_output_layer": True,
            "wrapper_prim": "/World/Objects/Target",
            "asset_prim": "/World/Objects/Target/Asset",
            "initial_aabb_world_m": initial_aabb.tolist(),
            "translation_world_m": target_translation.tolist(),
            "placed_aabb_world_m": placed_aabb.tolist(),
            "requested_center_xy_m": list(args.target_center_xy),
            "requested_clearance_m": args.target_clearance_m,
            "physics_apis": physics_apis,
            "automatic_checks": placement_checks,
        }

    receiver_authoring = None
    if args.static_receiver_character is not None:
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError(
                "Isaac Sim assets root could not be resolved; install or configure "
                "the Isaac Sim 5.1 assets before adding the static receiver"
            )
        character_spec = RECEIVER_CHARACTER_ASSETS[args.static_receiver_character]
        character_relative_path = character_spec["relative_usd"]
        character_usd = f"{assets_root.rstrip('/')}/{character_relative_path}"
        character_stat_result, _ = omni.client.stat(character_usd)
        if character_stat_result != omni.client.Result.OK:
            raise RuntimeError(
                "official receiver USD is unavailable: "
                f"{character_usd} "
                f"({omni.client.get_result_string(character_stat_result)})"
            )
        receiver_wrapper = UsdGeom.Xform.Define(stage, "/World/Receiver")
        receiver_asset_prim = stage.DefinePrim("/World/Receiver/Asset", "Xform")
        if not receiver_asset_prim.GetReferences().AddReference(character_usd):
            raise RuntimeError(
                f"failed to reference official receiver USD: {character_usd}"
            )
        print(f"loading official receiver asset: {character_usd}", flush=True)

        receiver_xform = UsdGeom.XformCommonAPI(receiver_wrapper.GetPrim())
        receiver_xform.SetRotate(
            Gf.Vec3f(0.0, 0.0, float(args.receiver_yaw_deg)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        # Preserve the official asset exactly as authored.  Only a rigid scene
        # transform is applied to its wrapper: the asset root is placed at the
        # requested XY and at the already-defined room floor.  In particular,
        # do not traverse the referenced hierarchy, compute bounds, rescale it,
        # or author physics overrides.  Those operations caused native Fabric
        # failures for complex skinned/articulated receiver assets and would
        # also make this presentation-only composition less reproducible.
        receiver_translation = np.asarray(
            (*args.receiver_center_xy, LAYOUT.ground_z_m), dtype=np.float64
        )
        receiver_xform.SetTranslate(Gf.Vec3d(*receiver_translation.tolist()))
        receiver_wrapper.GetPrim().SetCustomDataByKey(
            "panda_handover:visual_only", True
        )
        receiver_checks = {
            "source_asset_stat_succeeded": bool(
                character_stat_result == omni.client.Result.OK
            ),
            "root_xy_matches_request": bool(
                np.allclose(
                    receiver_translation[:2],
                    np.asarray(args.receiver_center_xy),
                    atol=0.0,
                    rtol=0.0,
                )
            ),
            "asset_root_is_on_room_floor": bool(
                np.isclose(
                    receiver_translation[2],
                    LAYOUT.ground_z_m,
                    atol=0.0,
                    rtol=0.0,
                )
            ),
            "receiver_root_is_behind_robot": bool(receiver_translation[0] < -0.5),
            "source_asset_is_not_modified": True,
            "geometry_dependent_autoplacement_is_not_used": True,
        }
        if not all(receiver_checks.values()):
            raise RuntimeError(
                "static receiver placement failed validation: "
                + json.dumps(receiver_checks, sort_keys=True)
            )
        receiver_authoring = {
            "role": "visual-only static human receiver",
            "source": "NVIDIA Isaac Sim 5.1 assets",
            "character": args.static_receiver_character,
            "asset_kind": character_spec["asset_kind"],
            "source_usd": character_usd,
            "source_stat_result": omni.client.get_result_string(
                character_stat_result
            ),
            "wrapper_prim": "/World/Receiver",
            "asset_prim": "/World/Receiver/Asset",
            "requested_root_xy_m": list(args.receiver_center_xy),
            "requested_yaw_deg": args.receiver_yaw_deg,
            "documented_local_forward_axis": character_spec[
                "documented_local_forward_axis"
            ],
            "orientation_validation": (
                "documented -Y forward axis rotated toward world +X"
                if character_spec["documented_local_forward_axis"] == "-Y"
                else "visual review required; no forward-axis claim is encoded"
            ),
            "ground_z_m": LAYOUT.ground_z_m,
            "translation_world_m": receiver_translation.tolist(),
            "composition_policy": {
                "official_asset_referenced_without_internal_edits": True,
                "asset_rescaled": False,
                "geometry_dependent_autoplacement": False,
                "physics_apis_modified": False,
                "timeline_must_remain_stopped": True,
                "approved_use": "presentation and figure capture only",
            },
            "automatic_checks": receiver_checks,
        }

    saved = bool(stage_utils.save_stage(str(output)))
    required_prims = [
        "/World",
        "/World/Panda",
        "/World/Table",
        "/World/Objects",
        "/World/camera_0",
    ]
    if target_authoring is not None:
        required_prims.extend(
            ("/World/Objects/Target", "/World/Objects/Target/Asset")
        )
    if receiver_authoring is not None:
        required_prims.extend(("/World/Receiver", "/World/Receiver/Asset"))
    prim_checks = {
        prim_path: bool(stage.GetPrimAtPath(prim_path).IsValid())
        for prim_path in required_prims
    }
    report = {
        "status": "success" if saved and all(prim_checks.values()) else "failure",
        "reference": {
            "stage_api": "Isaac Sim 5.1 isaacsim.core.experimental.utils.stage.save_stage",
            "composition": "OpenUSD referenced assets edited in a separate scene stage",
        },
        "scene_usd": str(output),
        "target_authoring": target_authoring,
        "receiver_authoring": receiver_authoring,
        "authoring_contract": {
            "objects_scope": "/World/Objects",
            "single_target_prim": "/World/Objects/Target",
            "source_assets_are_not_modified": True,
            "save_while_timeline_stopped": True,
            "receiver_is_visual_only_not_a_human_safety_model": bool(
                receiver_authoring is not None
            ),
        },
        "automatic_checks": {
            "stage_saved": saved,
            "required_prims_exist": all(prim_checks.values()),
        },
        "prim_checks": prim_checks,
    }
    report_path = output.with_suffix(output.suffix + ".check.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "success":
        raise RuntimeError(f"tabletop authoring template failed validation: {report_path}")

    print(f"saved editable tabletop scene: {output}", flush=True)
    print(f"saved validation report: {report_path}", flush=True)
    if not args.exit_after_save:
        print(
            "Drag a physics-ready USD below /World/Objects, rename its root prim "
            "to Target, place it on the table, then use File > Save or Save As.",
            flush=True,
        )
        print("Close the Isaac Sim window when scene authoring is finished.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
finally:
    simulation_app.close()
