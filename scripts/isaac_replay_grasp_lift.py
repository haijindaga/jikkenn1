#!/usr/bin/env python3
"""Physically close the Panda gripper and replay a cuRobo grasp/lift in Isaac Sim."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import sys
import traceback

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT
from panda_handover.physics_baselines import (
    FINGER_DRIVE_PRESETS,
    drive_value_matches_float_storage,
    resolve_finger_drive_values,
)
from panda_handover.trajectory_replay import (
    load_grasp_lift_replay,
    sample_positions_at_physics_rate,
)


LAYOUT = DEFAULT_TABLETOP_LAYOUT
PHYSICS_DT_S = 1.0 / 60.0
PANDA_OPEN_FINGER_JOINT_M = 0.04
PANDA_CLOSED_FINGER_JOINT_M = 0.0
ATTACHMENT_TRANSLATION_TOLERANCE_M = 0.005
ATTACHMENT_ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
FIXED_JOINT_MIN_FINGER_CLOSURE_TRAVEL_M = 0.001
FIXED_JOINT_POST_CREATE_SETTLE_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-usd",
        type=Path,
        help="Open the authored USD used by capture instead of the legacy block scene",
    )
    parser.add_argument("--panda-prim", default="/World/Panda")
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument("--camera-prim", default="/World/camera_0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--close-frames", type=int, default=60)
    parser.add_argument("--hold-frames", type=int, default=180)
    parser.add_argument(
        "--open-finger-position-m",
        type=float,
        default=PANDA_OPEN_FINGER_JOINT_M,
    )
    parser.add_argument(
        "--closed-finger-position-m",
        type=float,
        default=PANDA_CLOSED_FINGER_JOINT_M,
    )
    parser.add_argument(
        "--finger-drive-preset",
        choices=tuple(FINGER_DRIVE_PRESETS),
        default="authored-usd",
        help=(
            "Named Panda finger-drive condition. 'authored-usd' preserves the "
            "loaded USD; 'isaaclab-franka' applies the source-backed Isaac Lab "
            "Franka simulator actuator values."
        ),
    )
    parser.add_argument(
        "--finger-drive-max-force-n",
        type=float,
        help=(
            "Optional OpenUSD linear DriveAPI max-force value for the actuated "
            "Panda finger joint. This is recorded as a simulation drive limit and "
            "is not claimed to equal calibrated total hardware grasp force."
        ),
    )
    parser.add_argument(
        "--finger-drive-scale",
        type=float,
        default=1.0,
        help=(
            "Simulation-only diagnostic scale: multiply max force and stiffness "
            "by this value and damping by its square root. This is not a "
            "hardware-calibrated grasp-force command."
        ),
    )
    parser.add_argument(
        "--fingertip-friction-coefficient",
        type=float,
        help=(
            "Simulation-only diagnostic override applied to both static and "
            "dynamic friction on the Panda fingertip collision geometry. The "
            "runtime material uses PhysX friction combine mode 'max', so the "
            "target and table materials remain unchanged."
        ),
    )
    parser.add_argument(
        "--grasp-retention-mode",
        choices=(
            "physics",
            "physx-auto-attachment",
            "rigid-attachment",
            "surface-gripper-attachment",
            "kinematic-pose-lock",
        ),
        default="physics",
        help=(
            "'physics' evaluates contact-only retention. "
            "'physx-auto-attachment' applies the PhysX runtime attachment "
            "recipe to the current simulated poses. 'rigid-attachment' "
            "creates a post-close FixedJoint for legacy comparison. "
            "'surface-gripper-attachment' uses Isaac Sim 5.1's official "
            "Surface Gripper manager and bundled D6-joint template. "
            "'kinematic-pose-lock' preserves the measured post-close "
            "target-to-hand transform exactly during lift/transport."
        ),
    )
    parser.add_argument(
        "--solver-position-iterations",
        type=int,
        help=(
            "Simulation-only FixedJoint diagnostic: override the PhysX solver "
            "position iterations on both the Panda articulation and target rigid body"
        ),
    )
    parser.add_argument(
        "--solver-velocity-iterations",
        type=int,
        help=(
            "Simulation-only FixedJoint diagnostic: override the PhysX solver "
            "velocity iterations on both the Panda articulation and target rigid body"
        ),
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Required acknowledgement: this command controls only an Isaac Sim robot",
    )
    args = parser.parse_args()
    if not args.simulation_only:
        parser.error("--simulation-only is required")
    if min(args.settle_frames, args.close_frames, args.hold_frames) < 0:
        parser.error("frame counts must be non-negative")
    if not (
        0.0
        <= args.closed_finger_position_m
        <= args.open_finger_position_m
        <= PANDA_OPEN_FINGER_JOINT_M
    ):
        parser.error(
            "finger positions must satisfy 0 <= closed <= open <= 0.04 metres"
        )
    if args.finger_drive_max_force_n is not None and (
        not math.isfinite(args.finger_drive_max_force_n)
        or args.finger_drive_max_force_n <= 0.0
    ):
        parser.error("--finger-drive-max-force-n must be positive and finite")
    if not math.isfinite(args.finger_drive_scale) or args.finger_drive_scale <= 0.0:
        parser.error("--finger-drive-scale must be positive and finite")
    if args.finger_drive_scale != 1.0 and args.finger_drive_preset == "authored-usd":
        parser.error("--finger-drive-scale requires --finger-drive-preset isaaclab-franka")
    if args.finger_drive_scale != 1.0 and args.finger_drive_max_force_n is not None:
        parser.error(
            "--finger-drive-scale cannot be combined with --finger-drive-max-force-n"
        )
    if args.fingertip_friction_coefficient is not None and (
        not math.isfinite(args.fingertip_friction_coefficient)
        or args.fingertip_friction_coefficient < 0.0
    ):
        parser.error("--fingertip-friction-coefficient must be finite and non-negative")
    if (
        args.fingertip_friction_coefficient is not None
        and args.grasp_retention_mode != "physics"
    ):
        parser.error(
            "--fingertip-friction-coefficient is meaningful only with "
            "--grasp-retention-mode physics"
        )
    if (
        args.fingertip_friction_coefficient is not None
        and args.finger_drive_scale != 1.0
    ):
        parser.error(
            "do not combine fingertip friction and finger-drive diagnostics in "
            "one controlled run"
        )
    solver_iteration_values = (
        args.solver_position_iterations,
        args.solver_velocity_iterations,
    )
    if (solver_iteration_values[0] is None) != (solver_iteration_values[1] is None):
        parser.error(
            "--solver-position-iterations and --solver-velocity-iterations must "
            "be supplied together"
        )
    if args.solver_position_iterations is not None:
        if not 1 <= args.solver_position_iterations <= 255:
            parser.error("--solver-position-iterations must be in 1..255")
        if not 0 <= args.solver_velocity_iterations <= 255:
            parser.error("--solver-velocity-iterations must be in 0..255")
        if args.grasp_retention_mode != "rigid-attachment":
            parser.error(
                "solver-iteration diagnostics currently require "
                "--grasp-retention-mode rigid-attachment"
            )
    return args


args = parse_args()
finger_drive_preset = FINGER_DRIVE_PRESETS[args.finger_drive_preset]
requested_finger_drive_values = resolve_finger_drive_values(
    args.finger_drive_preset,
    explicit_max_force=args.finger_drive_max_force_n,
    diagnostic_scale=args.finger_drive_scale,
)
replay = load_grasp_lift_replay(args.capture, args.plan)
if (
    args.grasp_retention_mode
    in {
        "physx-auto-attachment",
        "rigid-attachment",
        "surface-gripper-attachment",
        "kinematic-pose-lock",
    }
    and "transport" not in replay.phase_positions
):
    raise ValueError(
        "non-physical retention modes require an attached collision-checked "
        "transport phase in the cuRobo plan"
    )
scene_usd = None
if args.scene_usd is not None:
    scene_usd = args.scene_usd.expanduser().resolve()
    if not scene_usd.is_file():
        raise FileNotFoundError(f"authored scene USD does not exist: {scene_usd}")
    if scene_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--scene-usd must end in .usd, .usda, or .usdc")

    scene_layout_path = args.capture / "scene_layout.json"
    if not scene_layout_path.is_file():
        raise FileNotFoundError(
            "authored-scene replay requires the capture scene report: "
            f"{scene_layout_path}"
        )
    scene_layout_report = json.loads(scene_layout_path.read_text(encoding="utf-8"))
    scene_source = scene_layout_report.get("scene_source", {})
    if scene_source.get("kind") != "authored_usd_scene":
        raise ValueError(
            "--scene-usd was provided, but the capture was not recorded from an "
            "authored USD scene"
        )
    recorded_scene_value = scene_source.get("scene_usd")
    if not recorded_scene_value:
        raise ValueError("capture scene report has no authored scene_usd path")
    recorded_scene = Path(recorded_scene_value).expanduser().resolve()
    if recorded_scene != scene_usd:
        raise ValueError(
            "replay scene does not match the scene recorded by capture: "
            f"{scene_usd} != {recorded_scene}"
        )

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

try:
    import numpy as np
    from PIL import Image

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import RigidPrim, SingleArticulation, XFormPrim
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.experimental.utils import stage as stage_utils
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera
    from omni.physx import get_physx_simulation_interface
    from pxr import (
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Sdf,
        Usd,
        UsdPhysics,
        UsdShade,
    )

    if args.grasp_retention_mode == "surface-gripper-attachment":
        import isaacsim.robot.surface_gripper as surface_gripper_package
        import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
        import usd.schema.isaac.robot_schema as robot_schema

    from panda_handover.geometry import (
        aabb_ray_origin_toward_center,
        look_at_quaternion_world,
        matrix_from_pose,
        quaternion_wxyz_from_rotation_matrix,
        relative_pose,
        rotation_matrix_align_axis_to_vector,
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    target_physics_apis = None
    target_rigid_prim_path = None
    if scene_usd is not None:
        stage_opened, stage = stage_utils.open_stage(str(scene_usd))
        if not stage_opened or stage is None:
            raise RuntimeError(f"Isaac Sim could not open authored scene: {scene_usd}")
        required_prim_paths = (
            args.panda_prim,
            args.target_prim,
            args.camera_prim,
        )
        missing_prims = [
            prim_path
            for prim_path in required_prim_paths
            if not stage.GetPrimAtPath(prim_path).IsValid()
        ]
        if missing_prims:
            raise RuntimeError(
                "authored scene is missing required prims: " + ", ".join(missing_prims)
            )

        target_root_prim = stage.GetPrimAtPath(args.target_prim)
        target_prims = tuple(Usd.PrimRange(target_root_prim))
        rigid_body_prims = tuple(
            prim for prim in target_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        target_physics_apis = {
            "rigid_body": bool(rigid_body_prims),
            "collision": any(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                for prim in target_prims
            ),
            "mass": any(prim.HasAPI(UsdPhysics.MassAPI) for prim in target_prims),
        }
        if not all(target_physics_apis.values()):
            missing_apis = [
                name for name, present in target_physics_apis.items() if not present
            ]
            raise RuntimeError(
                f"saved-scene target {args.target_prim} is not physics-ready; "
                f"missing USD APIs: {missing_apis}"
            )
        if len(rigid_body_prims) != 1:
            rigid_body_paths = [str(prim.GetPath()) for prim in rigid_body_prims]
            raise RuntimeError(
                "authored-scene grasp replay supports one rigid target body; "
                f"found {len(rigid_body_prims)} under {args.target_prim}: "
                f"{rigid_body_paths}"
            )
        target_rigid_prim_path = str(rigid_body_prims[0].GetPath())

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT_S,
        rendering_dt=1.0 / 30.0,
    )
    if scene_usd is None:
        world.scene.add_default_ground_plane(z_position=LAYOUT.ground_z_m)
        panda = world.scene.add(
            Franka(
                prim_path=args.panda_prim,
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
        target = world.scene.add(
            DynamicCuboid(
                prim_path="/World/TestObject",
                name="test_object",
                position=np.asarray(LAYOUT.target_center_m),
                scale=np.asarray(LAYOUT.target_size_m),
                color=np.array([0.1, 0.5, 0.9]),
            )
        )
        target_prim_path = "/World/TestObject"
        target_rigid_prim_path = target_prim_path
        target_physics_apis = {
            "rigid_body": True,
            "collision": True,
            "mass": True,
        }
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Obstacle",
                name="obstacle",
                position=np.asarray(LAYOUT.obstacle_center_m),
                scale=np.asarray(LAYOUT.obstacle_size_m),
                color=np.array([0.9, 0.2, 0.1]),
            )
        )
        camera_position = np.asarray(LAYOUT.camera_position_m, dtype=np.float64)
        camera_target = np.asarray(LAYOUT.camera_target_m, dtype=np.float64)
        camera_orientation = look_at_quaternion_world(camera_position, camera_target)
        camera = Camera(
            prim_path="/World/replay_camera",
            position=camera_position,
            orientation=camera_orientation,
            frequency=30,
            resolution=(640, 480),
        )
    else:
        panda = world.scene.add(
            SingleArticulation(
                prim_path=args.panda_prim,
                name="panda",
            )
        )
        target = world.scene.add(
            RigidPrim(
                prim_paths_expr=target_rigid_prim_path,
                name="target",
                reset_xform_properties=False,
            )
        )
        target_prim_path = args.target_prim
        camera = Camera(
            prim_path=args.camera_prim,
            frequency=30,
            resolution=(640, 480),
        )

    stage = stage_utils.get_current_stage()

    def unique_named_rigid_body_path(prim_name: str) -> str:
        root = stage.GetPrimAtPath(args.panda_prim)
        matching = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.GetName() == prim_name
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"expected one {prim_name} rigid body, found: "
                + str([str(prim.GetPath()) for prim in matching])
            )
        return str(matching[0].GetPath())

    fixed_joint_contact_setup = {
        "enabled": False,
        "source": "Isaac Sim PhysxContactReportAPI",
        "authored_before_world_reset": False,
        "target_rigid_body_prim": target_rigid_prim_path,
        "finger_rigid_body_prims": None,
        "threshold_before": None,
        "threshold_after": None,
    }
    fixed_joint_finger_paths: dict[str, str] = {}
    physx_simulation_interface = None
    if args.grasp_retention_mode == "rigid-attachment":
        fixed_joint_finger_paths = {
            "left": unique_named_rigid_body_path("panda_leftfinger"),
            "right": unique_named_rigid_body_path("panda_rightfinger"),
        }
        target_rigid_prim = stage.GetPrimAtPath(target_rigid_prim_path)
        contact_report_was_present = target_rigid_prim.HasAPI(
            PhysxSchema.PhysxContactReportAPI
        )
        contact_report_api = PhysxSchema.PhysxContactReportAPI.Get(
            stage, target_rigid_prim.GetPath()
        )
        if not contact_report_api or not contact_report_api.GetPrim().IsValid():
            contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(
                target_rigid_prim
            )
        if not contact_report_api or not contact_report_api.GetPrim().IsValid():
            raise RuntimeError(
                "could not apply PhysxContactReportAPI to the target before reset"
            )
        threshold_attribute = contact_report_api.GetThresholdAttr()
        threshold_before = (
            threshold_attribute.Get() if threshold_attribute else None
        )
        contact_report_api.CreateThresholdAttr().Set(0.0)
        threshold_after = contact_report_api.GetThresholdAttr().Get()
        if threshold_after is None or float(threshold_after) != 0.0:
            raise RuntimeError("target contact-report threshold was not set to zero")
        physx_simulation_interface = get_physx_simulation_interface()
        fixed_joint_contact_setup = {
            "enabled": True,
            "source": "Isaac Sim PhysxContactReportAPI",
            "authored_before_world_reset": True,
            "api_was_already_present": contact_report_was_present,
            "target_rigid_body_prim": target_rigid_prim_path,
            "finger_rigid_body_prims": fixed_joint_finger_paths,
            "threshold_before": (
                float(threshold_before) if threshold_before is not None else None
            ),
            "threshold_after": float(threshold_after),
        }

    solver_iteration_override = {
        "applied": False,
        "diagnostic_only": False,
        "panda_articulation_prim": None,
        "target_rigid_body_prim": target_rigid_prim_path,
        "requested": None,
        "before": None,
        "after": None,
    }
    if args.solver_position_iterations is not None:
        panda_root_prim = stage.GetPrimAtPath(args.panda_prim)
        articulation_root_prims = tuple(
            prim
            for prim in Usd.PrimRange(panda_root_prim)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if not articulation_root_prims:
            raise RuntimeError(
                f"Panda articulation root was not found below {args.panda_prim}"
            )
        panda_articulation_prim = next(
            (
                prim
                for prim in articulation_root_prims
                if str(prim.GetPath()) == args.panda_prim
            ),
            articulation_root_prims[0] if len(articulation_root_prims) == 1 else None,
        )
        if panda_articulation_prim is None:
            raise RuntimeError(
                "multiple Panda articulation roots were found: "
                f"{[str(prim.GetPath()) for prim in articulation_root_prims]}"
            )
        target_rigid_prim = stage.GetPrimAtPath(target_rigid_prim_path)
        if not target_rigid_prim.IsValid():
            raise RuntimeError(
                f"target rigid body prim does not exist: {target_rigid_prim_path}"
            )

        articulation_api = PhysxSchema.PhysxArticulationAPI.Get(
            stage, panda_articulation_prim.GetPath()
        )
        if not articulation_api or not articulation_api.GetPrim().IsValid():
            articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(
                panda_articulation_prim
            )
        target_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Get(
            stage, target_rigid_prim.GetPath()
        )
        if not target_rigid_api or not target_rigid_api.GetPrim().IsValid():
            target_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target_rigid_prim)

        def solver_iterations(api) -> dict:
            position = api.GetSolverPositionIterationCountAttr().Get()
            velocity = api.GetSolverVelocityIterationCountAttr().Get()
            return {
                "position": int(position) if position is not None else None,
                "velocity": int(velocity) if velocity is not None else None,
            }

        before_iterations = {
            "panda_articulation": solver_iterations(articulation_api),
            "target_rigid_body": solver_iterations(target_rigid_api),
        }
        for api in (articulation_api, target_rigid_api):
            api.CreateSolverPositionIterationCountAttr().Set(
                args.solver_position_iterations
            )
            api.CreateSolverVelocityIterationCountAttr().Set(
                args.solver_velocity_iterations
            )
        after_iterations = {
            "panda_articulation": solver_iterations(articulation_api),
            "target_rigid_body": solver_iterations(target_rigid_api),
        }
        requested_iterations = {
            "position": args.solver_position_iterations,
            "velocity": args.solver_velocity_iterations,
        }
        if any(
            values != requested_iterations for values in after_iterations.values()
        ):
            raise RuntimeError(
                "requested PhysX solver iteration overrides were not applied: "
                f"{after_iterations}"
            )
        solver_iteration_override = {
            "applied": True,
            "diagnostic_only": True,
            "panda_articulation_prim": str(panda_articulation_prim.GetPath()),
            "target_rigid_body_prim": target_rigid_prim_path,
            "requested": requested_iterations,
            "before": before_iterations,
            "after": after_iterations,
        }

    world.reset()
    camera.initialize()
    if scene_usd is None:
        camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")

    stage = stage_utils.get_current_stage()

    def usd_attribute_value(attribute):
        if not attribute:
            return None
        value = attribute.Get()
        if value is None:
            return None
        if isinstance(value, (bool, int, str)):
            return value
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return str(value)
        return scalar if np.isfinite(scalar) else str(scalar)

    def collision_materials_below(root_prim_path: str) -> list[dict]:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return []
        records = []
        # Franka collision meshes may live below instanceable USD prims. The
        # default PrimRange stops at instances, so include their read-only
        # instance proxies when resolving effective materials.
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if not (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            ):
                continue
            material, binding_relationship = (
                UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                    "physics"
                )
            )
            material_prim = material.GetPrim() if material else None
            material_api = (
                UsdPhysics.MaterialAPI(material_prim)
                if material_prim
                and material_prim.HasAPI(UsdPhysics.MaterialAPI)
                else None
            )
            physx_material_api = (
                PhysxSchema.PhysxMaterialAPI(material_prim)
                if material_prim
                and material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
                else None
            )
            records.append(
                {
                    "collision_prim": str(prim.GetPath()),
                    "physics_material": (
                        str(material_prim.GetPath()) if material_prim else None
                    ),
                    "binding_relationship": (
                        str(binding_relationship.GetPath())
                        if binding_relationship
                        else None
                    ),
                    "static_friction": (
                        usd_attribute_value(material_api.GetStaticFrictionAttr())
                        if material_api
                        else None
                    ),
                    "dynamic_friction": (
                        usd_attribute_value(material_api.GetDynamicFrictionAttr())
                        if material_api
                        else None
                    ),
                    "restitution": (
                        usd_attribute_value(material_api.GetRestitutionAttr())
                        if material_api
                        else None
                    ),
                    "friction_combine_mode": (
                        usd_attribute_value(
                            physx_material_api.GetFrictionCombineModeAttr()
                        )
                        if physx_material_api
                        else None
                    ),
                }
            )
        return records

    def apply_fingertip_friction_override(coefficient: float) -> dict:
        """Bind one high-friction runtime material only to fingertip colliders."""
        material_path = "/World/RuntimePhysicsMaterials/FingertipFrictionDiagnostic"
        material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
        material_prim = material.GetPrim()
        material_api = (
            UsdPhysics.MaterialAPI(material_prim)
            if material_prim.HasAPI(UsdPhysics.MaterialAPI)
            else UsdPhysics.MaterialAPI.Apply(material_prim)
        )
        material_api.CreateStaticFrictionAttr().Set(float(coefficient))
        material_api.CreateDynamicFrictionAttr().Set(float(coefficient))
        material_api.CreateRestitutionAttr().Set(0.0)
        physx_material_api = (
            PhysxSchema.PhysxMaterialAPI(material_prim)
            if material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
            else PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        )
        physx_material_api.CreateFrictionCombineModeAttr().Set(
            PhysxSchema.Tokens.max
        )

        collision_prim_paths = []
        finger_link_prim_paths = []
        resolved_binding_relationships = []
        panda_root = stage.GetPrimAtPath(args.panda_prim)
        if not panda_root.IsValid():
            raise RuntimeError(f"Panda root prim does not exist: {args.panda_prim}")
        for finger_link_name in ("panda_leftfinger", "panda_rightfinger"):
            # The Franka asset repeats the link name on a geometry descendant,
            # e.g. panda_leftfinger/geometry/panda_leftfinger. Select the
            # articulation link by its direct path below the configured Panda
            # root instead of relying on a non-unique leaf name.
            finger_link_path = Sdf.Path(args.panda_prim).AppendChild(
                finger_link_name
            )
            finger_link_prim = stage.GetPrimAtPath(finger_link_path)
            if not finger_link_prim.IsValid():
                raise RuntimeError(
                    f"Panda finger link prim does not exist: {finger_link_path}"
                )
            if finger_link_prim.IsInstanceProxy():
                raise RuntimeError(
                    f"{finger_link_name} is an uneditable instance proxy: "
                    f"{finger_link_prim.GetPath()}"
                )

            # Instance proxies are read-only. Bind once on the editable finger
            # link and use standard USD material inheritance. The stronger
            # binding intentionally overrides descendant materials for this
            # runtime diagnostic only.
            binding_api = (
                UsdShade.MaterialBindingAPI(finger_link_prim)
                if finger_link_prim.HasAPI(UsdShade.MaterialBindingAPI)
                else UsdShade.MaterialBindingAPI.Apply(finger_link_prim)
            )
            if not binding_api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            ):
                raise RuntimeError(
                    f"failed to bind fingertip physics material to "
                    f"{finger_link_prim.GetPath()}"
                )
            finger_link_prim_paths.append(str(finger_link_prim.GetPath()))

            colliders = [
                prim
                for prim in Usd.PrimRange(
                    finger_link_prim, Usd.TraverseInstanceProxies()
                )
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            ]
            if not colliders:
                traversed_paths = [
                    str(prim.GetPath())
                    for prim in Usd.PrimRange(
                        finger_link_prim, Usd.TraverseInstanceProxies()
                    )
                ]
                raise RuntimeError(
                    f"no collision geometry resolved below {finger_link_prim.GetPath()}; "
                    f"traversed={traversed_paths}"
                )
            for collider in colliders:
                resolved_material, binding_relationship = (
                    UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
                        "physics"
                    )
                )
                resolved_prim = (
                    resolved_material.GetPrim() if resolved_material else None
                )
                resolved_path = (
                    str(resolved_prim.GetPath()) if resolved_prim else None
                )
                if resolved_path != material_path:
                    raise RuntimeError(
                        "fingertip physics material read-back failed for "
                        f"{collider.GetPath()}: resolved {resolved_path!r}"
                    )
                collision_prim_paths.append(str(collider.GetPath()))
                resolved_binding_relationships.append(
                    str(binding_relationship.GetPath())
                    if binding_relationship
                    else None
                )
        if not collision_prim_paths:
            raise RuntimeError("Panda fingertip collision geometry was not found")
        return {
            "applied": True,
            "diagnostic_only": True,
            "material_path": material_path,
            "static_friction": float(coefficient),
            "dynamic_friction": float(coefficient),
            "restitution": 0.0,
            "friction_combine_mode": "max",
            "binding_scope": "editable finger links inherited by collision prims",
            "finger_link_prim_paths": finger_link_prim_paths,
            "collision_prim_paths": collision_prim_paths,
            "resolved_binding_relationships": resolved_binding_relationships,
            "effective_material_readback_passed": True,
            "target_material_changed": False,
            "table_material_changed": False,
            "hardware_calibrated": False,
        }

    def finger_drive_configuration(joint_name: str, *, apply_requested: bool) -> dict:
        panda_root = stage.GetPrimAtPath(args.panda_prim)
        matching_prims = [
            prim
            for prim in Usd.PrimRange(panda_root)
            if prim.GetName() == joint_name
        ]
        if len(matching_prims) != 1:
            return {
                "joint_name": joint_name,
                "found": False,
                "matching_prim_paths": [str(prim.GetPath()) for prim in matching_prims],
            }
        joint_prim = matching_prims[0]
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
        if not drive:
            return {
                "joint_name": joint_name,
                "joint_prim": str(joint_prim.GetPath()),
                "found": False,
                "reason": "linear DriveAPI is absent",
            }
        drive_attributes = {
            "max_force": drive.GetMaxForceAttr(),
            "stiffness": drive.GetStiffnessAttr(),
            "damping": drive.GetDampingAttr(),
        }
        before = {
            name: usd_attribute_value(attribute)
            for name, attribute in drive_attributes.items()
        }
        if apply_requested:
            for name, requested_value in requested_finger_drive_values.items():
                if requested_value is not None:
                    drive_attributes[name].Set(float(requested_value))
        after = {
            name: usd_attribute_value(attribute)
            for name, attribute in drive_attributes.items()
        }
        return {
            "joint_name": joint_name,
            "joint_prim": str(joint_prim.GetPath()),
            "found": True,
            "drive_type": usd_attribute_value(drive.GetTypeAttr()),
            "max_force_before": before["max_force"],
            "max_force_after": after["max_force"],
            "max_force_changed": before["max_force"] != after["max_force"],
            "stiffness_before": before["stiffness"],
            "stiffness_after": after["stiffness"],
            "stiffness_changed": before["stiffness"] != after["stiffness"],
            "damping_before": before["damping"],
            "damping_after": after["damping"],
            "damping_changed": before["damping"] != after["damping"],
            # Backward-compatible aliases describe the effective values.
            "stiffness": after["stiffness"],
            "damping": after["damping"],
            "target_position": usd_attribute_value(drive.GetTargetPositionAttr()),
            "target_velocity": usd_attribute_value(drive.GetTargetVelocityAttr()),
        }

    def unique_panda_hand_rigid_body_path() -> str:
        return unique_named_rigid_body_path("panda_hand")

    def fixed_joint_target_contacts_for_latest_step() -> dict:
        """Return target contact evidence for each Panda finger in this step."""
        if physx_simulation_interface is None:
            return {
                "left": False,
                "right": False,
                "matching_headers": [],
            }
        contact_headers, _ = physx_simulation_interface.get_contact_report()
        contact_by_finger = {"left": False, "right": False}
        matching_headers = []
        for header in contact_headers:
            event_type = str(header.type)
            if event_type.endswith("CONTACT_LOST"):
                continue
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            actors = {actor0, actor1}
            for finger_name, finger_path in fixed_joint_finger_paths.items():
                if actors != {target_rigid_prim_path, finger_path}:
                    continue
                contact_count = int(header.num_contact_data)
                if contact_count <= 0:
                    continue
                contact_by_finger[finger_name] = True
                matching_headers.append(
                    {
                        "finger": finger_name,
                        "event_type": event_type,
                        "actor0": actor0,
                        "actor1": actor1,
                        "collider0": str(
                            PhysicsSchemaTools.intToSdfPath(header.collider0)
                        ),
                        "collider1": str(
                            PhysicsSchemaTools.intToSdfPath(header.collider1)
                        ),
                        "contact_count": contact_count,
                    }
                )
        return {
            **contact_by_finger,
            "matching_headers": matching_headers,
        }

    def world_pose_for_xform(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        view = (
            panda_hand_pose_view
            if prim_path == panda_hand_rigid_body_path
            else XFormPrim(
                prim_paths_expr=prim_path,
                name="runtime_grasp_attachment_frame_fallback",
                reset_xform_properties=False,
            )
        )
        positions, orientations = view.get_world_poses()
        positions = np.asarray(positions, dtype=np.float64)
        orientations = np.asarray(orientations, dtype=np.float64)
        if positions.shape != (1, 3) or orientations.shape != (1, 4):
            raise RuntimeError(
                f"unexpected world-pose shape for {prim_path}: "
                f"{positions.shape}, {orientations.shape}"
            )
        return positions[0], orientations[0]

    def get_target_world_pose() -> tuple[np.ndarray, np.ndarray]:
        if scene_usd is None:
            position, orientation = target.get_world_pose()
            return np.asarray(position), np.asarray(orientation)
        positions, orientations = target.get_world_poses()
        positions = np.asarray(positions)
        orientations = np.asarray(orientations)
        if positions.shape != (1, 3) or orientations.shape != (1, 4):
            raise RuntimeError(
                "authored target RigidPrim returned unexpected pose shapes: "
                f"{positions.shape}, {orientations.shape}"
            )
        return positions[0], orientations[0]

    def set_target_world_pose(position: np.ndarray, orientation: np.ndarray) -> None:
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if scene_usd is None:
            target.set_world_pose(position=position, orientation=orientation)
        else:
            target.set_world_poses(
                positions=position.reshape(1, 3),
                orientations=orientation.reshape(1, 4),
            )

    def clear_target_velocities() -> None:
        """Remove residual dynamic motion before switching to kinematic control."""
        if scene_usd is None:
            target.set_linear_velocity(np.zeros(3, dtype=np.float64))
            target.set_angular_velocity(np.zeros(3, dtype=np.float64))
        else:
            target.set_velocities(np.zeros((1, 6), dtype=np.float64))

    def set_target_kinematic_enabled(enabled: bool) -> tuple[bool, bool]:
        """Set and verify the standard USD rigid-body kinematic flag."""
        target_rigid_prim = stage.GetPrimAtPath(target_rigid_prim_path)
        if not target_rigid_prim.IsValid():
            raise RuntimeError(
                f"target rigid body prim does not exist: {target_rigid_prim_path}"
            )
        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(
            stage, target_rigid_prim.GetPath()
        )
        if not rigid_body_api or not rigid_body_api.GetPrim().IsValid():
            raise RuntimeError(
                "target rigid body has no UsdPhysics.RigidBodyAPI: "
                f"{target_rigid_prim_path}"
            )
        kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()
        previous_value = bool(kinematic_attr.Get()) if kinematic_attr else False
        rigid_body_api.CreateKinematicEnabledAttr().Set(bool(enabled))
        applied_value = bool(rigid_body_api.GetKinematicEnabledAttr().Get())
        if applied_value != bool(enabled):
            raise RuntimeError(
                "requested target kinematic state was not applied: "
                f"requested={enabled}, readback={applied_value}"
            )
        return previous_value, applied_value

    panda_hand_rigid_body_path = unique_panda_hand_rigid_body_path()
    panda_hand_pose_view = XFormPrim(
        prim_paths_expr=panda_hand_rigid_body_path,
        name="runtime_grasp_attachment_frame",
        reset_xform_properties=False,
    )
    attachment_state: dict[str, object | None] = {
        "hand_rigid_body_path": panda_hand_rigid_body_path,
        "T_hand_target": None,
    }

    def apply_target_robot_collision_filter() -> None:
        target_rigid_prim = stage.GetPrimAtPath(target_rigid_prim_path)
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(target_rigid_prim)
        if not filtered_pairs:
            raise RuntimeError(
                "could not apply pairwise collision filtering to attached target"
            )
        filtered_pairs.CreateFilteredPairsRel().AddTarget(Sdf.Path(args.panda_prim))

    def attachment_pose_jump(
        target_position_before: np.ndarray,
        target_orientation_before: np.ndarray,
    ) -> tuple[float, float]:
        target_position_after, target_orientation_after = get_target_world_pose()
        position_jump_m = float(
            np.linalg.norm(target_position_after - target_position_before)
        )
        orientation_dot = float(
            np.clip(
                abs(np.dot(target_orientation_after, target_orientation_before)),
                0.0,
                1.0,
            )
        )
        orientation_jump_rad = float(2.0 * np.arccos(orientation_dot))
        if (
            position_jump_m > ATTACHMENT_TRANSLATION_TOLERANCE_M
            or orientation_jump_rad > ATTACHMENT_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "post-close attachment changed the target pose unexpectedly: "
                f"position={position_jump_m:.6g} m, "
                f"orientation={orientation_jump_rad:.6g} rad"
            )
        return position_jump_m, orientation_jump_rad

    def register_attachment_reference(T_hand_target: np.ndarray) -> str:
        transform_path = output / "T_panda_hand_target_at_attachment.npy"
        np.save(transform_path, T_hand_target)
        attachment_state["T_hand_target"] = np.asarray(
            T_hand_target, dtype=np.float64
        ).copy()
        return str(transform_path)

    def enforce_kinematic_pose_lock() -> None:
        if args.grasp_retention_mode != "kinematic-pose-lock":
            return
        T_hand_target = attachment_state["T_hand_target"]
        if T_hand_target is None:
            return
        hand_position, hand_orientation = world_pose_for_xform(
            panda_hand_rigid_body_path
        )
        T_world_hand = matrix_from_pose(hand_position, hand_orientation)
        T_world_target = T_world_hand @ np.asarray(T_hand_target)
        set_target_world_pose(
            T_world_target[:3, 3],
            quaternion_wxyz_from_rotation_matrix(T_world_target[:3, :3]),
        )

    def step_world_with_attachment_sync() -> None:
        """Render kinematic attachment poses only after their final frame update."""
        if (
            args.grasp_retention_mode == "kinematic-pose-lock"
            and attachment_state["T_hand_target"] is not None
        ):
            # Pull the just-simulated Panda link pose into Fabric, update the
            # target from that pose, and only then render the visible frame.
            world.step(render=False, update_fabric=True)
            enforce_kinematic_pose_lock()
            world.render()
        else:
            world.step(render=True)

    def create_post_close_fixed_attachment() -> dict:
        """Lock the current target-to-hand pose without snapping either body."""
        hand_rigid_body_path = panda_hand_rigid_body_path
        hand_position, hand_orientation = world_pose_for_xform(hand_rigid_body_path)
        target_position_before, target_orientation_before = get_target_world_pose()
        relative_position, relative_orientation, T_hand_target = relative_pose(
            hand_position,
            hand_orientation,
            target_position_before,
            target_orientation_before,
        )

        joint_path = "/World/RuntimeGraspAttachment"
        if stage.GetPrimAtPath(joint_path).IsValid():
            raise RuntimeError(f"runtime grasp attachment already exists: {joint_path}")
        joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_rigid_body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(target_rigid_prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*relative_position.astype(float)))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(relative_orientation[0]),
                Gf.Vec3f(*relative_orientation[1:].astype(float)),
            )
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint_readback = {
            "body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
            "body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
            "collision_enabled": bool(joint.GetCollisionEnabledAttr().Get()),
            "exclude_from_articulation": bool(
                joint.GetExcludeFromArticulationAttr().Get()
            ),
        }
        expected_readback = {
            "body0": [hand_rigid_body_path],
            "body1": [target_rigid_prim_path],
            "collision_enabled": False,
            "exclude_from_articulation": True,
        }
        if joint_readback != expected_readback:
            raise RuntimeError(
                "FixedJoint USD relationship readback did not match the request: "
                f"{joint_readback}"
            )
        apply_target_robot_collision_filter()
        transform_file = register_attachment_reference(T_hand_target)
        world.step(render=True)
        joint_prim_after_step = stage.GetPrimAtPath(joint_path)
        if not joint_prim_after_step.IsValid() or not joint_prim_after_step.IsActive():
            raise RuntimeError("FixedJoint prim was not active after its first physics step")
        record_physics_sample("attach")
        position_jump_m, orientation_jump_rad = attachment_pose_jump(
            target_position_before, target_orientation_before
        )
        return {
            "mode": "rigid-attachment",
            "applied": True,
            "assumption": (
                "grasp accepted; target-to-panda_hand transform remains fixed"
            ),
            "joint_type": "UsdPhysics.FixedJoint",
            "joint_prim": joint_path,
            "joint_usd_readback": joint_readback,
            "joint_active_after_first_physics_step": True,
            "hand_rigid_body_prim": hand_rigid_body_path,
            "target_rigid_body_prim": target_rigid_prim_path,
            "connected_body_collision_enabled": False,
            "target_robot_collision_filtered": True,
            "target_robot_collision_filter_path": args.panda_prim,
            "excluded_from_robot_articulation": True,
            "T_panda_hand_target": T_hand_target.tolist(),
            "transform_file": transform_file,
            "target_pose_jump_after_attachment_m": position_jump_m,
            "target_orientation_jump_after_attachment_rad": orientation_jump_rad,
        }

    def create_post_close_physx_auto_attachment() -> dict:
        """Apply the PhysX runtime auto-attachment recipe at the current poses."""
        if not hasattr(PhysxSchema, "PhysxPhysicsAttachment") or not hasattr(
            PhysxSchema, "PhysxAutoAttachmentAPI"
        ):
            raise RuntimeError(
                "this Isaac Sim build has no PhysxPhysicsAttachment/"
                "PhysxAutoAttachmentAPI schema"
            )
        hand_position, hand_orientation = world_pose_for_xform(
            panda_hand_rigid_body_path
        )
        target_position_before, target_orientation_before = get_target_world_pose()
        hand_position_at_attachment, hand_orientation_at_attachment = (
            world_pose_for_xform(panda_hand_rigid_body_path)
        )
        target_position_at_attachment, target_orientation_at_attachment = (
            get_target_world_pose()
        )
        _, _, T_hand_target = relative_pose(
            hand_position_at_attachment,
            hand_orientation_at_attachment,
            target_position_at_attachment,
            target_orientation_at_attachment,
        )
        attachment_path = Sdf.Path(panda_hand_rigid_body_path).AppendChild(
            "runtime_target_attachment"
        )
        if stage.GetPrimAtPath(attachment_path).IsValid():
            raise RuntimeError(
                f"runtime PhysX attachment already exists: {attachment_path}"
            )
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        attachment.GetActor0Rel().SetTargets(
            [Sdf.Path(panda_hand_rigid_body_path)]
        )
        attachment.GetActor1Rel().SetTargets([Sdf.Path(target_rigid_prim_path)])
        auto_attachment = PhysxSchema.PhysxAutoAttachmentAPI.Apply(
            attachment.GetPrim()
        )
        if not auto_attachment or not auto_attachment.GetPrim().IsValid():
            raise RuntimeError("could not apply PhysxAutoAttachmentAPI")
        apply_target_robot_collision_filter()
        transform_file = register_attachment_reference(T_hand_target)
        world.step(render=True)
        record_physics_sample("attach")
        if target_gripped:
            position_jump_m, orientation_jump_rad = attachment_pose_jump(
                target_position_before, target_orientation_before
            )
        else:
            position_jump_m = None
            orientation_jump_rad = None
        return {
            "mode": "physx-auto-attachment",
            "applied": True,
            "assumption": (
                "grasp accepted; PhysX attachment preserves the current "
                "target-to-panda_hand relation"
            ),
            "schema_type": "PhysxSchema.PhysxPhysicsAttachment",
            "auto_attachment_api": "PhysxSchema.PhysxAutoAttachmentAPI",
            "attachment_prim": str(attachment_path),
            "hand_rigid_body_prim": panda_hand_rigid_body_path,
            "target_rigid_body_prim": target_rigid_prim_path,
            "target_robot_collision_filtered": True,
            "target_robot_collision_filter_path": args.panda_prim,
            "T_panda_hand_target": T_hand_target.tolist(),
            "transform_file": transform_file,
            "target_pose_jump_after_attachment_m": position_jump_m,
            "target_orientation_jump_after_attachment_rad": orientation_jump_rad,
            "implementation_reference": (
                "https://github.com/isaac-sim/IsaacLab/discussions/4189"
            ),
            "experimental_runtime_schema": True,
        }

    def create_post_close_surface_gripper_attachment() -> dict:
        """Close the official Isaac Sim 5.1 Surface Gripper on the target.

        The D6 attachment-point physics are copied from the exact bundled
        SurfaceGripper_gantry.usda installed in this environment. Only Body 0
        and the attachment point pose are adapted to the current Panda grasp.
        """

        def template_path() -> Path:
            package_path = Path(inspect.getfile(surface_gripper_package)).resolve()
            candidates = [
                parent / "data" / "SurfaceGripper_gantry.usda"
                for parent in package_path.parents
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            matches = sorted(
                Path(sys.prefix).rglob("SurfaceGripper_gantry.usda")
            )
            if not matches:
                raise RuntimeError(
                    "Isaac Sim SurfaceGripper_gantry.usda was not found below "
                    f"{sys.prefix}"
                )
            return matches[0]

        source_path = template_path()
        source_stage = Usd.Stage.Open(str(source_path))
        if source_stage is None:
            raise RuntimeError(f"could not open official Surface Gripper USD: {source_path}")
        attachment_relation_name = robot_schema.Relations.ATTACHMENT_POINTS.name
        # NVIDIA's example USD stores the D6 joints, while the Surface Gripper
        # prim and its attachment-points relationship are added by example code.
        # This exact joint-root path is documented in the 5.1 code example.
        source_joint_root = source_stage.GetPrimAtPath(
            "/World/Surface_Gripper_Joints"
        )
        source_joint_candidates = (
            [
                prim
                for prim in source_joint_root.GetChildren()
                if prim.IsA(UsdPhysics.Joint)
            ]
            if source_joint_root.IsValid()
            else []
        )
        if not source_joint_candidates:
            source_joint_candidates = [
                prim
                for prim in source_stage.Traverse()
                if prim.IsA(UsdPhysics.Joint)
                and "gripper" in str(prim.GetPath()).lower()
            ]
        if not source_joint_candidates:
            raise RuntimeError(
                "official Surface Gripper USD contains no gripper D6 joints"
            )
        source_joint_prim = source_joint_candidates[0]
        source_joint_path = source_joint_prim.GetPath()
        source_joint_prim = source_stage.GetPrimAtPath(source_joint_path)
        if not source_joint_prim.IsValid():
            raise RuntimeError(
                f"official Surface Gripper D6 joint is missing: {source_joint_path}"
            )

        runtime_joint_path = Sdf.Path("/World/RuntimeSurfaceGripperAttachmentPoint")
        runtime_gripper_path = Sdf.Path("/World/RuntimeSurfaceGripper")
        for runtime_path in (runtime_joint_path, runtime_gripper_path):
            if stage.GetPrimAtPath(runtime_path).IsValid():
                raise RuntimeError(f"runtime Surface Gripper prim already exists: {runtime_path}")

        flattened_template = source_stage.Flatten()
        copied = Sdf.CopySpec(
            flattened_template,
            source_joint_path,
            stage.GetEditTarget().GetLayer(),
            runtime_joint_path,
        )
        if not copied:
            raise RuntimeError("could not copy official Surface Gripper D6 template")
        runtime_joint_prim = stage.GetPrimAtPath(runtime_joint_path)
        if not runtime_joint_prim.IsValid():
            raise RuntimeError("copied Surface Gripper D6 joint is invalid")
        runtime_joint = UsdPhysics.Joint(runtime_joint_prim)
        runtime_joint.CreateBody0Rel().SetTargets(
            [Sdf.Path(panda_hand_rigid_body_path)]
        )
        runtime_joint.CreateBody1Rel().ClearTargets(True)
        runtime_joint.CreateJointEnabledAttr().Set(True)
        runtime_joint.CreateExcludeFromArticulationAttr().Set(True)
        runtime_joint.GetBreakForceAttr().Clear()
        runtime_joint.GetBreakTorqueAttr().Clear()
        robot_schema.ApplyAttachmentPointAPI(runtime_joint_prim)

        forward_axis_attr = runtime_joint_prim.GetAttribute(
            robot_schema.Attributes.FORWARD_AXIS.name
        )
        forward_axis = str(forward_axis_attr.Get() or "X")
        clearance_offset_attr = runtime_joint_prim.GetAttribute(
            robot_schema.Attributes.CLEARANCE_OFFSET.name
        )
        clearance_offset_m = float(clearance_offset_attr.Get() or 0.0)
        hand_position, hand_orientation = world_pose_for_xform(
            panda_hand_rigid_body_path
        )
        target_position_before, target_orientation_before = get_target_world_pose()
        current_aabb = np.asarray(
            compute_aabb(create_bbox_cache(), target_prim_path, include_children=True),
            dtype=np.float64,
        )
        # The manager advances the ray origin by the template's clearance
        # offset. Keep the advanced origin 2 mm outside the observed AABB.
        ray_origin_clearance_m = clearance_offset_m + 0.002
        attachment_position_world, forward_world = aabb_ray_origin_toward_center(
            current_aabb,
            hand_position,
            ray_origin_clearance_m,
        )
        attachment_rotation_world = rotation_matrix_align_axis_to_vector(
            forward_axis, forward_world
        )
        attachment_orientation_world = quaternion_wxyz_from_rotation_matrix(
            attachment_rotation_world
        )
        local_position, local_orientation, _ = relative_pose(
            hand_position,
            hand_orientation,
            attachment_position_world,
            attachment_orientation_world,
        )
        runtime_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*local_position.astype(float))
        )
        runtime_joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_orientation[0]),
                Gf.Vec3f(*local_orientation[1:].astype(float)),
            )
        )
        runtime_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        runtime_joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )

        robot_schema.CreateSurfaceGripper(stage, str(runtime_gripper_path))
        runtime_gripper_prim = stage.GetPrimAtPath(runtime_gripper_path)
        if not runtime_gripper_prim.IsValid():
            raise RuntimeError("CreateSurfaceGripper returned an invalid prim")
        runtime_gripper_prim.GetRelationship(attachment_relation_name).SetTargets(
            [runtime_joint_path]
        )
        # These are the values in NVIDIA's Isaac Sim 5.1 "Creating a Surface
        # Gripper fully on code" example. They are not candidate-specific tuning.
        documented_gripper_properties = {
            robot_schema.Attributes.MAX_GRIP_DISTANCE.name: 0.011,
            robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name: 0.005,
            robot_schema.Attributes.SHEAR_FORCE_LIMIT.name: 5.0,
            robot_schema.Attributes.RETRY_INTERVAL.name: 1.0,
        }
        copied_gripper_properties = {}
        for attribute_name, value in documented_gripper_properties.items():
            runtime_gripper_prim.GetAttribute(attribute_name).Set(value)
            copied_gripper_properties[attribute_name] = float(value)

        gripper_interface = surface_gripper.acquire_surface_gripper_interface()
        write_to_usd_enabled = bool(gripper_interface.set_write_to_usd(True))
        world.step(render=True)
        opened = bool(gripper_interface.open_gripper(str(runtime_gripper_path)))
        world.step(render=True)
        close_requested = bool(
            gripper_interface.close_gripper(str(runtime_gripper_path))
        )
        gripped_objects: list[str] = []
        status = None
        close_wait_frames = 0
        for close_wait_frames in range(1, 61):
            world.step(render=True)
            status = str(
                gripper_interface.get_gripper_status(str(runtime_gripper_path))
            )
            gripped_objects = [
                str(path)
                for path in gripper_interface.get_gripped_objects(
                    str(runtime_gripper_path)
                )
            ]
            if gripped_objects:
                break
        target_gripped = any(
            path == target_rigid_prim_path
            or path.startswith(target_rigid_prim_path + "/")
            or path == target_prim_path
            or path.startswith(target_prim_path + "/")
            for path in gripped_objects
        )
        _, _, T_hand_target = relative_pose(
            hand_position,
            hand_orientation,
            target_position_before,
            target_orientation_before,
        )
        if target_gripped:
            apply_target_robot_collision_filter()
        transform_file = register_attachment_reference(T_hand_target)
        world.step(render=True)
        record_physics_sample("attach")
        position_jump_m, orientation_jump_rad = attachment_pose_jump(
            target_position_before, target_orientation_before
        )
        return {
            "mode": "surface-gripper-attachment",
            "applied": target_gripped,
            "assumption": (
                "grasp accepted; Isaac Sim Surface Gripper manages an official "
                "D6 attachment point copied from its bundled example"
                if target_gripped
                else "Surface Gripper close was attempted but did not attach the target"
            ),
            "surface_gripper_prim": str(runtime_gripper_path),
            "attachment_point_prim": str(runtime_joint_path),
            "hand_rigid_body_prim": panda_hand_rigid_body_path,
            "target_rigid_body_prim": target_rigid_prim_path,
            "target_robot_collision_filtered": target_gripped,
            "target_robot_collision_filter_path": (
                args.panda_prim if target_gripped else None
            ),
            "interface": {
                "set_write_to_usd_returned": write_to_usd_enabled,
                "open_gripper_returned": opened,
                "close_gripper_returned": close_requested,
                "status_after_close": status,
                "gripped_objects": gripped_objects,
                "target_gripped": target_gripped,
                "close_wait_frames": close_wait_frames,
            },
            "official_template": str(source_path),
            "official_template_surface_gripper": None,
            "official_template_attachment_point": str(source_joint_path),
            "copied_gripper_properties": copied_gripper_properties,
            "gripper_property_source": (
                "Isaac Sim 5.1 documented Creating a Surface Gripper fully on code example"
            ),
            "forward_axis": forward_axis,
            "template_clearance_offset_m": clearance_offset_m,
            "ray_origin_clearance_m": ray_origin_clearance_m,
            "attachment_position_world_m": attachment_position_world.tolist(),
            "T_panda_hand_target": T_hand_target.tolist(),
            "transform_file": transform_file,
            "target_pose_jump_after_attachment_m": position_jump_m,
            "target_orientation_jump_after_attachment_rad": orientation_jump_rad,
            "implementation_reference": (
                "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/"
                "robot_simulation/ext_isaacsim_robot_surface_gripper.html"
            ),
            "simulation_retention_abstraction": True,
            "parallel_jaw_contact_model": False,
        }

    def create_post_close_kinematic_pose_lock() -> dict:
        """Switch the target to kinematic control and preserve its hand-relative pose."""
        hand_position, hand_orientation = world_pose_for_xform(
            panda_hand_rigid_body_path
        )
        target_position_before, target_orientation_before = get_target_world_pose()
        _, _, T_hand_target = relative_pose(
            hand_position,
            hand_orientation,
            target_position_before,
            target_orientation_before,
        )
        apply_target_robot_collision_filter()
        transform_file = register_attachment_reference(T_hand_target)
        clear_target_velocities()
        kinematic_before, kinematic_after = set_target_kinematic_enabled(True)
        # Preserve the measured attachment pose while PhysX consumes the
        # runtime kinematic-state change. No constraint is added to the Panda.
        enforce_kinematic_pose_lock()
        step_world_with_attachment_sync()
        record_physics_sample("attach")
        position_jump_m, orientation_jump_rad = attachment_pose_jump(
            target_position_before, target_orientation_before
        )
        return {
            "mode": "kinematic-pose-lock",
            "applied": True,
            "assumption": (
                "grasp accepted; target is kinematic and follows the measured "
                "target-to-panda_hand transform without a physics joint"
            ),
            "hand_rigid_body_prim": panda_hand_rigid_body_path,
            "target_rigid_body_prim": target_rigid_prim_path,
            "kinematic_enabled_before_attachment": kinematic_before,
            "kinematic_enabled_after_attachment": kinematic_after,
            "dynamic_velocity_cleared_before_attachment": True,
            "physics_joint_created": False,
            "target_robot_collision_filtered": True,
            "target_robot_collision_filter_path": args.panda_prim,
            "T_panda_hand_target": T_hand_target.tolist(),
            "transform_file": transform_file,
            "target_pose_jump_after_attachment_m": position_jump_m,
            "target_orientation_jump_after_attachment_rad": orientation_jump_rad,
            "simulation_pose_override": True,
        }

    def save_rgb(label: str) -> str | None:
        frame = camera.get_current_frame()
        rgba = frame.get("rgba")
        if rgba is None or np.asarray(rgba).size <= 1:
            rgba = frame.get("rgb")
        if rgba is None or np.asarray(rgba).size <= 1:
            return None
        rgb = np.asarray(rgba)[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        path = output / f"{label}.png"
        Image.fromarray(rgb).save(path)
        return str(path)

    for _ in range(args.settle_frames):
        world.step(render=True)
    target_settled_position, target_settled_orientation = get_target_world_pose()
    target_settled_position = np.asarray(target_settled_position, dtype=np.float64)
    target_settled_aabb = np.asarray(
        compute_aabb(create_bbox_cache(), target_prim_path, include_children=True),
        dtype=np.float64,
    )
    target_settled_extent = target_settled_aabb[3:] - target_settled_aabb[:3]
    if (
        target_settled_aabb.shape != (6,)
        or not np.all(np.isfinite(target_settled_aabb))
        or not np.all(target_settled_extent > 1e-4)
    ):
        raise RuntimeError(
            f"invalid target AABB after settling: {target_settled_aabb}"
        )
    saved_frames: list[str] = []
    first_frame = save_rgb("00_settled")
    if first_frame:
        saved_frames.append(first_frame)

    isaac_names = tuple(str(name) for name in panda.dof_names)
    if len(set(isaac_names)) != len(isaac_names):
        raise RuntimeError("Isaac Panda DOF names are not unique")
    index_by_name = {name: index for index, name in enumerate(isaac_names)}
    missing = [name for name in replay.joint_names if name not in index_by_name]
    if missing:
        raise RuntimeError(f"planned joints are missing from Isaac Panda: {missing}")
    arm_indices = np.asarray(
        [index_by_name[name] for name in replay.joint_names], dtype=np.int64
    )
    finger_names = ("panda_finger_joint1", "panda_finger_joint2")
    if any(name not in index_by_name for name in finger_names):
        raise RuntimeError("Isaac Panda finger joint names changed")
    finger_indices = np.asarray([index_by_name[name] for name in finger_names], dtype=np.int64)
    capture_by_name = {
        name: replay.capture_joint_positions[index]
        for index, name in enumerate(replay.capture_joint_names)
    }
    expected_start = np.asarray(
        [capture_by_name[name] for name in replay.joint_names], dtype=np.float64
    )
    captured_fingers = np.asarray(
        [capture_by_name[name] for name in finger_names], dtype=np.float64
    )
    open_fingers = np.full(2, args.open_finger_position_m, dtype=np.float64)
    actual_start = np.asarray(panda.get_joint_positions(), dtype=np.float64)[arm_indices]
    start_error = np.abs(actual_start - expected_start)
    if not np.all(start_error <= 2e-3):
        raise RuntimeError(
            "Isaac Panda did not reproduce the capture start state; "
            f"maximum error={float(start_error.max()):.6g}"
        )

    if scene_usd is None:
        target_mass_kg = float(target.get_mass())
        target_density_kg_m3 = None
        target_inertia = None
    else:
        target_masses = np.asarray(target.get_masses(), dtype=np.float64).reshape(-1)
        target_densities = np.asarray(
            target.get_densities(), dtype=np.float64
        ).reshape(-1)
        target_inertias = np.asarray(target.get_inertias(), dtype=np.float64)
        if target_masses.shape != (1,) or target_densities.shape != (1,):
            raise RuntimeError(
                "authored target returned unexpected mass or density shapes: "
                f"{target_masses.shape}, {target_densities.shape}"
            )
        target_mass_kg = float(target_masses[0])
        target_density_kg_m3 = float(target_densities[0])
        target_inertia = target_inertias.reshape(1, -1)[0].tolist()
    if not np.isfinite(target_mass_kg) or target_mass_kg <= 0.0:
        raise RuntimeError(f"target effective mass must be positive: {target_mass_kg}")

    finger_drive_report = [
        finger_drive_configuration(joint_name, apply_requested=True)
        for joint_name in finger_names
    ]
    configured_finger_drives = [
        item for item in finger_drive_report if item.get("found") is True
    ]
    if not configured_finger_drives:
        raise RuntimeError("Panda has no configurable linear finger DriveAPI")
    for attribute_name, requested_value in requested_finger_drive_values.items():
        if requested_value is None:
            continue
        report_key = f"{attribute_name}_after"
        if not all(
            drive_value_matches_float_storage(item[report_key], requested_value)
            for item in configured_finger_drives
        ):
            raise RuntimeError(
                f"requested Panda finger DriveAPI {attribute_name} was not applied"
            )
    fingertip_friction_override = {
        "applied": False,
        "diagnostic_only": False,
        "requested_coefficient": args.fingertip_friction_coefficient,
    }
    if args.fingertip_friction_coefficient is not None:
        fingertip_friction_override = apply_fingertip_friction_override(
            args.fingertip_friction_coefficient
        )

    target_collision_materials = collision_materials_below(target_prim_path)
    finger_collision_materials = []
    for finger_link_name in ("panda_leftfinger", "panda_rightfinger"):
        finger_link_prims = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(args.panda_prim))
            if prim.GetName() == finger_link_name
        ]
        for finger_link_prim in finger_link_prims:
            for material_record in collision_materials_below(
                str(finger_link_prim.GetPath())
            ):
                material_record["finger_link"] = finger_link_name
                finger_collision_materials.append(material_record)

    measurements: dict[str, np.ndarray] = {}
    commands: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    diagnostic_phases: list[str] = []
    diagnostic_target_positions: list[np.ndarray] = []
    diagnostic_target_orientations: list[np.ndarray] = []
    diagnostic_finger_positions: list[np.ndarray] = []
    diagnostic_hand_positions: list[np.ndarray] = []
    diagnostic_hand_orientations: list[np.ndarray] = []
    diagnostic_attachment_translation_errors: list[float] = []
    diagnostic_attachment_orientation_errors: list[float] = []

    def record_physics_sample(phase: str) -> None:
        target_position, target_orientation = get_target_world_pose()
        hand_position, hand_orientation = world_pose_for_xform(
            panda_hand_rigid_body_path
        )
        finger_position = np.asarray(
            panda.get_joint_positions(), dtype=np.float64
        )[finger_indices]
        if not (
            np.isfinite(target_position).all()
            and np.isfinite(target_orientation).all()
            and np.isfinite(hand_position).all()
            and np.isfinite(hand_orientation).all()
            and np.isfinite(finger_position).all()
        ):
            raise RuntimeError(f"non-finite retention diagnostic during {phase}")
        T_hand_target_reference = attachment_state["T_hand_target"]
        if T_hand_target_reference is None:
            attachment_translation_error_m = math.nan
            attachment_orientation_error_rad = math.nan
        else:
            _, _, T_hand_target_current = relative_pose(
                hand_position,
                hand_orientation,
                target_position,
                target_orientation,
            )
            T_hand_target_reference = np.asarray(T_hand_target_reference)
            attachment_translation_error_m = float(
                np.linalg.norm(
                    T_hand_target_current[:3, 3]
                    - T_hand_target_reference[:3, 3]
                )
            )
            rotation_delta = (
                T_hand_target_reference[:3, :3].T
                @ T_hand_target_current[:3, :3]
            )
            attachment_orientation_error_rad = float(
                np.arccos(
                    np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
                )
            )
        diagnostic_phases.append(phase)
        diagnostic_target_positions.append(
            np.asarray(target_position, dtype=np.float64).copy()
        )
        diagnostic_target_orientations.append(
            np.asarray(target_orientation, dtype=np.float64).copy()
        )
        diagnostic_finger_positions.append(finger_position.copy())
        diagnostic_hand_positions.append(
            np.asarray(hand_position, dtype=np.float64).copy()
        )
        diagnostic_hand_orientations.append(
            np.asarray(hand_orientation, dtype=np.float64).copy()
        )
        diagnostic_attachment_translation_errors.append(
            attachment_translation_error_m
        )
        diagnostic_attachment_orientation_errors.append(
            attachment_orientation_error_rad
        )

    record_physics_sample("settled")

    def execute_phase(phase: str, finger_target: np.ndarray) -> None:
        phase_time, phase_commands = sample_positions_at_physics_rate(
            replay.phase_positions[phase],
            replay.phase_segment_dt_s[phase],
            PHYSICS_DT_S,
        )
        phase_measured = np.empty_like(phase_commands)
        all_indices = np.concatenate((arm_indices, finger_indices))
        for index, arm_target in enumerate(phase_commands):
            panda.apply_action(
                ArticulationAction(
                    joint_positions=np.concatenate((arm_target, finger_target)),
                    joint_indices=all_indices,
                )
            )
            step_world_with_attachment_sync()
            record_physics_sample(phase)
            phase_measured[index] = np.asarray(
                panda.get_joint_positions(), dtype=np.float64
            )[arm_indices]
            if not np.isfinite(phase_measured[index]).all():
                raise RuntimeError(f"non-finite Panda state during {phase}")
        commands[phase] = phase_commands
        measurements[phase] = phase_measured
        durations[phase] = float(phase_time[-1])
        frame_path = save_rgb(f"{len(saved_frames):02d}_{phase}_end")
        if frame_path:
            saved_frames.append(frame_path)

    execute_phase("approach", open_fingers)
    execute_phase("grasp", open_fingers)
    measured_fingers_before_close = np.asarray(
        panda.get_joint_positions(), dtype=np.float64
    )[finger_indices]

    grasp_arm_target = commands["grasp"][-1]
    all_indices = np.concatenate((arm_indices, finger_indices))
    closed_finger_target = np.full(
        2, args.closed_finger_position_m, dtype=np.float64
    )
    fixed_joint_attachment_gate = {
        "required": args.grasp_retention_mode == "rigid-attachment",
        "passed": None,
        "source": (
            "NVIDIA recommendation: finger closure threshold plus target contact "
            "reported on both fingers"
        ),
        "implementation_reference": (
            "https://forums.developer.nvidia.com/t/"
            "pick-and-place-in-space-zero-gravity-0g/370975"
        ),
        "contact_monitor": fixed_joint_contact_setup,
        "minimum_closure_travel_per_finger_m": (
            FIXED_JOINT_MIN_FINGER_CLOSURE_TRAVEL_M
        ),
        "maximum_close_frames": args.close_frames,
        "close_frames_executed": 0,
        "final_closure_travel_m": None,
        "final_finger_target_contacts": None,
        "contact_frame": None,
        "samples": [],
    }
    for close_index in range(args.close_frames):
        panda.apply_action(
            ArticulationAction(
                joint_positions=np.concatenate((grasp_arm_target, closed_finger_target)),
                joint_indices=all_indices,
            )
        )
        step_world_with_attachment_sync()
        record_physics_sample("close")
        if args.grasp_retention_mode != "rigid-attachment":
            continue
        measured_close_fingers = np.asarray(
            panda.get_joint_positions(), dtype=np.float64
        )[finger_indices]
        closure_travel_m = measured_fingers_before_close - measured_close_fingers
        contact_sample = fixed_joint_target_contacts_for_latest_step()
        closure_threshold_passed = bool(
            np.all(
                closure_travel_m
                >= FIXED_JOINT_MIN_FINGER_CLOSURE_TRAVEL_M
            )
        )
        bilateral_target_contact = bool(
            contact_sample["left"] and contact_sample["right"]
        )
        fixed_joint_attachment_gate["samples"].append(
            {
                "frame": close_index + 1,
                "measured_finger_positions_m": measured_close_fingers.tolist(),
                "closure_travel_m": closure_travel_m.tolist(),
                "closure_threshold_passed": closure_threshold_passed,
                "left_target_contact": bool(contact_sample["left"]),
                "right_target_contact": bool(contact_sample["right"]),
                "bilateral_target_contact": bilateral_target_contact,
                "matching_contact_headers": contact_sample["matching_headers"],
            }
        )
        if closure_threshold_passed and bilateral_target_contact:
            fixed_joint_attachment_gate["passed"] = True
            fixed_joint_attachment_gate["contact_frame"] = close_index + 1
            break
    measured_fingers_after_close = np.asarray(
        panda.get_joint_positions(), dtype=np.float64
    )[finger_indices]
    if args.grasp_retention_mode == "rigid-attachment":
        final_contacts = fixed_joint_target_contacts_for_latest_step()
        fixed_joint_attachment_gate["close_frames_executed"] = len(
            fixed_joint_attachment_gate["samples"]
        )
        fixed_joint_attachment_gate["final_closure_travel_m"] = (
            measured_fingers_before_close - measured_fingers_after_close
        ).tolist()
        fixed_joint_attachment_gate["final_finger_target_contacts"] = {
            "left": bool(final_contacts["left"]),
            "right": bool(final_contacts["right"]),
            "matching_headers": final_contacts["matching_headers"],
        }
        if fixed_joint_attachment_gate["passed"] is not True:
            fixed_joint_attachment_gate["passed"] = False
            gate_failure_path = output / "fixed_joint_attachment_gate_failure.json"
            gate_failure_path.write_text(
                json.dumps(
                    {
                        "status": "fixed_joint_attachment_gate_failed",
                        "grasp_retention_mode": args.grasp_retention_mode,
                        "gate": fixed_joint_attachment_gate,
                        "safety": {
                            "fixed_joint_created": False,
                            "lift_executed": False,
                            "simulation_only": True,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "refusing to create FixedJoint: finger closure and bilateral "
                "target contact were not observed together; inspect "
                f"{gate_failure_path}"
            )
    else:
        fixed_joint_attachment_gate["close_frames_executed"] = args.close_frames
    frame_path = save_rgb("03_gripper_closed")
    if frame_path:
        saved_frames.append(frame_path)

    attachment_report = {
        "mode": args.grasp_retention_mode,
        "applied": False,
        "assumption": None,
    }
    if args.grasp_retention_mode == "rigid-attachment":
        attachment_report = create_post_close_fixed_attachment()
        attachment_report["creation_gate"] = fixed_joint_attachment_gate
        settle_start_index = len(diagnostic_phases)
        for _ in range(FIXED_JOINT_POST_CREATE_SETTLE_FRAMES):
            panda.apply_action(
                ArticulationAction(
                    joint_positions=np.concatenate(
                        (grasp_arm_target, closed_finger_target)
                    ),
                    joint_indices=all_indices,
                )
            )
            step_world_with_attachment_sync()
            record_physics_sample("post_attach_settle")
        settle_translation_errors = np.asarray(
            diagnostic_attachment_translation_errors[settle_start_index:],
            dtype=np.float64,
        )
        settle_orientation_errors = np.asarray(
            diagnostic_attachment_orientation_errors[settle_start_index:],
            dtype=np.float64,
        )
        post_create_settle = {
            "frames": FIXED_JOINT_POST_CREATE_SETTLE_FRAMES,
            "arm_command": "hold final grasp joint positions",
            "finger_command": "hold closed finger target positions",
            "maximum_relative_translation_error_m": float(
                np.max(settle_translation_errors)
            ),
            "maximum_relative_orientation_error_rad": float(
                np.max(settle_orientation_errors)
            ),
        }
        post_create_settle["relative_pose_within_tolerance"] = bool(
            post_create_settle["maximum_relative_translation_error_m"]
            <= ATTACHMENT_TRANSLATION_TOLERANCE_M
            and post_create_settle["maximum_relative_orientation_error_rad"]
            <= ATTACHMENT_ORIENTATION_TOLERANCE_RAD
        )
        attachment_report["post_create_settle"] = post_create_settle
        if post_create_settle["relative_pose_within_tolerance"] is not True:
            settle_failure_path = output / "fixed_joint_post_create_failure.json"
            settle_failure_path.write_text(
                json.dumps(
                    {
                        "status": "fixed_joint_not_stable_before_lift",
                        "post_close_attachment": attachment_report,
                        "safety": {
                            "lift_executed": False,
                            "simulation_only": True,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "FixedJoint did not retain the measured target-to-hand pose "
                "during the post-create hold; refusing to lift; inspect "
                f"{settle_failure_path}"
            )
    elif args.grasp_retention_mode == "physx-auto-attachment":
        attachment_report = create_post_close_physx_auto_attachment()
    elif args.grasp_retention_mode == "surface-gripper-attachment":
        attachment_report = create_post_close_surface_gripper_attachment()
    elif args.grasp_retention_mode == "kinematic-pose-lock":
        attachment_report = create_post_close_kinematic_pose_lock()

    target_before_lift_position, target_before_lift_orientation = get_target_world_pose()
    target_before_lift_position = np.asarray(target_before_lift_position, dtype=np.float64)
    lift_diagnostic_start_index = len(diagnostic_phases)
    execute_phase("lift", closed_finger_target)
    target_after_lift_position, target_after_lift_orientation = get_target_world_pose()
    target_after_lift_position = np.asarray(target_after_lift_position, dtype=np.float64)

    transport_executed = "transport" in replay.phase_positions
    target_after_transport_position = None
    if transport_executed:
        execute_phase("transport", closed_finger_target)
        target_after_transport_position, target_after_transport_orientation = (
            get_target_world_pose()
        )
        target_after_transport_position = np.asarray(
            target_after_transport_position, dtype=np.float64
        )

    final_phase = "transport" if transport_executed else "lift"
    final_arm_target = commands[final_phase][-1]
    for _ in range(args.hold_frames):
        panda.apply_action(
            ArticulationAction(
                joint_positions=np.concatenate((final_arm_target, closed_finger_target)),
                joint_indices=all_indices,
            )
        )
        step_world_with_attachment_sync()
        record_physics_sample("hold")
    target_held_position, target_held_orientation = get_target_world_pose()
    target_held_position = np.asarray(target_held_position, dtype=np.float64)
    measured_fingers_held = np.asarray(panda.get_joint_positions(), dtype=np.float64)[
        finger_indices
    ]
    frame_path = save_rgb(f"{len(saved_frames):02d}_{final_phase}_held")
    if frame_path:
        saved_frames.append(frame_path)

    for phase in commands:
        np.save(output / f"{phase}_commanded_joint_positions.npy", commands[phase])
        np.save(output / f"{phase}_measured_joint_positions.npy", measurements[phase])
        np.save(
            output / f"{phase}_tracking_error_rad.npy",
            measurements[phase] - commands[phase],
        )
    np.save(output / "target_settled_position_world.npy", target_settled_position)
    np.save(output / "target_before_lift_position_world.npy", target_before_lift_position)
    np.save(output / "target_after_lift_position_world.npy", target_after_lift_position)
    if target_after_transport_position is not None:
        np.save(
            output / "target_after_transport_position_world.npy",
            target_after_transport_position,
        )
    np.save(output / "target_held_position_world.npy", target_held_position)

    diagnostic_time_s = np.arange(len(diagnostic_phases), dtype=np.float64) * PHYSICS_DT_S
    diagnostic_phase_array = np.asarray(diagnostic_phases, dtype="U16")
    diagnostic_target_position_array = np.stack(diagnostic_target_positions)
    diagnostic_target_orientation_array = np.stack(diagnostic_target_orientations)
    diagnostic_finger_position_array = np.stack(diagnostic_finger_positions)
    diagnostic_finger_gap_array = np.sum(diagnostic_finger_position_array, axis=1)
    diagnostic_hand_position_array = np.stack(diagnostic_hand_positions)
    diagnostic_hand_orientation_array = np.stack(diagnostic_hand_orientations)
    diagnostic_attachment_translation_error_array = np.asarray(
        diagnostic_attachment_translation_errors, dtype=np.float64
    )
    diagnostic_attachment_orientation_error_array = np.asarray(
        diagnostic_attachment_orientation_errors, dtype=np.float64
    )
    np.save(output / "retention_time_s.npy", diagnostic_time_s)
    np.save(output / "retention_phase.npy", diagnostic_phase_array)
    np.save(
        output / "retention_target_position_world_m.npy",
        diagnostic_target_position_array,
    )
    np.save(
        output / "retention_target_orientation_world_wxyz.npy",
        diagnostic_target_orientation_array,
    )
    np.save(
        output / "retention_finger_positions_m.npy",
        diagnostic_finger_position_array,
    )
    np.save(output / "retention_finger_gap_m.npy", diagnostic_finger_gap_array)
    np.save(
        output / "retention_hand_position_world_m.npy",
        diagnostic_hand_position_array,
    )
    np.save(
        output / "retention_hand_orientation_world_wxyz.npy",
        diagnostic_hand_orientation_array,
    )
    np.save(
        output / "attachment_relative_translation_error_m.npy",
        diagnostic_attachment_translation_error_array,
    )
    np.save(
        output / "attachment_relative_orientation_error_rad.npy",
        diagnostic_attachment_orientation_error_array,
    )

    object_lift_m = float(target_after_lift_position[2] - target_before_lift_position[2])
    held_object_lift_m = float(target_held_position[2] - target_before_lift_position[2])
    transport_displacement_m = (
        float(np.linalg.norm(target_after_transport_position - target_after_lift_position))
        if target_after_transport_position is not None
        else None
    )
    minimum_clear_lift_m = float(target_settled_extent[2])
    post_close_target_positions = diagnostic_target_position_array[
        lift_diagnostic_start_index:
    ]
    post_close_finger_gaps = diagnostic_finger_gap_array[lift_diagnostic_start_index:]
    post_close_lift_m = post_close_target_positions[:, 2] - target_before_lift_position[2]
    peak_lift_local_index = int(np.argmax(post_close_lift_m))
    peak_lift_index = lift_diagnostic_start_index + peak_lift_local_index
    peak_object_lift_m = float(post_close_lift_m[peak_lift_local_index])
    lift_lost_from_peak_to_final_m = float(
        peak_object_lift_m - post_close_lift_m[-1]
    )
    phase_max_errors = {
        phase: float(np.max(np.abs(measurements[phase] - commands[phase])))
        for phase in commands
    }
    object_lifted_and_retained = bool(
        object_lift_m >= minimum_clear_lift_m
        and held_object_lift_m >= minimum_clear_lift_m
    )
    attachment_error_valid = np.isfinite(
        diagnostic_attachment_translation_error_array
    ) & np.isfinite(diagnostic_attachment_orientation_error_array)
    if np.any(attachment_error_valid):
        valid_translation_errors = diagnostic_attachment_translation_error_array[
            attachment_error_valid
        ]
        valid_orientation_errors = diagnostic_attachment_orientation_error_array[
            attachment_error_valid
        ]
        attachment_drift_summary = {
            "sample_count": int(np.count_nonzero(attachment_error_valid)),
            "maximum_translation_error_m": float(np.max(valid_translation_errors)),
            "final_translation_error_m": float(valid_translation_errors[-1]),
            "maximum_orientation_error_rad": float(np.max(valid_orientation_errors)),
            "final_orientation_error_rad": float(valid_orientation_errors[-1]),
        }
    else:
        attachment_drift_summary = None
    attachment_relative_pose_within_tolerance = (
        None
        if args.grasp_retention_mode == "physics"
        else bool(
            attachment_drift_summary is not None
            and attachment_drift_summary["maximum_translation_error_m"]
            <= ATTACHMENT_TRANSLATION_TOLERANCE_M
            and attachment_drift_summary["maximum_orientation_error_rad"]
            <= ATTACHMENT_ORIENTATION_TOLERANCE_RAD
        )
    )
    attachment_report["relative_pose_drift"] = attachment_drift_summary
    attachment_report["relative_pose_tolerance"] = {
        "maximum_translation_error_m": ATTACHMENT_TRANSLATION_TOLERANCE_M,
        "maximum_orientation_error_rad": ATTACHMENT_ORIENTATION_TOLERANCE_RAD,
        "within_tolerance": attachment_relative_pose_within_tolerance,
    }
    execution_success = bool(
        object_lifted_and_retained
        and (
            args.grasp_retention_mode == "physics"
            or attachment_relative_pose_within_tolerance is True
        )
    )
    contact_only_physical_pick_observed = (
        execution_success if args.grasp_retention_mode == "physics" else None
    )
    rigid_grasp_execution_succeeded = (
        execution_success
        if args.grasp_retention_mode == "rigid-attachment"
        else None
    )
    assumed_grasp_execution_succeeded = (
        execution_success if args.grasp_retention_mode != "physics" else None
    )
    if args.grasp_retention_mode == "physics":
        failure_status = "physical_pick_not_observed"
    elif attachment_relative_pose_within_tolerance is False:
        failure_status = "attachment_pose_not_retained"
    else:
        failure_status = "assumed_grasp_execution_failed"
    report = {
        "status": "success" if execution_success else failure_status,
        "reference": {
            "controller": "Isaac Sim 5.1 ArticulationAction position targets",
            "dynamic_target": (
                "Existing physics-ready authored USD rigid body via RigidPrim"
                if scene_usd is not None
                else "Isaac Sim DynamicCuboid with default physical properties"
            ),
            "target_bounds": (
                "Isaac Sim compute_aabb with include_children=True"
            ),
            "finger_close": (
                "Isaac Sim 5.1 articulation controller example: finger joints 7 and 8 to 0"
            ),
            "finger_open": "cuRobo franka.yml locked finger joints at 0.04 metres",
            "source_plan": str(args.plan / "grasp_lift_plan_check.json"),
            "physics_diagnostics": (
                "Isaac Sim RigidPrim runtime mass and OpenUSD DriveAPI/MaterialAPI"
            ),
        },
        "inputs": {
            "capture": str(args.capture),
            "plan": str(args.plan),
            "scene_usd": str(scene_usd) if scene_usd is not None else None,
            "scene_kind": (
                "authored_usd_scene" if scene_usd is not None else "legacy_block_scene"
            ),
            "panda_prim": args.panda_prim,
            "target_prim": target_prim_path,
            "target_rigid_body_prim": target_rigid_prim_path,
            "camera_prim": (
                args.camera_prim if scene_usd is not None else "/World/replay_camera"
            ),
        },
        "replay": {
            "physics_dt_s": PHYSICS_DT_S,
            "phase_duration_s": durations,
            "phase_command_count": {
                phase: int(value.shape[0]) for phase, value in commands.items()
            },
            "phase_maximum_tracking_error_rad": phase_max_errors,
            "maximum_close_frames": args.close_frames,
            "close_frames_executed": fixed_joint_attachment_gate[
                "close_frames_executed"
            ],
            "hold_frames": args.hold_frames,
            "transport_executed": transport_executed,
            "final_hold_phase": final_phase,
            "captured_finger_positions_m": captured_fingers.tolist(),
            "open_finger_targets_m": open_fingers.tolist(),
            "measured_fingers_before_close_m": measured_fingers_before_close.tolist(),
            "closed_finger_targets_m": closed_finger_target.tolist(),
            "measured_fingers_after_close_m": measured_fingers_after_close.tolist(),
            "measured_fingers_held_m": measured_fingers_held.tolist(),
            "saved_review_frames": saved_frames,
        },
        "execution": {
            "success_observed": execution_success,
            "evidence_kind": (
                "contact_only_physics"
                if args.grasp_retention_mode == "physics"
                else (
                    "kinematic_grasp_assumption"
                    if args.grasp_retention_mode == "kinematic-pose-lock"
                    else "physics_attachment_assumption"
                )
            ),
            "grasp_retention_mode": args.grasp_retention_mode,
            "contact_only_physical_pick_observed": (
                contact_only_physical_pick_observed
            ),
            "rigid_grasp_execution_succeeded": rigid_grasp_execution_succeeded,
            "assumed_grasp_execution_succeeded": assumed_grasp_execution_succeeded,
        },
        "post_close_attachment": attachment_report,
        "fixed_joint_attachment_gate": fixed_joint_attachment_gate,
        "physical_object": {
            "physics_apis": target_physics_apis,
            "settled_aabb_world_m": target_settled_aabb.tolist(),
            "settled_aabb_extent_m": target_settled_extent.tolist(),
            "settled_position_world_m": target_settled_position.tolist(),
            "before_lift_position_world_m": target_before_lift_position.tolist(),
            "after_lift_position_world_m": target_after_lift_position.tolist(),
            "after_transport_position_world_m": (
                target_after_transport_position.tolist()
                if target_after_transport_position is not None
                else None
            ),
            "held_position_world_m": target_held_position.tolist(),
            "lift_during_motion_m": object_lift_m,
            "lift_after_hold_m": held_object_lift_m,
            "transport_displacement_m": transport_displacement_m,
            "minimum_clear_lift_evidence_m": minimum_clear_lift_m,
            "object_lifted_and_retained": object_lifted_and_retained,
            "physical_pick_observed": contact_only_physical_pick_observed,
            "physical_pick_evidence_is_contact_only": bool(
                args.grasp_retention_mode == "physics"
            ),
        },
        "physical_parameters": {
            "finger_drive_preset": args.finger_drive_preset,
            "finger_drive_preset_definition": finger_drive_preset.to_dict(),
            "effective_requested_finger_drive_values": requested_finger_drive_values,
            "requested_finger_drive_max_force_n": args.finger_drive_max_force_n,
            "finger_drive_diagnostic_scale": args.finger_drive_scale,
            "finger_drive_scaling_policy": {
                "max_force_multiplier": args.finger_drive_scale,
                "stiffness_multiplier": args.finger_drive_scale,
                "damping_multiplier": math.sqrt(args.finger_drive_scale),
                "diagnostic_only": args.finger_drive_scale != 1.0,
                "hardware_force_calibrated": False,
            },
            "finger_drive_force_interpretation": (
                "OpenUSD linear DriveAPI max-force value; not calibrated as total "
                "Franka Hand grasping force"
            ),
            "target_effective_mass_kg": target_mass_kg,
            "target_effective_density_kg_m3": target_density_kg_m3,
            "target_inertia_flat": target_inertia,
            "finger_joint_drives": finger_drive_report,
            "target_collision_materials": target_collision_materials,
            "finger_collision_materials": finger_collision_materials,
            "fingertip_friction_override": fingertip_friction_override,
            "solver_iteration_override": solver_iteration_override,
            "null_material_coefficients_mean_no_explicit_bound_physics_material": True,
        },
        "retention_diagnostics": {
            "sample_count": int(diagnostic_time_s.size),
            "sample_period_s": PHYSICS_DT_S,
            "trace_files": {
                "time_s": str(output / "retention_time_s.npy"),
                "phase": str(output / "retention_phase.npy"),
                "target_position_world_m": str(
                    output / "retention_target_position_world_m.npy"
                ),
                "target_orientation_world_wxyz": str(
                    output / "retention_target_orientation_world_wxyz.npy"
                ),
                "finger_positions_m": str(
                    output / "retention_finger_positions_m.npy"
                ),
                "finger_gap_m": str(output / "retention_finger_gap_m.npy"),
                "hand_position_world_m": str(
                    output / "retention_hand_position_world_m.npy"
                ),
                "hand_orientation_world_wxyz": str(
                    output / "retention_hand_orientation_world_wxyz.npy"
                ),
                "attachment_relative_translation_error_m": str(
                    output / "attachment_relative_translation_error_m.npy"
                ),
                "attachment_relative_orientation_error_rad": str(
                    output / "attachment_relative_orientation_error_rad.npy"
                ),
            },
            "attachment_relative_pose_drift": attachment_drift_summary,
            "attachment_relative_pose_tolerance": {
                "maximum_translation_error_m": ATTACHMENT_TRANSLATION_TOLERANCE_M,
                "maximum_orientation_error_rad": (
                    ATTACHMENT_ORIENTATION_TOLERANCE_RAD
                ),
                "within_tolerance": attachment_relative_pose_within_tolerance,
            },
            "peak_object_lift_m": peak_object_lift_m,
            "peak_sample_index": peak_lift_index,
            "peak_time_s": float(diagnostic_time_s[peak_lift_index]),
            "peak_phase": str(diagnostic_phase_array[peak_lift_index]),
            "finger_gap_at_lift_start_m": float(post_close_finger_gaps[0]),
            "finger_gap_at_peak_lift_m": float(
                post_close_finger_gaps[peak_lift_local_index]
            ),
            "finger_gap_at_final_hold_m": float(post_close_finger_gaps[-1]),
            "lift_lost_from_peak_to_final_m": lift_lost_from_peak_to_final_m,
        },
        "automatic_checks": {
            "replay_scene_matches_capture": True,
            "target_is_physics_ready": bool(
                scene_usd is None or all(target_physics_apis.values())
            ),
            "capture_start_state_reproduced": bool(np.all(start_error <= 2e-3)),
            "all_arm_commands_finite": bool(
                all(np.isfinite(value).all() for value in commands.values())
            ),
            "all_arm_measurements_finite": bool(
                all(np.isfinite(value).all() for value in measurements.values())
            ),
            "finger_measurements_finite": bool(
                np.isfinite(measured_fingers_before_close).all()
                and np.isfinite(measured_fingers_after_close).all()
                and np.isfinite(measured_fingers_held).all()
            ),
            "retention_trace_is_finite": bool(
                np.isfinite(diagnostic_target_position_array).all()
                and np.isfinite(diagnostic_target_orientation_array).all()
                and np.isfinite(diagnostic_finger_position_array).all()
                and np.isfinite(diagnostic_hand_position_array).all()
                and np.isfinite(diagnostic_hand_orientation_array).all()
            ),
            "object_lifted_by_at_least_one_object_height": (
                object_lifted_and_retained
            ),
            "object_remained_lifted_after_transport": bool(
                not transport_executed
                or held_object_lift_m >= minimum_clear_lift_m
            ),
            "attachment_relative_pose_within_tolerance": (
                attachment_relative_pose_within_tolerance
            ),
        },
        "safety": {
            "simulation_only": True,
            "target_started_as_dynamic": True,
            "target_switched_to_kinematic_after_close": bool(
                args.grasp_retention_mode == "kinematic-pose-lock"
            ),
            "object_not_fixed_to_gripper": bool(
                args.grasp_retention_mode == "physics"
            ),
            "rigid_grasp_assumption_applied_after_close": bool(
                attachment_report.get("applied") is True
            ),
            "grasp_slip_evaluated": bool(args.grasp_retention_mode == "physics"),
            "physical_gripper_close_commanded": True,
            "physical_contact_monitoring_automated": bool(
                args.grasp_retention_mode == "rigid-attachment"
            ),
            "bilateral_target_contact_required_before_fixed_joint": bool(
                args.grasp_retention_mode == "rigid-attachment"
            ),
            "first_lift_held_object_collision_checked_by_curobo": False,
            "attached_transport_planned_and_collision_checked": bool(
                transport_executed
                and replay.plan_report.get("safety", {}).get(
                    "held_object_collision_checked_during_transport"
                )
                is True
            ),
            "transport_executed_with_closed_gripper": transport_executed,
            "human_or_receiver_collision_model_present": False,
            "handover_release_executed": False,
            "manual_review_required": True,
            "safe_for_real_robot_execution": False,
        },
    }
    report_path = output / "grasp_lift_replay_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"object lift: {object_lift_m:.6g} m; held after {args.hold_frames} frames: "
        f"{held_object_lift_m:.6g} m"
    )
    print(f"saved: {report_path}")
    if not execution_success:
        raise SystemExit(2)
except Exception as exc:
    if isinstance(exc, SystemExit):
        raise
    failure_traceback = traceback.format_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grasp_lift_replay_check.json").write_text(
        json.dumps(
            {
                "status": "failure",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": failure_traceback,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(failure_traceback, file=sys.stderr, flush=True)
    raise
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
