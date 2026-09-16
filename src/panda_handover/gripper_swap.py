"""Contracts for a separate official Panda/Robotiq variant smoke test."""

import math
import numpy as np


PANDA_ASSET = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
GRIPPER_VARIANT = "Robotiq_2F_85"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def authored_stage_metadata(stage):
    """Read authored root-layer metadata using SdfSpec's supported API.

    Usd.Stage does not expose GetAllMetadata. Pseudo-root structural fields
    describe layer contents/composition and must not be copied as metadata.
    """
    root = stage.GetRootLayer().pseudoRoot
    excluded = {"primChildren", "propertyChildren", "subLayers", "subLayerOffsets"}
    return {str(key): root.GetInfo(key) for key in root.ListInfoKeys()
            if str(key) not in excluded}


def create_variant_scene(source_stage, scene_path):
    """Author only stage metadata and a variant opinion over the original scene.

    References stay anchored to their original layers, so relative object paths
    are not reinterpreted relative to the experiment output directory.
    """
    from pxr import Sdf, Usd, UsdGeom

    robot_path = "/World/Panda"
    original = source_stage.GetPrimAtPath(robot_path)
    if not original or not original.IsActive():
        raise RuntimeError("Input scene lacks active /World/Panda")
    if UsdGeom.GetStageMetersPerUnit(source_stage) != 1.0:
        raise RuntimeError("Input scene must use metres")
    # World assumes a Z-up tabletop. Do not silently rotate a Y-up source.
    if UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Input tabletop scene must be Z-up; source was not changed")
    layer = Sdf.Layer.CreateNew(str(scene_path))
    layer.subLayerPaths = [source_stage.GetRootLayer().realPath]
    stage = Usd.Stage.Open(layer)
    # Stage metadata is read from the root/session layer, not from sublayers.
    for key, value in authored_stage_metadata(source_stage).items():
        layer.pseudoRoot.SetInfo(key, value)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    available = select_official_gripper(stage.GetPrimAtPath(robot_path))
    layer.Save()
    return stage, robot_path, available


def scene_invariants(stage):
    """Static composition check before simulation; excludes only robot children."""
    from pxr import UsdGeom

    rows = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Panda/"):
            continue
        rows.append((path, str(prim.GetTypeName()),
                     tuple((str(a.GetName()), str(a.Get()),
                            tuple((t, str(a.Get(t))) for t in a.GetTimeSamples()))
                           for a in prim.GetAttributes()),
                     tuple((str(r.GetName()), tuple(map(str, r.GetTargets())))
                           for r in prim.GetRelationships())))
    return {
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "stage_metadata": {k: str(v) for k, v in authored_stage_metadata(stage).items()},
        "non_robot_scene": rows,
    }


def select_official_gripper(prim):
    variants = prim.GetVariantSet("Gripper")
    available = list(variants.GetVariantNames())
    if GRIPPER_VARIANT not in available:
        raise RuntimeError(f"Official Panda asset lacks {GRIPPER_VARIANT}: {available}")
    if not variants.SetVariantSelection(GRIPPER_VARIANT):
        raise RuntimeError("Could not select official Robotiq gripper variant")
    if variants.GetVariantSelection() != GRIPPER_VARIANT:
        raise RuntimeError("Robotiq variant selection did not take effect")
    return available


def captured_arm_positions(names, positions):
    positions = np.asarray(positions, dtype=float)
    if len(set(names)) != len(names) or positions.shape != (len(names),):
        raise ValueError("Capture joint names/positions are inconsistent")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Capture contains non-finite joint positions")
    missing = set(ARM_JOINTS) - set(names)
    if missing:
        raise ValueError(f"Capture lacks Panda arm joints: {sorted(missing)}")
    return np.asarray([positions[list(names).index(name)] for name in ARM_JOINTS])


def gripper_control_spec(names, properties, phase_duration_s):
    if phase_duration_s <= 0 or not math.isfinite(phase_duration_s):
        raise ValueError("Phase duration must be positive and finite")
    if list(names).count("finger_joint") != 1:
        raise RuntimeError(f"Expected one official finger_joint master DOF; found {names}")
    index = list(names).index("finger_joint")
    row = properties[index]
    lower, upper = float(row["lower"]), float(row["upper"])
    # Isaac Sim 5.1 tensor API: Rotation=0, Translation=1 (not legacy DC enums).
    if int(row["type"]) != 0 or not (
        math.isfinite(lower) and math.isfinite(upper) and lower < upper
    ):
        raise RuntimeError("finger_joint must have finite revolute limits")
    stiffness, damping = float(row["stiffness"]), float(row["damping"])
    if stiffness < 0 or damping < 0 or not all(map(math.isfinite, (stiffness, damping))):
        raise RuntimeError("Invalid authored finger drive gains")
    if stiffness == 0 and damping == 0:
        raise RuntimeError("Official finger drive has no position/velocity gain")
    # The official Robotiq joint opens at zero and closes at increasing angle.
    if abs(lower) > 1e-3 or upper <= 0:
        raise RuntimeError("Unexpected official finger_joint limits; inspect before driving")
    max_velocity = float(row["maxVelocity"])
    if not math.isfinite(max_velocity) or max_velocity <= 0:
        raise RuntimeError("Invalid authored finger velocity limit")
    max_effort = float(row["maxEffort"])
    if not math.isfinite(max_effort) or max_effort <= 0:
        raise RuntimeError("Invalid authored finger effort limit")
    return {
        "joint": "finger_joint", "index": index,
        "mode": "position" if stiffness > 0 else "velocity",
        "open_rad": lower, "closed_rad": upper,
        # An explicitly labelled smoke-test speed, NOT a new feedback solver.
        "diagnostic_speed_rad_s": min(max_velocity, (upper - lower) / phase_duration_s),
        "authored_stiffness": stiffness, "authored_damping": damping,
        "authored_max_effort": max_effort,
    }


def bounded_gripper_velocity(current, target, speed, dt):
    """Constant signed velocity until the limit; no proportional controller."""
    if not all(map(math.isfinite, (current, target, speed, dt))) or dt <= 0 or speed <= 0:
        raise ValueError("Invalid gripper velocity command inputs")
    if abs(target - current) <= max(1e-3, speed * dt):
        return 0.0
    return math.copysign(speed, target - current)
