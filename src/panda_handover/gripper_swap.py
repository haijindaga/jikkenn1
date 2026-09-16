"""Contracts for a separate official Panda/Robotiq variant smoke test."""

import math
import numpy as np


PANDA_ASSET = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
GRIPPER_VARIANT = "Robotiq_2F_85"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


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
    if int(row["type"]) != 1 or not (
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
    return {
        "joint": "finger_joint", "index": index,
        "mode": "position" if stiffness > 0 else "velocity",
        "open_rad": lower, "closed_rad": upper,
        "diagnostic_speed_rad_s": min(max_velocity, 2 * (upper - lower) / phase_duration_s),
        "authored_stiffness": stiffness, "authored_damping": damping,
        "authored_max_effort": float(row["maxEffort"]),
    }


def bounded_gripper_velocity(current, target, speed, dt):
    if not all(map(math.isfinite, (current, target, speed, dt))) or dt <= 0 or speed <= 0:
        raise ValueError("Invalid gripper velocity command inputs")
    if abs(target - current) < 1e-3:
        return 0.0
    return float(np.clip((target - current) / dt, -speed, speed))
