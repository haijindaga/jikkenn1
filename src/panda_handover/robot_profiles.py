"""Reviewed robot/gripper profiles shared by the simulation pipeline.

The profile boundary follows NVIDIA GraspGenX's ``end2end/robot_profiles.py``
design: perception stays robot-independent, while kinematics, grasp-frame
conversion, gripper geometry, and execution names belong to one explicit
robot/gripper profile.  Keeping these values together prevents a run from
silently mixing (for example) a Panda grasp mesh with a UR10e trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RobotProfile:
    """Immutable robot/gripper contract used across pipeline stages."""

    name: str
    robot_family: str
    gripper_name: str
    default_robot_prim: str
    curobo_robot_config: str
    tool_frame: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    gripper_open: tuple[float, ...]
    gripper_closed: tuple[float, ...]
    gripper_joint_position_unit: str
    observation_arm_joint_positions: tuple[float, ...] | None
    hand_rigid_body_link: str
    contact_link_names: tuple[str, ...]
    grasp_to_tool_transform: tuple[tuple[float, float, float, float], ...]
    replay_drive_preset: str
    isaac_asset_relative_path: str | None
    isaac_gripper_variant: str | None
    implementation_reference: str

    def validate(self) -> None:
        if not self.name or not self.gripper_name or not self.tool_frame:
            raise ValueError("robot profile names must not be empty")
        if not self.default_robot_prim.startswith("/"):
            raise ValueError("default_robot_prim must be an absolute USD path")
        if not self.arm_joint_names or len(set(self.arm_joint_names)) != len(
            self.arm_joint_names
        ):
            raise ValueError("arm_joint_names must be non-empty and unique")
        if not self.gripper_joint_names or len(set(self.gripper_joint_names)) != len(
            self.gripper_joint_names
        ):
            raise ValueError("gripper_joint_names must be non-empty and unique")
        if len(self.gripper_open) != len(self.gripper_joint_names) or len(
            self.gripper_closed
        ) != len(self.gripper_joint_names):
            raise ValueError("gripper positions must match gripper_joint_names")
        if self.observation_arm_joint_positions is not None and len(
            self.observation_arm_joint_positions
        ) != len(self.arm_joint_names):
            raise ValueError(
                "observation_arm_joint_positions must match arm_joint_names"
            )
        if self.gripper_joint_position_unit not in {"metre", "radian"}:
            raise ValueError("unsupported gripper_joint_position_unit")
        if len(self.contact_link_names) != 2:
            raise ValueError("the current bilateral-contact gate requires two links")
        if len(self.grasp_to_tool_transform) != 4 or any(
            len(row) != 4 for row in self.grasp_to_tool_transform
        ):
            raise ValueError("grasp_to_tool_transform must be 4x4")

    def resolve_curobo_config(self, graspgenx_root: str | Path) -> str:
        """Resolve external generated configs while preserving bare built-ins."""
        if not self.curobo_robot_config.startswith("${GRASPGENX}/"):
            return self.curobo_robot_config
        relative = self.curobo_robot_config.removeprefix("${GRASPGENX}/")
        return str(Path(graspgenx_root).expanduser().resolve() / relative)

    def report(self, *, resolved_curobo_config: str | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot_family": self.robot_family,
            "gripper_name": self.gripper_name,
            "default_robot_prim": self.default_robot_prim,
            "curobo_robot_config": resolved_curobo_config
            or self.curobo_robot_config,
            "tool_frame": self.tool_frame,
            "arm_joint_names": list(self.arm_joint_names),
            "gripper_joint_names": list(self.gripper_joint_names),
            "gripper_open": list(self.gripper_open),
            "gripper_closed": list(self.gripper_closed),
            "gripper_joint_position_unit": self.gripper_joint_position_unit,
            "observation_arm_joint_positions": (
                list(self.observation_arm_joint_positions)
                if self.observation_arm_joint_positions is not None
                else None
            ),
            "hand_rigid_body_link": self.hand_rigid_body_link,
            "contact_link_names": list(self.contact_link_names),
            "replay_drive_preset": self.replay_drive_preset,
            "isaac_asset_relative_path": self.isaac_asset_relative_path,
            "isaac_gripper_variant": self.isaac_gripper_variant,
            "implementation_reference": self.implementation_reference,
        }


FRANKA_PANDA = RobotProfile(
    name="franka_panda",
    robot_family="franka_panda",
    gripper_name="franka_panda",
    default_robot_prim="/World/Panda",
    curobo_robot_config="franka.yml",
    tool_frame="panda_hand",
    arm_joint_names=tuple(f"panda_joint{index}" for index in range(1, 8)),
    gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
    gripper_open=(0.04, 0.04),
    gripper_closed=(0.0, 0.0),
    gripper_joint_position_unit="metre",
    observation_arm_joint_positions=None,
    hand_rigid_body_link="panda_hand",
    contact_link_names=("panda_leftfinger", "panda_rightfinger"),
    grasp_to_tool_transform=(
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    replay_drive_preset="isaaclab-franka",
    isaac_asset_relative_path=None,
    isaac_gripper_variant=None,
    implementation_reference=(
        "NVIDIA GraspGenX end2end/robots/franka_panda.yaml and cuRobo franka.yml"
    ),
)


UR10E_ROBOTIQ_2F_140 = RobotProfile(
    name="ur10e_robotiq_2f_140",
    robot_family="ur10e",
    gripper_name="robotiq_2f_140",
    default_robot_prim="/World/UR10e",
    # Produced once by scripts/prepare_ur10e_robot_profile.py from NVIDIA's
    # build_ur10e_gripper.py.  It remains outside this repository with the
    # source meshes and records its provenance in a sidecar JSON report.
    curobo_robot_config=(
        "${GRASPGENX}/end2end/curobo_assets/ur10e_robotiq_2f_140.jikkenn1.yml"
    ),
    # GraspGenX's reviewed UR10e+2F-140 profile plans the gripper base itself.
    # Its canonical gripper root is therefore the grasp frame (identity below).
    tool_frame="robotiq_arg2f_base_link",
    arm_joint_names=(
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    gripper_joint_names=("finger_joint",),
    gripper_open=(0.0,),
    gripper_closed=(0.7,),
    gripper_joint_position_unit="radian",
    # Isaac Lab's official UR10e configuration uses this initial arm pose.
    # With this project's table on robot-base +X, shoulder_pan=pi parks the
    # arm away from the RGB-D observation region.
    observation_arm_joint_positions=(
        3.141592653589793,
        -1.5707963267948966,
        1.5707963267948966,
        -1.5707963267948966,
        -1.5707963267948966,
        0.0,
    ),
    hand_rigid_body_link="robotiq_arg2f_base_link",
    contact_link_names=("left_inner_finger_pad", "right_inner_finger_pad"),
    grasp_to_tool_transform=(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    # Isaac Sim 5.1's configured asset is the source of its own drive gains.
    replay_drive_preset="authored-usd",
    isaac_asset_relative_path=(
        "Isaac/Samples/Rigging/Manipulator/import_manipulator/"
        "ur10e/ur/ur_gripper.usd"
    ),
    # Exact variant spelling from Isaac Sim 5.1's assembled UR10e tutorial
    # asset.  Keep this case-sensitive so an asset revision fails loudly.
    isaac_gripper_variant="robotiq_2f_140",
    implementation_reference=(
        "NVIDIA GraspGenX UR10eRobotiq2F140Profile plus Isaac Sim 5.1 "
        "UR10e/Robotiq 2F-140 manipulator tutorials"
    ),
)


ROBOT_PROFILES = {
    profile.name: profile for profile in (FRANKA_PANDA, UR10E_ROBOTIQ_2F_140)
}


def robot_profile_names() -> tuple[str, ...]:
    return tuple(ROBOT_PROFILES)


def get_robot_profile(name: str) -> RobotProfile:
    try:
        profile = ROBOT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown robot profile {name!r}; choose one of {robot_profile_names()}"
        ) from exc
    profile.validate()
    return profile
