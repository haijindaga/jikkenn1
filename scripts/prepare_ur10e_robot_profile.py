#!/usr/bin/env python3
"""Build and adapt NVIDIA's UR10e + Robotiq 2F-140 cuRobo profile.

This intentionally delegates URDF merging and collision-sphere fitting to
GraspGenX's official ``end2end/build_ur10e_gripper.py``.  The only local
adaptation adds the contact-link and attached-object contract already used by
the pinned cuRobo Franka profile.  Run it with GraspGenX's ``uv`` Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ARM_LINKS = {
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
}
TOOL_FRAME = "robotiq_arg2f_base_link"
CONTACT_LINKS = ("left_inner_finger_pad", "right_inner_finger_pad")
SOURCE_STEM = "ur10e_robotiq_2f_140"
# Isaac Sim 5.1's official Robot Assembler recipe rotates the attached
# Robotiq 2F-140 by Z +90 degrees.  The GraspGenX builder default is explicitly
# documented as a first guess and produced a measured 180-degree frame error
# against the assembled Isaac asset.  Composing that measured offset with the
# builder default gives this exact URDF RPY (Rz @ Ry @ Rx convention).
REVIEWED_MOUNT_RPY = (0.0, 0.0, math.pi / 2.0)
REVIEWED_MOUNT_XYZ = (0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graspgenx-root", type=Path, default=Path("/home/suzutaro/GraspGenX")
    )
    parser.add_argument("--skip-official-build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tool_mount(urdf_path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read the unique fixed tool0-to-Robotiq mount from a generated URDF."""
    root = ET.parse(urdf_path).getroot()
    matches = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            joint.get("type") == "fixed"
            and parent is not None
            and parent.get("link") == "tool0"
            and child is not None
            and child.get("link") == TOOL_FRAME
        ):
            matches.append(joint)
    if len(matches) != 1:
        raise ValueError(
            f"expected one fixed tool0-to-{TOOL_FRAME} joint, found {len(matches)}"
        )
    origin = matches[0].find("origin")
    if origin is None:
        raise ValueError("reviewed tool mount joint has no origin")
    rpy = tuple(float(value) for value in origin.get("rpy", "").split())
    xyz = tuple(float(value) for value in origin.get("xyz", "").split())
    if len(rpy) != 3 or len(xyz) != 3:
        raise ValueError(f"invalid tool mount origin: rpy={rpy}, xyz={xyz}")
    return rpy, xyz


def verify_reviewed_tool_mount(urdf_path: Path) -> dict[str, object]:
    rpy, xyz = read_tool_mount(urdf_path)
    tolerance = 1e-8
    rpy_error = max(abs(a - b) for a, b in zip(rpy, REVIEWED_MOUNT_RPY))
    xyz_error = max(abs(a - b) for a, b in zip(xyz, REVIEWED_MOUNT_XYZ))
    if rpy_error > tolerance or xyz_error > tolerance:
        raise ValueError(
            "generated URDF does not contain the reviewed Isaac-compatible "
            f"Robotiq mount: rpy={rpy}, xyz={xyz}"
        )
    return {
        "parent_link": "tool0",
        "child_link": TOOL_FRAME,
        "rpy_rad": list(rpy),
        "xyz_m": list(xyz),
        "maximum_error": max(rpy_error, xyz_error),
        "tolerance": tolerance,
        "passed": True,
    }


def adapt_config(source: dict) -> dict:
    """Add only the reviewed grasp/attachment fields to the official config."""
    robot_cfg = source.get("robot_cfg")
    if not isinstance(robot_cfg, dict):
        raise ValueError("official config has no robot_cfg mapping")
    kin = robot_cfg.get("kinematics")
    if not isinstance(kin, dict):
        raise ValueError("official config has no robot_cfg.kinematics mapping")

    collision_links = list(kin.get("collision_link_names", ()))
    required = {TOOL_FRAME, *CONTACT_LINKS}
    missing = sorted(required - set(collision_links))
    if missing:
        raise ValueError(
            "official Robotiq profile is missing required collision links: "
            f"{missing}; do not guess replacement names"
        )
    cspace_names = list(kin.get("cspace", {}).get("joint_names", ()))
    expected_arm = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    if cspace_names[:6] != expected_arm or "finger_joint" not in cspace_names:
        raise ValueError(f"unexpected official cspace joints: {cspace_names}")

    kin["tool_frames"] = [TOOL_FRAME]
    kin["grasp_contact_link_names"] = [TOOL_FRAME, *CONTACT_LINKS, "attached_object"]
    if "attached_object" not in collision_links:
        collision_links.append("attached_object")
    kin["collision_link_names"] = collision_links
    kin.setdefault("extra_collision_spheres", {})["attached_object"] = 4
    kin.setdefault("extra_links", {})["attached_object"] = {
        "link_name": "attached_object",
        "parent_link_name": TOOL_FRAME,
        "joint_name": "attach_joint",
        "joint_type": "FIXED",
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
    }
    kin.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
    ignores = kin.setdefault("self_collision_ignore", {})
    # The attached proxy represents an object already enclosed by the hand. It
    # may overlap the gripper, but remains checked against the UR arm and world.
    gripper_links = sorted(set(collision_links) - ARM_LINKS - {"attached_object"})
    for link in gripper_links:
        values = set(ignores.get(link, ()))
        values.add("attached_object")
        ignores[link] = sorted(values)
    ignores["attached_object"] = gripper_links
    return source


def main() -> int:
    args = parse_args()
    root = args.graspgenx_root.expanduser().resolve()
    builder = root / "end2end" / "build_ur10e_gripper.py"
    assets = root / "end2end" / "curobo_assets"
    urdf = assets / f"{SOURCE_STEM}.urdf"
    source = assets / f"{SOURCE_STEM}.yml"
    output = assets / f"{SOURCE_STEM}.jikkenn1.yml"
    report_path = assets / f"{SOURCE_STEM}.jikkenn1.json"
    if not builder.is_file():
        raise FileNotFoundError(f"official GraspGenX builder does not exist: {builder}")
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"adapted profile already exists: {output}; pass --overwrite to rebuild"
        )
    if not args.skip_official_build:
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--gripper",
                "robotiq_2f_140",
                "--mount_rpy",
                *(f"{value:.17g}" for value in REVIEWED_MOUNT_RPY),
                "--mount_xyz",
                *(f"{value:.17g}" for value in REVIEWED_MOUNT_XYZ),
            ],
            cwd=root,
            check=True,
        )
    if not urdf.is_file():
        raise FileNotFoundError(f"official builder URDF does not exist: {urdf}")
    mount_check = verify_reviewed_tool_mount(urdf)
    if not source.is_file():
        raise FileNotFoundError(f"official builder output does not exist: {source}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; run with GraspGenX's uv environment") from exc
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    adapted = adapt_config(config)
    output.write_text(
        yaml.safe_dump(adapted, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )
    report = {
        "status": "success",
        "profile": "ur10e_robotiq_2f_140",
        "reference": {
            "builder": str(builder),
            "builder_repository": "https://github.com/NVlabs/GraspGenX",
            "isaac_mount_reference": (
                "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/"
                "robot_setup_tutorials/tutorial_import_assemble_manipulator.html"
            ),
            "curobo_contract": "pinned cuRobo franka.yml attached-object fields",
        },
        "generated_urdf": {"path": str(urdf), "sha256": _sha256(urdf)},
        "source": {"path": str(source), "sha256": _sha256(source)},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "adaptations": {
            "tool_frame": TOOL_FRAME,
            "contact_links": list(CONTACT_LINKS),
            "attached_object_spheres": 4,
            "attached_object_parent": TOOL_FRAME,
            "builder_geometry_modified": False,
            "builder_default_mount_overridden": True,
            "reviewed_mount": mount_check,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
