#!/usr/bin/env python3
"""Prepare the pinned NVIDIA UR10e + Robotiq 2F-140 cuRobo profile.

The kinematics and collision model come from NVIDIA Isaac ROS cuMotion's
validated UR10e/Robotiq pair. cuRobo's own XRDF converter creates the native
robot configuration; this script only adds the four-sphere runtime attachment
capacity and contact-link list needed by this project's grasp planner.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SOURCE_REPOSITORY = "https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion"
SOURCE_BRANCH = "release-3.2"
SOURCE_COMMIT = "dbaa7e8264f6314f8baca516511414186ad1105d"
SOURCE_DIRECTORY = Path("config/robots/isaac_ros_cumotion_release_3_2")
SOURCE_STEM = "ur10e_robotiq_2f_140"
OFFICIAL_URDF_SHA256 = (
    "47ea5d97ced93af17fe165f01a65058bdc913adcde22b1445598f34b1299ed4f"
)
OFFICIAL_XRDF_SHA256 = (
    "750b01a70755c737ef0d0e04d1927c63358ddef54473d999c81682444d8fcb52"
)

TOOL_FRAME = "grasp_frame"
GRIPPER_BASE_LINK = "robotiq_base_link"
CONTACT_LINKS = ("left_inner_finger_pad", "right_inner_finger_pad")
ATTACHED_OBJECT_SPHERE_COUNT = 4
EXPECTED_ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graspgenx-root", type=Path, default=Path("/home/suzutaro/GraspGenX")
    )
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"pinned official source does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"pinned official source hash changed for {path}: {actual} != {expected}"
        )
    return actual


def _unique_fixed_joint(
    root: ET.Element, *, parent_link: str, child_link: str
) -> ET.Element:
    matches = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            joint.get("type") == "fixed"
            and parent is not None
            and parent.get("link") == parent_link
            and child is not None
            and child.get("link") == child_link
        ):
            matches.append(joint)
    if len(matches) != 1:
        raise ValueError(
            f"expected one fixed {parent_link}-to-{child_link} joint, "
            f"found {len(matches)}"
        )
    return matches[0]


def _joint_origin(joint: ET.Element) -> tuple[tuple[float, ...], tuple[float, ...]]:
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"fixed joint {joint.get('name')} has no origin")
    rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
    xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
    if len(rpy) != 3 or len(xyz) != 3:
        raise ValueError(f"invalid fixed-joint origin: rpy={rpy}, xyz={xyz}")
    return rpy, xyz


def inspect_official_urdf(urdf_path: Path) -> dict[str, object]:
    """Verify the two fixed transforms on which the planning frame depends."""
    root = ET.parse(urdf_path).getroot()
    mount_rpy, mount_xyz = _joint_origin(
        _unique_fixed_joint(root, parent_link="tool0", child_link=GRIPPER_BASE_LINK)
    )
    grasp_rpy, grasp_xyz = _joint_origin(
        _unique_fixed_joint(root, parent_link="gripper_frame", child_link=TOOL_FRAME)
    )
    expected_mount_rpy = (0.0, 0.0, 1.57)
    expected_zero = (0.0, 0.0, 0.0)
    expected_grasp_xyz = (0.0, 0.0, 0.20)
    tolerance = 1e-9
    comparisons = (
        (mount_rpy, expected_mount_rpy),
        (mount_xyz, expected_zero),
        (grasp_rpy, expected_zero),
        (grasp_xyz, expected_grasp_xyz),
    )
    if any(
        abs(actual - expected) > tolerance
        for actual_values, expected_values in comparisons
        for actual, expected in zip(actual_values, expected_values)
    ):
        raise ValueError(
            "pinned official URDF tool transforms changed; refusing to guess a replacement"
        )
    return {
        "tool0_to_robotiq_base_link": {
            "rpy_rad": list(mount_rpy),
            "xyz_m": list(mount_xyz),
        },
        "gripper_frame_to_grasp_frame": {
            "rpy_rad": list(grasp_rpy),
            "xyz_m": list(grasp_xyz),
        },
        "passed": True,
    }


def inspect_official_xrdf(xrdf: dict) -> dict[str, object]:
    if xrdf.get("format") != "xrdf" or float(xrdf.get("format_version", -1)) != 1.0:
        raise ValueError("official robot description is not XRDF 1.0")
    if xrdf.get("tool_frames") != [TOOL_FRAME]:
        raise ValueError(f"official XRDF tool frame changed: {xrdf.get('tool_frames')}")
    if tuple(xrdf.get("cspace", {}).get("joint_names", ())) != EXPECTED_ARM_JOINTS:
        raise ValueError("official XRDF active arm joints changed")
    attached = [
        entry["add_frame"]
        for entry in xrdf.get("modifiers", ())
        if isinstance(entry, dict)
        and isinstance(entry.get("add_frame"), dict)
        and entry["add_frame"].get("frame_name") == "attached_object"
    ]
    if len(attached) != 1 or attached[0].get("parent_frame_name") != TOOL_FRAME:
        raise ValueError("official XRDF attached_object is not fixed below grasp_frame")
    collision_name = xrdf.get("collision", {}).get("geometry")
    spheres = xrdf.get("geometry", {}).get(collision_name, {}).get("spheres", {})
    required = {GRIPPER_BASE_LINK, *CONTACT_LINKS, "attached_object"}
    missing = sorted(required - set(spheres))
    if missing:
        raise ValueError(f"official XRDF is missing collision spheres for {missing}")
    return {
        "tool_frame": TOOL_FRAME,
        "active_arm_joints": list(EXPECTED_ARM_JOINTS),
        "attached_object_parent": TOOL_FRAME,
        "collision_geometry": collision_name,
        "passed": True,
    }


def adapt_config(source: dict) -> dict:
    """Add only runtime fields absent from the official XRDF conversion."""
    result = deepcopy(source)
    robot_cfg = result.get("robot_cfg")
    if not isinstance(robot_cfg, dict):
        raise ValueError("converted official config has no robot_cfg mapping")
    kin = robot_cfg.get("kinematics")
    if not isinstance(kin, dict):
        raise ValueError("converted official config has no kinematics mapping")
    if kin.get("tool_frames") != [TOOL_FRAME]:
        raise ValueError(f"converted tool frame changed: {kin.get('tool_frames')}")
    cspace_names = tuple(kin.get("cspace", {}).get("joint_names", ()))
    if cspace_names[: len(EXPECTED_ARM_JOINTS)] != EXPECTED_ARM_JOINTS:
        raise ValueError(f"unexpected converted cspace joints: {cspace_names}")
    collision_links = set(kin.get("collision_link_names", ()))
    required = {GRIPPER_BASE_LINK, *CONTACT_LINKS, "attached_object"}
    missing = sorted(required - collision_links)
    if missing:
        raise ValueError(f"converted official profile is missing collision links: {missing}")
    attached = kin.get("extra_links", {}).get("attached_object", {})
    if attached.get("parent_link_name") != TOOL_FRAME:
        raise ValueError("converted attached_object parent is not grasp_frame")

    kin["grasp_contact_link_names"] = [
        GRIPPER_BASE_LINK,
        *CONTACT_LINKS,
        "attached_object",
    ]
    kin.setdefault("extra_collision_spheres", {})[
        "attached_object"
    ] = ATTACHED_OBJECT_SPHERE_COUNT
    return result


def convert_official_xrdf(*, xrdf_path: Path, urdf_path: Path) -> dict:
    """Use the pinned cuRobo implementation instead of a local converter."""
    try:
        from curobo._src.types.content_path import ContentPath
        from curobo._src.util.xrdf_util import convert_xrdf_to_curobo
    except ImportError as exc:
        raise RuntimeError(
            "the pinned cuRobo XRDF converter is required; run with GraspGenX's uv environment"
        ) from exc
    content_path = ContentPath(
        robot_xrdf_absolute_path=str(xrdf_path),
        robot_urdf_absolute_path=str(urdf_path),
        robot_asset_absolute_path=str(urdf_path.parent),
    )
    return convert_xrdf_to_curobo(content_path)


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    source_directory = repository_root / SOURCE_DIRECTORY
    urdf = source_directory / f"{SOURCE_STEM}.urdf"
    xrdf = source_directory / f"{SOURCE_STEM}.xrdf"
    license_path = source_directory / "LICENSE"
    urdf_hash = _require_hash(urdf, OFFICIAL_URDF_SHA256)
    xrdf_hash = _require_hash(xrdf, OFFICIAL_XRDF_SHA256)
    if not license_path.is_file():
        raise FileNotFoundError(f"official source license is missing: {license_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required in the GraspGenX environment") from exc
    xrdf_data = yaml.safe_load(xrdf.read_text(encoding="utf-8"))
    urdf_check = inspect_official_urdf(urdf)
    xrdf_check = inspect_official_xrdf(xrdf_data)
    config = adapt_config(convert_official_xrdf(xrdf_path=xrdf, urdf_path=urdf))

    output_directory = (
        args.graspgenx_root.expanduser().resolve() / "end2end" / "curobo_assets"
    )
    output = output_directory / f"{SOURCE_STEM}.jikkenn1.yml"
    report_path = output_directory / f"{SOURCE_STEM}.jikkenn1.json"
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"adapted profile already exists: {output}; pass --overwrite to rebuild"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )
    report = {
        "status": "success",
        "profile": SOURCE_STEM,
        "reference": {
            "repository": SOURCE_REPOSITORY,
            "branch": SOURCE_BRANCH,
            "commit": SOURCE_COMMIT,
            "license": "Apache-2.0",
            "license_path": str(license_path),
            "converter": "pinned cuRobo convert_xrdf_to_curobo",
        },
        "official_sources": {
            "urdf": {"path": str(urdf), "sha256": urdf_hash},
            "xrdf": {"path": str(xrdf), "sha256": xrdf_hash},
        },
        "official_contract_checks": {"urdf": urdf_check, "xrdf": xrdf_check},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "local_runtime_adaptations": {
            "grasp_contact_link_names": [
                GRIPPER_BASE_LINK,
                *CONTACT_LINKS,
                "attached_object",
            ],
            "attached_object_sphere_capacity": ATTACHED_OBJECT_SPHERE_COUNT,
            "kinematics_reestimated": False,
            "collision_spheres_refitted": False,
            "self_collision_pairs_modified": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
