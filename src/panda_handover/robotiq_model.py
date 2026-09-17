"""Evidence-gated composition of source URDFs, for FK inspection only.

This is not an Isaac asset, a dynamics controller or a collision-ready profile.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .gripper_swap import ARM_JOINTS
from .robot_model_evidence import compare_arm_poses


def file_identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        header = stream.read(256)
        if not header or b"version https://git-lfs.github.com/spec/v1" in header:
            raise ValueError(f"Empty file or Git LFS pointer, not an asset: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def body_transform(evidence, name):
    matches = [b for b in evidence["rigid_bodies"] if b["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one rigid {name}, found {len(matches)}")
    value = np.asarray(matches[0]["T_world_body"], dtype=float)
    if (value.shape != (4, 4) or not np.all(np.isfinite(value))
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6)):
        raise ValueError(f"Invalid rigid transform: {name}")
    return value


def installed_mount(evidence):
    if evidence.get("open_close_checks_passed") is not True:
        raise ValueError("A completed installed open/close check is required")
    hand = body_transform(evidence, "panda_hand")
    base = body_transform(evidence, "base_link")
    # Identity is a mounting hypothesis supported by the official assembled USD,
    # not an arbitrary compensating rotation. Never fit an offset to one pose.
    delta = np.linalg.inv(hand) @ base
    comparison = compare_arm_poses({"mount": np.eye(4)}, {"mount": delta},
                                  translation_tolerance=1e-5, rotation_tolerance=1e-4)
    if not comparison["mount"]["passed"]:
        raise ValueError("Installed Robotiq base is not coincident with panda_hand; review the mount")
    paths = {name: next(b["path"] for b in evidence["rigid_bodies"] if b["name"] == name)
             for name in ("panda_hand", "base_link")}
    joints = [j for j in evidence["joints"] if j["type"] == "PhysicsFixedJoint"
              and set(j["body0"] + j["body1"]) == set(paths.values())]
    if len(joints) != 1:
        raise ValueError("Expected the official fixed mounting joint between hand and base")
    return {"T_hand_gripper_base": np.eye(4).tolist(),
            "measured_T_hand_gripper_base": delta.tolist(),
            "mount_joint": joints[0]["path"], "comparison": comparison["mount"]}


def canonical_grasp_transform(gripper, config):
    roots = [j for j in gripper.findall("joint") if j.find("parent").get("link") == "world"]
    if len(roots) != 1 or roots[0].get("type") != "fixed":
        raise ValueError("Expected a fixed canonical world-to-gripper-base joint")
    joint = roots[0]
    if joint.find("child").get("link") != "robotiq_arg2f_base_link":
        raise ValueError("Unexpected canonical gripper base")
    origin = joint.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    if (xyz.shape != (3,) or rpy.shape != (3,) or not np.allclose(xyz, 0, atol=1e-9)
            or not np.allclose(rpy, [0, 0, np.pi / 2], atol=1e-5)):
        raise ValueError("Canonical root differs from the supplied official 2F-85; review, do not guess")
    rotation = np.asarray(config.get("base_rotation", np.eye(3)), dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation, np.eye(3)):
        raise ValueError("Additional canonical base_rotation requires separate geometry review")
    if config.get("open") != {"finger_joint": 0.0} or config.get("close") != {"finger_joint": 0.8}:
        raise ValueError("Unexpected 2F-85 master joint configuration")
    transform = np.eye(4)
    c, s = np.cos(rpy[2]), np.sin(rpy[2])
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    # The 136 mm fingertip field is not an additional tool-origin translation.
    return transform, joint


def compose_fk_urdf(arm_path, gripper_path, config):
    """Keep the official arm, replace its hand geometry with source gripper links.

    Mesh filenames become absolute but mesh bytes, scale, origins, joint axes,
    limits and mimic relationships are unchanged. No collision spheres are fitted.
    """
    arm_path, gripper_path = Path(arm_path).resolve(), Path(gripper_path).resolve()
    arm, gripper = ET.parse(arm_path).getroot(), ET.parse(gripper_path).getroot()
    transform, canonical_joint = canonical_grasp_transform(gripper, config)
    if not set(ARM_JOINTS).issubset({j.get("name") for j in arm.findall("joint")}):
        raise ValueError("Source is not the expected seven-joint Panda arm")
    hand = arm.find("link[@name='panda_hand']")
    if hand is None:
        raise ValueError("Source Panda hand frame is missing")
    for child in list(hand):
        hand.remove(child)  # Native Robotiq base has its own source geometry.
    removed = {"panda_leftfinger", "panda_rightfinger", "ee_link", "right_gripper"}
    for element in list(arm):
        if (element.tag == "link" and element.get("name") in removed
                or element.tag == "joint" and element.find("child").get("link") in removed):
            arm.remove(element)
    inputs = []
    for root, directory in ((arm, arm_path.parent), (gripper, gripper_path.parent)):
        for mesh in root.findall(".//mesh"):
            value = mesh.get("filename")
            if "://" in value:
                raise ValueError(f"Unresolved mesh URI: {value}")
            path = (directory / value).resolve()
            inputs.append(file_identity(path))
            mesh.set("filename", str(path))
    names = {e.get("name") for e in arm if e.tag in ("link", "joint")}
    for element in gripper:
        if element is canonical_joint or element.tag == "link" and element.get("name") == "world":
            continue
        if element.tag not in ("link", "joint", "material"):
            raise ValueError(f"Unsupported source URDF element: {element.tag}")
        if element.tag in ("link", "joint") and element.get("name") in names:
            raise ValueError(f"URDF name collision: {element.get('name')}")
        if element.tag in ("link", "joint"):
            names.add(element.get("name"))
        arm.append(deepcopy(element))
    joint = ET.SubElement(arm, "joint", name="jikkenn1_robotiq_mount", type="fixed")
    ET.SubElement(joint, "parent", link="panda_hand")
    ET.SubElement(joint, "child", link="robotiq_arg2f_base_link")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    arm.set("name", "panda_robotiq_2f_85_fk_only")
    ET.indent(arm)
    return ET.tostring(arm, encoding="unicode"), transform, inputs
