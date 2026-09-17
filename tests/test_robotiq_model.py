from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np

from panda_handover.robotiq_model import (
    body_transform, canonical_grasp_transform, compose_fk_urdf, file_identity, installed_mount,
)


CONFIG = {"open": {"finger_joint": 0.0}, "close": {"finger_joint": 0.8}, "fingertip": [0, 0, .136]}
GRIPPER = '''<robot name="robotiq_arg2f_85_model">
<link name="world"/><link name="robotiq_arg2f_base_link"><collision><geometry>
<mesh filename="base.stl" scale="0.001 0.001 0.001"/></geometry></collision></link>
<joint name="world_joint" type="fixed"><parent link="world"/><child link="robotiq_arg2f_base_link"/>
<origin xyz="0 0 0" rpy="0 0 1.5708"/></joint>
<link name="left_outer_knuckle"/>
<joint name="finger_joint" type="revolute"><parent link="robotiq_arg2f_base_link"/>
<child link="left_outer_knuckle"/><origin xyz="0 -0.0306011 0.054904" rpy="0 0 3.14159265359"/>
<axis xyz="1 0 0"/><limit lower="0" upper="0.8" effort="1000" velocity="2"/></joint>
<link name="right_outer_knuckle"/>
<joint name="right_outer_knuckle_joint" type="revolute"><parent link="robotiq_arg2f_base_link"/>
<child link="right_outer_knuckle"/><axis xyz="1 0 0"/>
<mimic joint="finger_joint" multiplier="1" offset="0"/></joint></robot>'''


def evidence_fixture():
    value = np.eye(4)
    value[:3, 3] = [3, 2, 1]
    bodies = [{"path": f"/World/Panda/{name}", "name": name, "T_world_body": value.tolist()}
              for name in ("panda_hand", "base_link")]
    return {"open_close_checks_passed": True, "rigid_bodies": bodies,
            "joints": [{"path": "/World/Panda/AssemblerFixedJoint", "type": "PhysicsFixedJoint",
                        "body0": [bodies[0]["path"]], "body1": [bodies[1]["path"]]}]}


class RobotiqModelTests(unittest.TestCase):
    def test_mount_requires_body_poses_and_fixed_joint(self):
        result = installed_mount(evidence_fixture())
        np.testing.assert_allclose(result["T_hand_gripper_base"], np.eye(4))
        self.assertTrue(result["comparison"]["passed"])
        for mutate in ("joint", "translation", "rotation", "openclose"):
            evidence = evidence_fixture()
            if mutate == "joint":
                evidence["joints"] = []
            elif mutate == "translation":
                evidence["rigid_bodies"][1]["T_world_body"][0][3] += .01
            elif mutate == "rotation":
                evidence["rigid_bodies"][1]["T_world_body"][0][0] = -1
                evidence["rigid_bodies"][1]["T_world_body"][1][1] = -1
            else:
                evidence["open_close_checks_passed"] = False
            with self.assertRaises(ValueError):
                installed_mount(evidence)

    def test_invalid_affine_pose_and_duplicate_body_rejected(self):
        evidence = evidence_fixture()
        evidence["rigid_bodies"][0]["T_world_body"][0][0] = 2
        with self.assertRaises(ValueError):
            body_transform(evidence, "panda_hand")
        evidence = evidence_fixture()
        evidence["rigid_bodies"].append(deepcopy(evidence["rigid_bodies"][0]))
        with self.assertRaises(ValueError):
            body_transform(evidence, "panda_hand")

    def test_canonical_rotation_is_positive_root_rotation_not_fingertip_offset(self):
        value, _ = canonical_grasp_transform(ET.fromstring(GRIPPER), CONFIG)
        np.testing.assert_allclose(value[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=4e-6)
        np.testing.assert_allclose(value[:3, 3], 0)
        for xml, config in ((GRIPPER.replace("1.5708", "-1.5708"), CONFIG),
                            (GRIPPER, {**CONFIG, "base_rotation": np.diag([-1, -1, 1]).tolist()}),
                            (GRIPPER, {**CONFIG, "close": {"finger_joint": 1.0}})):
            with self.assertRaises(ValueError):
                canonical_grasp_transform(ET.fromstring(xml), config)

    def test_composition_preserves_arm_joints_scales_mimics_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = ET.Element("robot", name="panda")
            for i in range(8):
                ET.SubElement(arm, "link", name=f"panda_link{i}")
                if i:
                    joint = ET.SubElement(arm, "joint", name=f"panda_joint{i}", type="revolute")
                    ET.SubElement(joint, "parent", link=f"panda_link{i-1}")
                    ET.SubElement(joint, "child", link=f"panda_link{i}")
                    ET.SubElement(joint, "origin", xyz=f"0 0 {i/10}", rpy="0 0 0")
            hand = ET.SubElement(arm, "link", name="panda_hand")
            ET.SubElement(hand, "visual")
            ET.SubElement(hand, "inertial")
            joint = ET.SubElement(arm, "joint", name="panda_hand_joint", type="fixed")
            ET.SubElement(joint, "parent", link="panda_link7")
            ET.SubElement(joint, "child", link="panda_hand")
            ET.SubElement(joint, "origin", xyz="0 0 .107", rpy="0 0 -.785398")
            for side in ("left", "right"):
                ET.SubElement(arm, "link", name=f"panda_{side}finger")
                joint = ET.SubElement(arm, "joint", name=f"panda_finger_joint{1 if side == 'left' else 2}", type="prismatic")
                ET.SubElement(joint, "parent", link="panda_hand")
                ET.SubElement(joint, "child", link=f"panda_{side}finger")
            arm_path, gripper_path = root / "arm.urdf", root / "gripper.urdf"
            arm_path.write_text(ET.tostring(arm, encoding="unicode"))
            gripper_path.write_text(GRIPPER)
            mesh = root / "base.stl"
            mesh.write_bytes(b"test mesh content; unit fixture only")
            before = [file_identity(p) for p in (arm_path, gripper_path, mesh)]
            xml, transform, inputs = compose_fk_urdf(arm_path, gripper_path, CONFIG)
            result = ET.fromstring(xml)
            for name in [f"panda_joint{i}" for i in range(1, 8)] + ["panda_hand_joint"]:
                actual = result.find(f"joint[@name='{name}']")
                original = ET.fromstring(arm_path.read_text()).find(f"joint[@name='{name}']")
                self.assertEqual([(e.tag, e.attrib) for e in actual.iter()],
                                 [(e.tag, e.attrib) for e in original.iter()])
            self.assertEqual(len(result.find("link[@name='panda_hand']")), 0)
            self.assertIsNone(result.find("link[@name='world']"))
            self.assertIsNone(result.find("joint[@name='panda_finger_joint1']"))
            collision_mesh = result.find(".//mesh")
            self.assertEqual(collision_mesh.get("filename"), str(mesh.resolve()))
            self.assertEqual(collision_mesh.get("scale"), "0.001 0.001 0.001")
            self.assertEqual(result.find("joint[@name='right_outer_knuckle_joint']/mimic").get("multiplier"), "1")
            self.assertEqual(before, [file_identity(p) for p in (arm_path, gripper_path, mesh)])
            np.testing.assert_allclose(transform[:3, 3], 0)
            self.assertEqual(inputs[0], file_identity(mesh))
            # Exercise the actual preparation CLI, without pretending the tiny
            # fixture is a real/verified collision mesh or a robot runtime.
            official_arm = root / "ext/curobo/curobo/content/assets/robot/franka_description/franka_panda.urdf"
            official_arm.parent.mkdir(parents=True)
            shutil.copyfile(arm_path, official_arm)
            (root / "config.json").write_text(json.dumps(CONFIG))
            (root / "coll_mesh.obj").write_text("# canonical mesh fixture, not runtime geometry\n")
            evidence_path, check_path = root / "evidence.json", root / "arm_fk.json"
            evidence_path.write_text(json.dumps(evidence_fixture()))
            check_path.write_text(json.dumps({"status": "arm_alignment_passed",
                "inputs": {"evidence": str(evidence_path)},
                "frames": {f"panda_link{i}": {"passed": True} for i in range(8)}}))
            fake_x = types.ModuleType("graspgenx.x_grippers")
            fake_x.resolve_gripper_asset_dir = lambda name, **kwargs: str(root)
            script = Path(__file__).resolve().parents[1] / "scripts/prepare_panda_robotiq_model.py"
            spec = importlib.util.spec_from_file_location("test_prepare_robotiq_cli", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output = root / "prepared_model"
            arguments = [str(script), "--graspgenx-root", str(root), "--evidence", str(evidence_path),
                         "--arm-fk-check", str(check_path), "--output", str(output)]
            modules = {"graspgenx": types.ModuleType("graspgenx"), "graspgenx.x_grippers": fake_x}
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments):
                self.assertEqual(module.main(), 0)
                with self.assertRaises(SystemExit):
                    module.main()  # Refuse overwriting existing experiment files.
            report = json.loads((output / "model_preparation_check.json").read_text())
            self.assertEqual(report["status"], "fk_model_prepared")
            self.assertFalse(report["safety"]["profile_ready"])
            self.assertFalse(report["safety"]["collision_geometry_equivalence_verified"])
            self.assertFalse(report["safety"]["source_assets_modified"])
            self.assertFalse(list(output.glob("*.yml")))
            self.assertEqual(report["sources"]["gripper_urdf"], before[1])
            check_path.write_text(json.dumps({"status": "arm_alignment_failed"}))
            arguments[-1] = str(root / "rejected")
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments):
                self.assertEqual(module.main(), 2)
            rejected = json.loads((root / "rejected/model_preparation_check.json").read_text())
            self.assertEqual(rejected["status"], "failure")
            self.assertFalse((root / "rejected/panda_robotiq85_fk_only.urdf").exists())
            mesh.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 5000\n")
            with self.assertRaises(ValueError):
                compose_fk_urdf(arm_path, gripper_path, CONFIG)
            mesh.unlink()
            with self.assertRaises(FileNotFoundError):
                compose_fk_urdf(arm_path, gripper_path, CONFIG)
