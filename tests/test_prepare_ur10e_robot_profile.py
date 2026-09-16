from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_ur10e_robot_profile.py"
SPEC = importlib.util.spec_from_file_location("prepare_ur10e_robot_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareUr10eRobotProfileTests(unittest.TestCase):
    def test_source_hash_is_independent_of_checkout_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"first\nsecond\n")
            crlf.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(MODULE._sha256(lf), MODULE._sha256(crlf))

    def test_pinned_official_sources_match_recorded_hashes_and_contract(self) -> None:
        root = Path(__file__).parents[1] / MODULE.SOURCE_DIRECTORY
        urdf = root / f"{MODULE.SOURCE_STEM}.urdf"
        xrdf = root / f"{MODULE.SOURCE_STEM}.xrdf"
        self.assertEqual(MODULE._sha256(urdf), MODULE.OFFICIAL_URDF_SHA256)
        self.assertEqual(MODULE._sha256(xrdf), MODULE.OFFICIAL_XRDF_SHA256)
        self.assertTrue(MODULE.inspect_official_urdf(urdf)["passed"])
        # The digest above pins the full vendored XRDF. Keep this test runnable
        # in the minimal repository test environment, which need not install
        # PyYAML; the GraspGenX runtime performs the actual YAML parse.
        report = MODULE.inspect_official_xrdf(
            {
                "format": "xrdf",
                "format_version": 1.0,
                "modifiers": [
                    {
                        "add_frame": {
                            "frame_name": "attached_object",
                            "parent_frame_name": "grasp_frame",
                        }
                    }
                ],
                "cspace": {"joint_names": list(MODULE.EXPECTED_ARM_JOINTS)},
                "tool_frames": ["grasp_frame"],
                "collision": {"geometry": "official"},
                "geometry": {
                    "official": {
                        "spheres": {
                            "robotiq_base_link": [],
                            "left_inner_finger_pad": [],
                            "right_inner_finger_pad": [],
                            "attached_object": [],
                        }
                    }
                },
            }
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["tool_frame"], "grasp_frame")
        self.assertEqual(report["attached_object_parent"], "grasp_frame")

    def test_changed_official_mount_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "robot.urdf"
            path.write_text(
                """<robot name="test">
<link name="tool0"/><link name="robotiq_base_link"/>
<link name="gripper_frame"/><link name="grasp_frame"/>
<joint name="mount" type="fixed">
  <parent link="tool0"/><child link="robotiq_base_link"/>
  <origin rpy="0 0 0" xyz="0 0 0"/>
</joint>
<joint name="grasp" type="fixed">
  <parent link="gripper_frame"/><child link="grasp_frame"/>
  <origin rpy="0 0 0" xyz="0 0 0.2"/>
</joint></robot>""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tool transforms changed"):
                MODULE.inspect_official_urdf(path)

    def test_adapter_preserves_official_geometry_and_adds_runtime_contract(self) -> None:
        collision_spheres = {
            "robotiq_base_link": [{"center": [0, 0, 0], "radius": 0.03}],
            "left_inner_finger_pad": [{"center": [0, 0, 0], "radius": 0.01}],
            "right_inner_finger_pad": [{"center": [0, 0, 0], "radius": 0.01}],
            "attached_object": [{"center": [0, 0, 0], "radius": -100.0}],
        }
        official_ignore = {"attached_object": ["robotiq_base_link"]}
        source = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": "/tmp/official.urdf",
                    "tool_frames": ["grasp_frame"],
                    "collision_spheres": collision_spheres,
                    "collision_link_names": list(collision_spheres),
                    "cspace": {
                        "joint_names": [
                            *MODULE.EXPECTED_ARM_JOINTS,
                            "finger_joint",
                        ]
                    },
                    "extra_links": {
                        "attached_object": {
                            "parent_link_name": "grasp_frame",
                            "link_name": "attached_object",
                        }
                    },
                    "self_collision_ignore": official_ignore,
                }
            }
        }
        result = MODULE.adapt_config(source)
        kin = result["robot_cfg"]["kinematics"]
        self.assertEqual(kin["urdf_path"], "/tmp/official.urdf")
        self.assertEqual(kin["tool_frames"], ["grasp_frame"])
        self.assertEqual(kin["collision_spheres"], collision_spheres)
        self.assertEqual(kin["self_collision_ignore"], official_ignore)
        self.assertEqual(kin["extra_collision_spheres"], {"attached_object": 4})
        self.assertEqual(
            kin["grasp_contact_link_names"],
            [
                "robotiq_base_link",
                "left_inner_finger_pad",
                "right_inner_finger_pad",
                "attached_object",
            ],
        )


if __name__ == "__main__":
    unittest.main()
