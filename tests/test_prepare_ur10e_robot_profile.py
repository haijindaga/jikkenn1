from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_ur10e_robot_profile.py"
SPEC = importlib.util.spec_from_file_location("prepare_ur10e_robot_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareUr10eRobotProfileTests(unittest.TestCase):
    def test_adapter_preserves_official_fields_and_adds_attached_object_contract(self) -> None:
        links = [
            "shoulder_link",
            "tool0",
            "robotiq_arg2f_base_link",
            "left_inner_finger_pad",
            "right_inner_finger_pad",
        ]
        source = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": "/tmp/official.urdf",
                    "collision_link_names": links.copy(),
                    "cspace": {
                        "joint_names": [
                            "shoulder_pan_joint",
                            "shoulder_lift_joint",
                            "elbow_joint",
                            "wrist_1_joint",
                            "wrist_2_joint",
                            "wrist_3_joint",
                            "finger_joint",
                        ]
                    },
                    "self_collision_ignore": {},
                }
            }
        }
        result = MODULE.adapt_config(source)
        kin = result["robot_cfg"]["kinematics"]
        self.assertEqual(kin["urdf_path"], "/tmp/official.urdf")
        self.assertEqual(kin["tool_frames"], ["robotiq_arg2f_base_link"])
        self.assertEqual(kin["extra_collision_spheres"], {"attached_object": 4})
        self.assertEqual(
            kin["extra_links"]["attached_object"]["parent_link_name"],
            "robotiq_arg2f_base_link",
        )
        self.assertIn("attached_object", kin["collision_link_names"])
        self.assertNotIn(
            "attached_object", kin["self_collision_ignore"]["attached_object"]
        )
        self.assertNotIn(
            "shoulder_link", kin["self_collision_ignore"]["attached_object"]
        )


if __name__ == "__main__":
    unittest.main()
