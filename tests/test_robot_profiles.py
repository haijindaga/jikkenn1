from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from panda_handover.robot_profiles import get_robot_profile, robot_profile_names


class RobotProfileTests(unittest.TestCase):
    def test_reviewed_profiles_are_valid_and_frame_transforms_are_rigid(self) -> None:
        self.assertEqual(
            robot_profile_names(), ("franka_panda", "ur10e_robotiq_2f_140")
        )
        for name in robot_profile_names():
            profile = get_robot_profile(name)
            transform = np.asarray(profile.grasp_to_tool_transform)
            self.assertTrue(np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]))
            self.assertTrue(
                np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
            )
            self.assertTrue(np.isclose(np.linalg.det(transform[:3, :3]), 1.0))


    def test_external_ur10e_config_resolves_below_graspgenx(self) -> None:
        profile = get_robot_profile("ur10e_robotiq_2f_140")
        resolved = profile.resolve_curobo_config(Path("/opt/GraspGenX"))
        resolved_path = Path(resolved)
        self.assertEqual(resolved_path.name, "ur10e_robotiq_2f_140.jikkenn1.yml")
        self.assertEqual(resolved_path.parent.name, "curobo_assets")
        self.assertEqual(resolved_path.parent.parent.name, "end2end")
        self.assertEqual(profile.isaac_gripper_variant, "robotiq_2f_140")
        self.assertEqual(profile.gripper_joint_position_unit, "radian")
        self.assertEqual(
            profile.observation_arm_joint_positions,
            (
                3.141592653589793,
                -1.5707963267948966,
                1.5707963267948966,
                -1.5707963267948966,
                -1.5707963267948966,
                0.0,
            ),
        )


    def test_franka_keeps_bundled_curobo_config_name(self) -> None:
        profile = get_robot_profile("franka_panda")
        self.assertEqual(profile.resolve_curobo_config(Path("/unused")), "franka.yml")
        self.assertEqual(profile.gripper_joint_position_unit, "metre")
        self.assertIsNone(profile.observation_arm_joint_positions)

    def test_pregrasp_checks_isaac_and_curobo_tool_frame_alignment(self) -> None:
        script = (
            Path(__file__).parents[1] / "scripts" / "curobo_plan_pregrasp_a.py"
        ).read_text(encoding="utf-8")
        self.assertIn('args.capture / "T_robot_base_tool.npy"', script)
        self.assertIn(
            "start_kinematics.tool_poses.get_link_pose(profile.tool_frame)", script
        )
        self.assertIn('"robot_model_alignment": model_alignment', script)
        self.assertIn("Isaac and cuRobo tool frames disagree", script)

    def test_non_franka_capture_requires_matching_scene_profile(self) -> None:
        script = (
            Path(__file__).parents[1] / "scripts" / "isaac_capture_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn('GetCustomDataByKey("panda_handover:robot_profile")', script)
        self.assertIn("non-Franka authored scenes must record", script)


if __name__ == "__main__":
    unittest.main()
