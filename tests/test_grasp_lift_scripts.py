from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class GraspLiftScriptTests(unittest.TestCase):
    def test_planner_uses_reviewed_official_grasp_and_attachment_apis(self):
        source = (PROJECT / "scripts" / "curobo_plan_grasp_lift.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("planner.plan_grasp(", source)
        self.assertIn("planner.attachment_manager.attach(", source)
        self.assertIn("Mesh.from_pointcloud(", source)
        self.assertIn("load_singleview_observed_pointcloud(", source)
        self.assertNotIn("file_path=str(mesh_path", source)
        self.assertNotIn("observed_scene_mesh.obj", source)
        self.assertIn('disable_collision_links=[]', source)
        self.assertIn("https://github.com/NVlabs/curobo/issues/663", source)
        self.assertIn("https://github.com/NVlabs/curobo/issues/692", source)
        self.assertIn("trim_joint_state_trajectory(", source)

    def test_first_lift_limitation_is_explicit_and_fail_visible(self):
        source = (PROJECT / "scripts" / "curobo_plan_grasp_lift.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"held_object_collision_checked_during_first_lift": False', source)
        self.assertIn('"safe_for_real_robot_execution": False', source)
        self.assertIn('"manual_review_required": True', source)


if __name__ == "__main__":
    unittest.main()
