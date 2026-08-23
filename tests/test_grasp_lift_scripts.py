from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class GraspLiftScriptTests(unittest.TestCase):
    def test_planner_uses_reviewed_official_grasp_and_attachment_apis(self):
        source = (PROJECT / "scripts" / "curobo_plan_grasp_lift.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("planner.plan_grasp(", source)
        self.assertIn(
            "attachment_manager = planner.trajopt_solver.core.attachment_manager",
            source,
        )
        self.assertIn("attachment_manager.attach(", source)
        self.assertIn("Mesh.from_pointcloud(", source)
        self.assertIn("load_singleview_observed_pointcloud(", source)
        self.assertNotIn("file_path=str(mesh_path", source)
        self.assertNotIn("observed_scene_mesh.obj", source)
        self.assertNotIn('disable_collision_links=[]', source)
        self.assertIn("grasp_contact_link_names", source)
        self.assertIn("planner.enable_link_collision(contact_collision_links)", source)
        self.assertIn("all_returned_waypoints_clear_with_all_robot_links_enabled", source)
        self.assertIn('"status": "planning_failed"', source)
        self.assertIn("GraspGenX/end2end/e2e_grasp_demo.py::init_planner", source)
        self.assertNotIn("num_ik_seeds=16", source)
        self.assertNotIn("num_trajopt_seeds=2", source)
        self.assertNotIn("        use_cuda_graph=False,", source)
        self.assertIn("planner.warmup(enable_graph=False, num_warmup_iterations=1)", source)
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
