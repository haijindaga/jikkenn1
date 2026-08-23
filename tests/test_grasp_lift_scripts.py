from pathlib import Path
import runpy
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


class GraspLiftScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = runpy.run_path(
            str(PROJECT / "scripts" / "curobo_plan_grasp_lift.py"),
            run_name="curobo_plan_grasp_lift_test",
        )
        cls.restore_triangle_faces = staticmethod(namespace["_restore_triangle_faces"])

    def test_flat_curobo_faces_are_restored_without_geometry_changes(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        restored = self.restore_triangle_faces(vertices, [0, 1, 2, 0, 2, 3])
        np.testing.assert_array_equal(restored, [[0, 1, 2], [0, 2, 3]])

    def test_malformed_curobo_faces_fail_instead_of_being_repaired(self):
        vertices = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            self.restore_triangle_faces(vertices, [0, 1])
        with self.assertRaises(RuntimeError):
            self.restore_triangle_faces(vertices, [0, 1, 3])

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
        self.assertIn("joint_states=grasp_end", source)
        self.assertNotIn("joint_states=lift_end,\n        obstacles=[target_mesh]", source)
        self.assertIn("Mesh.from_pointcloud(", source)
        self.assertIn("_restore_triangle_faces(", source)
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

    def test_optional_handover_transport_uses_attached_official_pose_planner(self):
        source = (PROJECT / "scripts" / "curobo_plan_grasp_lift.py").read_text(
            encoding="utf-8"
        )
        attach_at = source.index("attachment_manager.attach(")
        transport_at = source.index("transport_result = planner.plan_pose(")
        self.assertLess(attach_at, transport_at)
        self.assertIn('"--handover-goal-position-robot-base-m"', source)
        self.assertIn("current_state=lift_end", source)
        self.assertIn("preserve_selected_grasp_orientation", source)
        self.assertIn("transport_all_waypoints_clear_of_observed_scene", source)
        self.assertIn('"human_or_receiver_collision_model_present": False', source)
        self.assertIn('"handover_release_planned": False', source)
        self.assertNotIn("transport_result.status", source)
        self.assertIn('"result_type": type(transport_result).__name__', source)

    def test_first_lift_limitation_is_explicit_and_fail_visible(self):
        source = (PROJECT / "scripts" / "curobo_plan_grasp_lift.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"held_object_collision_checked_during_first_lift": False', source)
        self.assertIn('"attachment_transform_defined_at_grasp_end": True', source)
        self.assertIn('"safe_for_real_robot_execution": False', source)
        self.assertIn('"manual_review_required": True', source)


if __name__ == "__main__":
    unittest.main()
