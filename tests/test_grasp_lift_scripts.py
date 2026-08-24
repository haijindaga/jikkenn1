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
        cls.collision_sphere_link_names = staticmethod(
            namespace["_collision_sphere_link_names"]
        )
        cls.phase_contact_diagnostics = staticmethod(
            namespace["_phase_contact_diagnostics"]
        )
        cls.review_transient_finger_support_contact = staticmethod(
            namespace["_review_transient_finger_support_contact"]
        )

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

    def test_collision_sphere_links_use_curobo_ownership_api(self):
        class FakeKinematicsParams:
            link_name_to_idx_map = {"arm": 0, "finger": 1}

            @staticmethod
            def get_sphere_index_from_link_name(link_name):
                return {
                    "arm": np.array([0, 1]),
                    "finger": np.array([2]),
                }[link_name]

        names, indices = self.collision_sphere_link_names(FakeKinematicsParams(), 3)
        self.assertEqual(names, ["arm", "arm", "finger"])
        self.assertEqual(indices, {"arm": [0, 1], "finger": [2]})

    def test_contact_diagnostics_preserve_link_and_nearest_surfaces(self):
        costs = np.zeros((2, 1, 2), dtype=np.float32)
        costs[1, 0, 1] = 0.002
        spheres = np.zeros((2, 1, 2, 4), dtype=np.float32)
        spheres[1, 0, 1] = [1.0, 0.0, 0.0, 0.1]
        report = self.phase_contact_diagnostics(
            "grasp",
            costs,
            spheres,
            ["arm", "finger"],
            np.array([[1.05, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.2, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(report["positive_count"], 1)
        contact = report["contacts"][0]
        self.assertEqual(contact["waypoint_index"], 1)
        self.assertEqual(contact["sphere_index"], 1)
        self.assertEqual(contact["link_name"], "finger")
        self.assertAlmostEqual(
            contact["nearest_observed_source_point"]["sphere_surface_clearance_m"],
            -0.05,
            places=5,
        )

    def test_reviewed_transient_finger_support_contact_is_narrowly_accepted(self):
        def phase(name, waypoint_count, contacts):
            return {
                "phase": name,
                "cost_shape": [waypoint_count, 1, 65],
                "contacts": contacts,
            }

        def contact(waypoint, cost=0.00049, link="panda_rightfinger"):
            return {
                "waypoint_index": waypoint,
                "collision_cost_m": cost,
                "link_name": link,
                "nearest_observed_source_point": {
                    "point_robot_base_m": [0.45, -0.02, 0.005]
                },
            }

        report = self.review_transient_finger_support_contact(
            {
                "approach": phase("approach", 41, []),
                "grasp": phase("grasp", 41, [contact(i) for i in range(37, 41)]),
                "lift": phase("lift", 41, [contact(i) for i in range(4)]),
            },
            support_surface_z_m=0.0,
            support_height_tolerance_m=0.0051,
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(all(report["checks"].values()))

    def test_support_contact_policy_rejects_arm_deep_or_persistent_contact(self):
        def contact(waypoint, cost=0.0005, link="panda_rightfinger"):
            return {
                "waypoint_index": waypoint,
                "collision_cost_m": cost,
                "link_name": link,
                "nearest_observed_source_point": {
                    "point_robot_base_m": [0.45, -0.02, 0.005]
                },
            }

        def report_for(contact_factory):
            return self.review_transient_finger_support_contact(
                {
                    "approach": {"cost_shape": [2, 1, 65], "contacts": []},
                    "grasp": {
                        "cost_shape": [2, 1, 65],
                        "contacts": [contact_factory(1)],
                    },
                    "lift": {
                        "cost_shape": [2, 1, 65],
                        "contacts": [contact_factory(0)],
                    },
                },
                support_surface_z_m=0.0,
                support_height_tolerance_m=0.0051,
            )

        self.assertTrue(report_for(contact)["accepted"])
        self.assertFalse(
            report_for(lambda waypoint: contact(waypoint, link="panda_hand"))[
                "accepted"
            ]
        )
        self.assertFalse(
            report_for(lambda waypoint: contact(waypoint, cost=0.0011))["accepted"]
        )

        persistent = self.review_transient_finger_support_contact(
            {
                "approach": {"cost_shape": [2, 1, 65], "contacts": []},
                "grasp": {"cost_shape": [2, 1, 65], "contacts": [contact(1)]},
                "lift": {
                    "cost_shape": [2, 1, 65],
                    "contacts": [contact(0), contact(1)],
                },
            },
            support_surface_z_m=0.0,
            support_height_tolerance_m=0.0051,
        )
        self.assertFalse(persistent["accepted"])

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
        self.assertIn("returned_waypoints_pass_simulation_contact_policy", source)
        self.assertIn(
            "strict_all_returned_waypoints_clear_with_all_robot_links_enabled",
            source,
        )
        self.assertIn("get_sphere_index_from_link_name", source)
        self.assertIn("grasp_contact_diagnostics.json", source)
        self.assertIn("_full_robot_spheres_world.npy", source)
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
