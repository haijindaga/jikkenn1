from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class CuroboVoxelFixScriptTests(unittest.TestCase):
    def test_patch_is_exactly_the_six_issue_699_rounding_changes(self):
        patch = (
            PROJECT / "patches" / "curobo" / "057a96f-voxel-grid-round.patch"
        ).read_text(encoding="utf-8")
        self.assertEqual(patch.count("+    dims_x = wp.int32(wp.round("), 2)
        self.assertEqual(patch.count("+    dims_y = wp.int32(wp.round("), 2)
        self.assertEqual(patch.count("+    dims_z = wp.int32(wp.round("), 2)
        self.assertEqual(patch.count("-    dims_x = wp.int32("), 2)
        self.assertEqual(patch.count("-    dims_y = wp.int32("), 2)
        self.assertEqual(patch.count("-    dims_z = wp.int32("), 2)
        self.assertIn("curobo/_src/geom/data/data_voxel.py", patch)

    def test_gpu_regression_is_fail_closed_before_real_esdf(self):
        source = (PROJECT / "scripts" / "check_curobo_voxel_round_fix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("shape = (120, 120, 100)", source)
        self.assertIn("sphere_obstacle_collision_kernel", source)
        self.assertIn("truncated_shape !=", source)
        self.assertIn('"safe_to_load_real_esdf": bool(all(checks.values()))', source)
        self.assertIn("patched_source_sha256", source)

    def test_pregrasp_script_uses_official_pose_planner_not_grasp_contact_mode(self):
        source = (PROJECT / "scripts" / "curobo_plan_pregrasp_a.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MotionPlannerCfg.create(", source)
        self.assertIn("planner.plan_pose(", source)
        self.assertNotIn("planner.plan_grasp(", source)
        self.assertIn('choices=("blocked", "free")', source)
        self.assertIn('"simulation_only": unknown_policy != "blocked"', source)
        self.assertIn('"final_approach_planned": False', source)
        self.assertIn('"safe_to_execute": False', source)

    def test_observed_mesh_backend_is_official_and_fail_closed(self):
        source = (PROJECT / "scripts" / "curobo_plan_pregrasp_a.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"observed_pointcloud_mesh"', source)
        self.assertIn("Mesh.from_pointcloud(", source)
        self.assertIn("SceneCfg(mesh=[scene_mesh])", source)
        self.assertIn("load_singleview_observed_pointcloud(", source)
        self.assertIn('"unknown_space_assumed_free": unknown_policy != "blocked"', source)
        self.assertIn('"simulation_only": unknown_policy != "blocked"', source)
        self.assertNotIn("all grasps collide", source.lower())


if __name__ == "__main__":
    unittest.main()
