import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PREGRASP_SCRIPT = PROJECT / "scripts" / "curobo_plan_pregrasp_a.py"
SPEC = importlib.util.spec_from_file_location("curobo_plan_pregrasp_test", PREGRASP_SCRIPT)
PREGRASP_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREGRASP_MODULE)


class CuroboVoxelFixScriptTests(unittest.TestCase):
    def test_trajectory_field_selects_only_single_official_batch_and_seed(self):
        trajectory = SimpleNamespace(
            position=np.zeros((1, 1, 41, 9), dtype=np.float32)
        )
        result = PREGRASP_MODULE._trajectory_field(
            trajectory, "position", required=True
        )
        self.assertEqual(result.shape, (41, 9))

    def test_trajectory_field_rejects_multiple_results(self):
        trajectory = SimpleNamespace(
            position=np.zeros((1, 2, 41, 9), dtype=np.float32)
        )
        with self.assertRaisesRegex(RuntimeError, "multiple batch/seed"):
            PREGRASP_MODULE._trajectory_field(
                trajectory, "position", required=True
            )

    def test_full_trajectory_uses_official_name_aware_active_conversion(self):
        full_names = [f"joint_{index}" for index in range(9)]
        active_names = full_names[:7]
        full = SimpleNamespace(
            position=np.zeros((1, 1, 41, 9), dtype=np.float32),
            joint_names=full_names,
        )

        class FakeTrajoptSolver:
            def __init__(self):
                self.called = False

            def get_active_js(self, trajectory):
                self.called = True
                return SimpleNamespace(
                    position=trajectory.position[..., :7],
                    joint_names=active_names,
                )

        solver = FakeTrajoptSolver()
        planner = SimpleNamespace(
            trajopt_solver=solver,
            joint_names=active_names,
        )
        active, recorded_full_names = PREGRASP_MODULE._get_active_trajectory(
            planner, full
        )
        self.assertTrue(solver.called)
        self.assertEqual(active.position.shape, (1, 1, 41, 7))
        self.assertEqual(recorded_full_names, full_names)

    def test_full_trajectory_rejects_missing_name_column(self):
        trajectory = SimpleNamespace(
            position=np.zeros((1, 1, 41, 9), dtype=np.float32),
            joint_names=[f"joint_{index}" for index in range(8)],
        )
        planner = SimpleNamespace(
            trajopt_solver=SimpleNamespace(),
            joint_names=[f"joint_{index}" for index in range(7)],
        )
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            PREGRASP_MODULE._get_active_trajectory(planner, trajectory)

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
