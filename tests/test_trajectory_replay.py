import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from panda_handover.trajectory_replay import (
    load_grasp_lift_replay,
    load_pregrasp_replay,
    normalize_segment_dt,
    sample_positions_at_physics_rate,
)


class TrajectoryReplayTests(unittest.TestCase):
    def _write_valid_inputs(self, root: Path) -> tuple[Path, Path]:
        capture = root / "capture"
        plan = root / "plan"
        capture.mkdir()
        plan.mkdir()
        names = ["panda_joint1", "panda_joint2", "panda_finger_joint1"]
        (capture / "robot_state.json").write_text(
            json.dumps({"joint_names": names}), encoding="utf-8"
        )
        np.save(capture / "panda_joint_positions.npy", np.array([0.1, -0.2, 0.04]))
        report = {
            "status": "success",
            "result": {
                "planner_reported_success": True,
                "trajectory_active_joint_names": names[:2],
            },
            "automatic_checks": {"finite": True, "starts_at_capture": True},
            "safety": {
                "simulation_only": True,
                "pregrasp_only_scope_gate_passed": True,
                "final_approach_planned": False,
                "gripper_close_planned": False,
            },
        }
        (plan / "pregrasp_plan_check.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        np.save(
            plan / "trajectory_position.npy",
            np.array([[0.1, -0.2], [0.2, -0.1], [0.3, 0.0]]),
        )
        np.save(plan / "trajectory_dt_s.npy", np.array(0.02))
        return capture, plan

    def test_loads_name_mapped_simulation_only_pregrasp(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture, plan = self._write_valid_inputs(Path(temporary_directory))
            replay = load_pregrasp_replay(capture, plan)
            self.assertEqual(replay.joint_names, ("panda_joint1", "panda_joint2"))
            np.testing.assert_array_equal(replay.capture_indices, [0, 1])
            np.testing.assert_allclose(replay.segment_dt_s, [0.02, 0.02])

    def test_rejects_non_simulation_plan(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture, plan = self._write_valid_inputs(Path(temporary_directory))
            report_path = plan / "pregrasp_plan_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["safety"]["simulation_only"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "simulation-only"):
                load_pregrasp_replay(capture, plan)

    def test_rejects_start_state_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture, plan = self._write_valid_inputs(Path(temporary_directory))
            positions = np.load(plan / "trajectory_position.npy")
            positions[0, 0] += 0.1
            np.save(plan / "trajectory_position.npy", positions)
            with self.assertRaisesRegex(ValueError, "does not start"):
                load_pregrasp_replay(capture, plan)

    def test_normalizes_supported_dt_shapes(self):
        np.testing.assert_allclose(normalize_segment_dt(np.array(0.1), 3), [0.1, 0.1])
        np.testing.assert_allclose(
            normalize_segment_dt(np.array([0.1, 0.2]), 3), [0.1, 0.2]
        )
        np.testing.assert_allclose(
            normalize_segment_dt(np.array([0.1, 0.2, 9.0]), 3), [0.1, 0.2]
        )

    def test_physics_sampling_preserves_endpoints(self):
        positions = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        time, sampled = sample_positions_at_physics_rate(
            positions, np.array([0.1, 0.2]), 0.06
        )
        np.testing.assert_allclose(sampled[0], positions[0])
        np.testing.assert_allclose(sampled[-1], positions[-1])
        self.assertAlmostEqual(time[-1], 0.3)
        self.assertTrue(np.all(np.diff(time) > 0.0))

    def _write_grasp_lift_inputs(
        self, root: Path, *, include_transport: bool = False
    ) -> tuple[Path, Path]:
        capture = root / "capture"
        plan = root / "plan"
        capture.mkdir()
        plan.mkdir()
        names = [
            "panda_joint1",
            "panda_joint2",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        (capture / "robot_state.json").write_text(
            json.dumps({"joint_names": names}), encoding="utf-8"
        )
        np.save(
            capture / "panda_joint_positions.npy",
            np.array([0.1, -0.2, 0.04, 0.04]),
        )
        result = {
            "planner_reported_success": True,
            "trajectory_active_joint_names": names[:2],
            "transport": (
                {"planner_reported_success": True} if include_transport else None
            ),
        }
        safety = {
            "simulation_only": True,
            "final_approach_planned": True,
            "lift_planned": True,
            "trajectory_executed": False,
            "handover_transport_planned": include_transport,
            "held_object_collision_checked_during_transport": include_transport,
        }
        (plan / "grasp_lift_plan_check.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "result": result,
                    "automatic_checks": {"all": True},
                    "safety": safety,
                }
            ),
            encoding="utf-8",
        )
        phases = {
            "approach": np.array([[0.1, -0.2], [0.2, -0.1]]),
            "grasp": np.array([[0.2, -0.1], [0.3, 0.0]]),
            "lift": np.array([[0.3, 0.0], [0.4, 0.1]]),
        }
        if include_transport:
            phases["transport"] = np.array([[0.4, 0.1], [0.5, 0.2]])
        for phase, positions in phases.items():
            np.save(plan / f"{phase}_trajectory_position.npy", positions)
            np.save(plan / f"{phase}_trajectory_dt_s.npy", np.array(0.02))
        return capture, plan

    def test_loads_continuous_simulation_only_grasp_lift_phases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture, plan = self._write_grasp_lift_inputs(root)
            loaded = load_grasp_lift_replay(capture, plan)
            self.assertEqual(tuple(loaded.phase_positions), ("approach", "grasp", "lift"))
            np.testing.assert_allclose(loaded.phase_segment_dt_s["lift"], [0.02])

    def test_loads_optional_attached_transport_phase(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture, plan = self._write_grasp_lift_inputs(root, include_transport=True)
            loaded = load_grasp_lift_replay(capture, plan)
            self.assertEqual(
                tuple(loaded.phase_positions),
                ("approach", "grasp", "lift", "transport"),
            )
            np.testing.assert_allclose(
                loaded.phase_positions["transport"][0],
                loaded.phase_positions["lift"][-1],
            )

    def test_rejects_transport_without_attached_collision_gate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture, plan = self._write_grasp_lift_inputs(root, include_transport=True)
            report_path = plan / "grasp_lift_plan_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["safety"]["held_object_collision_checked_during_transport"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collision-check"):
                load_grasp_lift_replay(capture, plan)

    def test_rejects_discontinuous_grasp_lift_phases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture"
            plan = root / "plan"
            capture.mkdir()
            plan.mkdir()
            (capture / "robot_state.json").write_text(
                json.dumps({"joint_names": ["panda_joint1"]}), encoding="utf-8"
            )
            np.save(capture / "panda_joint_positions.npy", np.array([0.1]))
            (plan / "grasp_lift_plan_check.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "result": {
                            "planner_reported_success": True,
                            "trajectory_active_joint_names": ["panda_joint1"],
                        },
                        "automatic_checks": {"all": True},
                        "safety": {
                            "simulation_only": True,
                            "final_approach_planned": True,
                            "lift_planned": True,
                            "trajectory_executed": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            phase_values = {
                "approach": [[0.1], [0.2]],
                "grasp": [[0.5], [0.6]],
                "lift": [[0.6], [0.7]],
            }
            for phase, values in phase_values.items():
                np.save(plan / f"{phase}_trajectory_position.npy", np.array(values))
                np.save(plan / f"{phase}_trajectory_dt_s.npy", np.array(0.02))
            with self.assertRaisesRegex(ValueError, "discontinuous"):
                load_grasp_lift_replay(capture, plan)

    def test_isaac_replay_uses_position_targets_not_joint_teleportation(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_replay_pregrasp.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ArticulationAction(", script)
        self.assertIn("panda.apply_action(", script)
        self.assertNotIn("set_joint_positions(", script)

    def test_isaac_pregrasp_replay_supports_matching_authored_scene(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_replay_pregrasp.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--scene-usd"', script)
        self.assertIn("stage_utils.open_stage(str(scene_usd))", script)
        self.assertIn("SingleArticulation(", script)
        self.assertIn("replay scene does not match", script)
        self.assertIn('"replay_scene_matches_capture": True', script)
        self.assertIn("if scene_usd is None:", script)

    def test_grasp_lift_replay_uses_dynamic_target_and_physical_fingers(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_replay_grasp_lift.py"
        ).read_text(encoding="utf-8")
        self.assertIn("DynamicCuboid(", script)
        self.assertIn('finger_names = ("panda_finger_joint1", "panda_finger_joint2")', script)
        self.assertIn("PANDA_OPEN_FINGER_JOINT_M = 0.04", script)
        self.assertIn("open_fingers = np.full(2, args.open_finger_position_m", script)
        self.assertIn("args.closed_finger_position_m", script)
        self.assertIn('execute_phase("transport", closed_finger_target)', script)
        self.assertIn('final_phase = "transport" if transport_executed else "lift"', script)
        self.assertIn('"handover_release_executed": False', script)
        self.assertNotIn("open_finger_targets_rad", script)
        self.assertIn("ArticulationAction(", script)
        self.assertNotIn("FixedJoint", script)
        self.assertNotIn("set_joint_positions(", script)

    def test_grasp_lift_replay_supports_generic_authored_usd_target(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_replay_grasp_lift.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--scene-usd"', script)
        self.assertIn("stage_utils.open_stage(str(scene_usd))", script)
        self.assertIn("SingleArticulation(", script)
        self.assertIn("RigidPrim(", script)
        self.assertIn("reset_xform_properties=False", script)
        self.assertIn("Usd.PrimRange(target_root_prim)", script)
        self.assertIn("prim.HasAPI(UsdPhysics.RigidBodyAPI)", script)
        self.assertIn("compute_aabb(create_bbox_cache()", script)
        self.assertIn("target_settled_extent[2]", script)
        self.assertIn("replay scene does not match", script)
        self.assertNotIn("hammer", script.lower())

    def test_grasp_lift_replay_records_read_only_retention_diagnostics(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_replay_grasp_lift.py"
        ).read_text(encoding="utf-8")
        self.assertIn("target.get_masses()", script)
        self.assertIn('UsdPhysics.DriveAPI.Get(joint_prim, "linear")', script)
        self.assertIn("ComputeBoundMaterial(", script)
        self.assertIn('Tf.Token("physics")', script)
        self.assertIn('record_physics_sample("close")', script)
        self.assertIn('record_physics_sample("hold")', script)
        self.assertIn('output / "retention_finger_gap_m.npy"', script)
        self.assertIn('"peak_object_lift_m": peak_object_lift_m', script)
        self.assertNotIn("GetMaxForceAttr().Set", script)
        self.assertNotIn("GetStaticFrictionAttr().Set", script)


if __name__ == "__main__":
    unittest.main()
