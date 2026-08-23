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

    def test_loads_continuous_simulation_only_grasp_lift_phases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
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
            report = {
                "status": "success",
                "result": {
                    "planner_reported_success": True,
                    "trajectory_active_joint_names": names[:2],
                },
                "automatic_checks": {"all": True},
                "safety": {
                    "simulation_only": True,
                    "final_approach_planned": True,
                    "lift_planned": True,
                    "trajectory_executed": False,
                },
            }
            (plan / "grasp_lift_plan_check.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            phases = {
                "approach": np.array([[0.1, -0.2], [0.2, -0.1]]),
                "grasp": np.array([[0.2, -0.1], [0.3, 0.0]]),
                "lift": np.array([[0.3, 0.0], [0.4, 0.1]]),
            }
            for phase, positions in phases.items():
                np.save(plan / f"{phase}_trajectory_position.npy", positions)
                np.save(plan / f"{phase}_trajectory_dt_s.npy", np.array(0.02))

            loaded = load_grasp_lift_replay(capture, plan)
            self.assertEqual(tuple(loaded.phase_positions), ("approach", "grasp", "lift"))
            np.testing.assert_allclose(loaded.phase_segment_dt_s["lift"], [0.02])

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
        self.assertNotIn("open_finger_targets_rad", script)
        self.assertIn("ArticulationAction(", script)
        self.assertNotIn("FixedJoint", script)
        self.assertNotIn("set_joint_positions(", script)


if __name__ == "__main__":
    unittest.main()
