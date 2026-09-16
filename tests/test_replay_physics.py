import unittest
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from panda_handover.replay_physics import (
    configure_replay_physics, validate_replay_physics,
)


class FakeContext:
    device = "cuda:0"
    use_gpu_pipeline = True
    use_fabric = True
    gpu = True

    def is_gpu_dynamics_enabled(self):
        return self.gpu

    def set_physics_sim_device(self, device):
        self.device = device
        self.use_gpu_pipeline = device != "cpu"

    def enable_gpu_dynamics(self, enabled):
        self.gpu = enabled

    def enable_fabric(self, enabled):
        self.use_fabric = enabled


class ReplayPhysicsTests(unittest.TestCase):
    def test_trial_runner_accepts_cpu_and_keeps_default_unchanged(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "isaac_replay_grasp_lift_trials.py"
        namespace = runpy.run_path(str(script), run_name="trial_runner_test")
        argv = [str(script), "--capture", "capture", "--plan-trials", "plans",
                "--scene-usd", "scene.usda", "--output", "out", "--simulation-only"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(namespace["parse_args"]().replay_physics, "default")
        with patch.object(sys, "argv", argv + ["--replay-physics", "cpu"]):
            self.assertEqual(namespace["parse_args"]().replay_physics, "cpu")

    def test_default_preserves_all_settings(self):
        context = FakeContext()
        report = configure_replay_physics(context, None, "default")
        self.assertEqual(report["before"], report["configured"])
        self.assertEqual(context.device, "cuda:0")

    def test_cpu_disables_gpu_pipeline_dynamics_and_fabric(self):
        context = FakeContext()
        report = configure_replay_physics(context, context, "cpu")
        self.assertEqual(report["configured"], {
            "device": "cpu", "gpu_pipeline_enabled": False,
            "gpu_dynamics_enabled": False, "fabric_enabled": False,
        })

    def test_rejects_incomplete_cpu_override(self):
        with self.assertRaisesRegex(RuntimeError, "not applied"):
            validate_replay_physics("cpu", {
                "device": "cpu", "gpu_pipeline_enabled": False,
                "gpu_dynamics_enabled": False, "fabric_enabled": True,
            })

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            configure_replay_physics(FakeContext(), None, "gpu")
