import unittest
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from panda_handover.replay_physics import (
    configure_replay_physics, validate_replay_physics, replay_physics_state,
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
    def test_isaac_51_broken_fabric_property_uses_extension_state(self):
        class BrokenFabricContext(FakeContext):
            fabric_enabled = True

            def is_fabric_enabled(self, enable):
                return self.fabric_enabled

            @property
            def use_fabric(self):
                return self.is_fabric_enabled()

            def enable_fabric(self, enabled):
                self.fabric_enabled = enabled

        context = BrokenFabricContext()
        with patch(
            "panda_handover.replay_physics._fabric_extension_enabled",
            side_effect=lambda: context.fabric_enabled,
        ) as reader:
            report = configure_replay_physics(context, context, "cpu")
            self.assertTrue(report["before"]["fabric_enabled"])
            self.assertFalse(report["configured"]["fabric_enabled"])
            validate_replay_physics("cpu", replay_physics_state(context))
            self.assertEqual(reader.call_count, 3)

    def test_fallback_checks_exact_physics_fabric_extension(self):
        from types import ModuleType
        from unittest.mock import Mock
        from panda_handover.replay_physics import _fabric_extension_enabled

        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        app = ModuleType("omni.kit.app")
        omni.kit = kit
        kit.app = app
        manager = Mock()
        app.get_app = Mock()
        app.get_app.return_value.get_extension_manager.return_value = manager
        with patch.dict(sys.modules, {
            "omni": omni, "omni.kit": kit, "omni.kit.app": app,
        }):
            for enabled in (True, False):
                manager.is_extension_enabled.return_value = enabled
                self.assertEqual(_fabric_extension_enabled(), enabled)
                manager.is_extension_enabled.assert_called_with("omni.physx.fabric")

    def test_unrelated_getter_failure_is_not_hidden(self):
        class BadContext(FakeContext):
            @property
            def use_fabric(self):
                raise TypeError("unrelated failure")

        with self.assertRaisesRegex(TypeError, "unrelated failure"):
            replay_physics_state(BadContext())

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
