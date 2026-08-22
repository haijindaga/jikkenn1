import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


class _FakeSweepVolumeParams:
    @classmethod
    def from_gripper_config(cls, gripper_name):
        if gripper_name != "franka_panda":
            raise AssertionError(gripper_name)
        return cls()

    def to_dict(self):
        return {
            "extents_open": np.ones(3, dtype=np.float32),
            "offset_open": np.zeros(3, dtype=np.float32),
            "extents_mid": np.ones(3, dtype=np.float32),
            "offset_mid": np.zeros(3, dtype=np.float32),
        }


class _FakeClient:
    def __init__(self, host, port, timeout_ms):
        self.server_metadata = {
            "actions": ["health", "metadata", "infer_scene_pc"],
            "precision": {"weights": "fp32"},
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def health(self):
        return {"status": "ok"}

    def infer_scene_pc(self, **kwargs):
        if kwargs["point_cloud"].shape != (2, 2, 3):
            raise AssertionError(kwargs["point_cloud"].shape)
        if kwargs["instance_mask"].dtype != np.int32:
            raise AssertionError(kwargs["instance_mask"].dtype)
        return {
            1: (
                np.eye(4, dtype=np.float32)[None],
                np.array([0.9], dtype=np.float32),
                ["diff"],
            )
        }


class GraspGenXScriptTests(unittest.TestCase):
    def test_official_client_contract_saves_frame_explicit_results(self):
        project = Path(__file__).resolve().parents[1]
        script_path = project / "scripts" / "graspgenx_infer_capture.py"
        fake_modules = {
            "graspgenx": types.ModuleType("graspgenx"),
            "graspgenx.serving": types.ModuleType("graspgenx.serving"),
            "graspgenx.serving.types": types.ModuleType("graspgenx.serving.types"),
            "graspgenx.serving.zmq_client": types.ModuleType(
                "graspgenx.serving.zmq_client"
            ),
        }
        fake_modules["graspgenx.serving.types"].SweepVolumeParams = _FakeSweepVolumeParams
        fake_modules["graspgenx.serving.zmq_client"].GraspGenXClient = _FakeClient

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            segmentation = root / "segmentation"
            output = root / "output"
            capture.mkdir()
            segmentation.mkdir()
            np.save(capture / "points_camera.npy", np.ones((2, 2, 3), dtype=np.float32))
            np.save(capture / "T_world_camera.npy", np.eye(4))
            np.save(segmentation / "union_mask.npy", np.ones((2, 2), dtype=bool))

            argv = [
                str(script_path),
                "--capture", str(capture),
                "--segmentation", str(segmentation),
                "--output", str(output),
                "--min-object-points", "1",
            ]
            with patch.dict(sys.modules, fake_modules), patch.object(sys, "argv", argv):
                spec = importlib.util.spec_from_file_location("graspgenx_capture_test", script_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                result = module.main()

            self.assertEqual(result, 0)
            self.assertTrue((output / "grasps_camera.npy").is_file())
            self.assertTrue((output / "panda_hand_world.npy").is_file())


if __name__ == "__main__":
    unittest.main()
