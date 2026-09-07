from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "anygrasp_infer_capture.py"
SPEC = importlib.util.spec_from_file_location("anygrasp_infer_capture_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FakeGraspGroup:
    def __init__(self, scores=(0.4, 0.9)):
        self.scores = np.asarray(scores, dtype=np.float32)
        count = len(self.scores)
        self.widths = np.full(count, 0.04, dtype=np.float32)
        self.heights = np.full(count, 0.03, dtype=np.float32)
        self.depths = np.full(count, 0.02, dtype=np.float32)
        self.translations = np.zeros((count, 3), dtype=np.float32)
        self.rotation_matrices = np.repeat(
            np.eye(3, dtype=np.float32)[None], count, axis=0
        )

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        indices = np.arange(len(self))[item]
        indices = np.atleast_1d(indices)
        result = _FakeGraspGroup(self.scores[indices])
        result.widths = self.widths[indices]
        result.heights = self.heights[indices]
        result.depths = self.depths[indices]
        result.translations = self.translations[indices]
        result.rotation_matrices = self.rotation_matrices[indices]
        return result

    def nms(self):
        return self

    def sort_by_score(self):
        order = np.argsort(-self.scores)
        return self[order]


class _FakeDetector:
    def __init__(self, observed):
        self.observed = observed

    def get_grasp(self, points, options):
        self.observed["points"] = points
        self.observed["options"] = options
        return _FakeGraspGroup()


class AnyGraspScriptTests(unittest.TestCase):
    def test_prepare_inputs_keeps_region_aligned_to_valid_points(self):
        points = np.array(
            [[[1, 2, 3], [np.nan, 0, 0]], [[4, 5, 6], [7, 8, 9]]],
            dtype=np.float32,
        )
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[True, True], [False, True]])
        flat, colors, region, valid = MODULE.prepare_anygrasp_inputs(
            points, rgb, mask
        )
        self.assertEqual(flat.shape, (3, 3))
        self.assertEqual(colors.shape, (3, 3))
        np.testing.assert_array_equal(region, [True, False, True])
        self.assertEqual(int(valid.sum()), 3)

    def test_official_region_steering_contract_and_outputs(self):
        observed = {}
        fake_gsnet = types.ModuleType("gsnet")
        fake_gsnet.__file__ = "/anygrasp/grasp_detection/gsnet.so"

        def create_detector(config):
            observed["config"] = config
            return _FakeDetector(observed)

        fake_gsnet.create_detector = create_detector
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            segmentation = root / "segmentation"
            output = root / "output"
            capture.mkdir()
            segmentation.mkdir()
            points = np.ones((2, 2, 3), dtype=np.float32)
            rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
            mask = np.array([[True, False], [True, False]])
            np.save(capture / "points_camera.npy", points)
            np.save(capture / "rgb.npy", rgb)
            np.save(segmentation / "union_mask.npy", mask)
            checkpoint = root / "checkpoint.tar"
            checkpoint.write_bytes(b"official-test-placeholder")
            argv = [
                str(SCRIPT),
                "--capture", str(capture),
                "--segmentation", str(segmentation),
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--min-region-points", "1",
                "--topk", "1",
            ]
            with patch.dict(sys.modules, {"gsnet": fake_gsnet}), patch.object(
                sys, "argv", argv
            ):
                result = MODULE.main()

            self.assertEqual(result, 0)
            self.assertEqual(observed["config"].max_gripper_width, 0.08)
            self.assertTrue(observed["options"]["collision_detection"])
            np.testing.assert_array_equal(
                observed["options"]["region_steering"], [True, False, True, False]
            )
            np.testing.assert_allclose(np.load(output / "scores.npy"), [0.9])
            self.assertTrue((output / "gripper_tips_camera.npy").is_file())
            report = __import__("json").loads(
                (output / "anygrasp_check.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "success")
            self.assertFalse(
                report["safety"]["panda_hand_frame_transform_validated"]
            )
            self.assertFalse(report["safety"]["safe_to_execute"])


if __name__ == "__main__":
    unittest.main()
