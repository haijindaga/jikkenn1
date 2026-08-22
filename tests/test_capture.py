import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from panda_handover.capture import RgbdCapture


def make_capture() -> RgbdCapture:
    return RgbdCapture(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        intrinsics=np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]]),
        T_world_camera=np.eye(4),
    )


class CaptureTests(unittest.TestCase):
    def test_capture_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = make_capture().save(Path(temporary_directory), write_previews=False)
            self.assertTrue(np.array_equal(np.load(output / "rgb.npy"), make_capture().rgb))
            self.assertTrue(np.array_equal(np.load(output / "depth_m.npy"), make_capture().depth_m))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["depth_unit"], "metre")
            self.assertEqual(metadata["valid_depth_pixels"], 6)

    def test_capture_rejects_mismatched_depth_shape(self):
        capture = make_capture()
        invalid = RgbdCapture(
            rgb=capture.rgb,
            depth_m=np.ones((1, 1), dtype=np.float32),
            intrinsics=capture.intrinsics,
            T_world_camera=capture.T_world_camera,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            invalid.validate()
