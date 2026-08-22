import importlib.util
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "curobo_map_capture.py"
SPEC = importlib.util.spec_from_file_location("curobo_map_capture_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CuroboMapScriptTests(unittest.TestCase):
    def test_multiview_inputs_are_paired_in_order(self):
        captures = [Path("capture_0"), Path("capture_1"), Path("capture_2")]
        segmentations = [Path("sam3_0"), Path("sam3_1"), Path("sam3_2")]

        pairs = MODULE.pair_capture_inputs(captures, segmentations)

        self.assertEqual(
            pairs,
            list(zip(captures, segmentations)),
        )

    def test_mismatched_multiview_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            MODULE.pair_capture_inputs(
                [Path("capture_0"), Path("capture_1")],
                [Path("sam3_0")],
            )

    def test_empty_multiview_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            MODULE.pair_capture_inputs([], [])


if __name__ == "__main__":
    unittest.main()
