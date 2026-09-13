from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "isaac_surface_gripper_preflight.py"
SPEC = importlib.util.spec_from_file_location("isaac_surface_gripper_preflight_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _CompleteInterface:
    def close_gripper(self):
        pass

    def open_gripper(self):
        pass

    def get_gripper_status(self):
        pass

    def get_gripped_objects(self):
        pass

    def set_write_to_usd(self):
        pass


class SurfaceGripperPreflightTests(unittest.TestCase):
    def test_callable_contract_requires_documented_interface(self):
        contract = MODULE.callable_contract(
            _CompleteInterface(), MODULE.REQUIRED_INTERFACE_METHODS
        )
        self.assertTrue(all(contract.values()))
        incomplete = MODULE.callable_contract(object(), MODULE.REQUIRED_INTERFACE_METHODS)
        self.assertFalse(any(incomplete.values()))

    def test_example_discovery_and_relevant_line_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "nested" / "gripper_grasp.py"
            example.parent.mkdir()
            example.write_text(
                "import unrelated\n"
                "iface.close_gripper('/World/SurfaceGripper')\n"
                "print('done')\n",
                encoding="utf-8",
            )
            (root / "SurfaceGripper_gantry.usda").write_text("#usda 1.0\n")

            matches = MODULE.find_named_files([root], MODULE.EXAMPLE_FILENAMES)
            self.assertEqual(
                {path.name for path in matches},
                {"gripper_grasp.py", "SurfaceGripper_gantry.usda"},
            )
            lines = MODULE.extract_relevant_source_lines(example)
            self.assertEqual(
                lines,
                [{"line": 2, "text": "iface.close_gripper('/World/SurfaceGripper')"}],
            )


if __name__ == "__main__":
    unittest.main()
