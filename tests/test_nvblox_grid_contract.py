import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT
    / "ros2"
    / "panda_handover_nvblox"
    / "panda_handover_nvblox"
    / "grid_contract.py"
)
SPEC = importlib.util.spec_from_file_location("nvblox_grid_contract_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NvbloxGridContractTests(unittest.TestCase):
    def test_inclusive_center_aabb_yields_exact_160_cell_range(self):
        origin, first_center, center_span = MODULE.inclusive_voxel_center_aabb(
            (0.5, 0.0, 0.75), (1.6, 1.6, 1.6), 0.01
        )
        np.testing.assert_allclose(origin, (-0.3, -0.8, -0.05), atol=1e-12)
        np.testing.assert_allclose(first_center, (-0.295, -0.795, -0.045))
        np.testing.assert_allclose(center_span, (1.59, 1.59, 1.59))
        last_center = first_center + center_span
        indices = np.floor(
            np.stack((first_center, last_center), axis=0) / 0.01 + 1e-5
        ).astype(int)
        np.testing.assert_array_equal(indices[1] - indices[0] + 1, (160, 160, 160))

    def test_non_integral_extent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer number"):
            MODULE.inclusive_voxel_center_aabb((0, 0, 0), (1.605, 1.6, 1.6), 0.01)


if __name__ == "__main__":
    unittest.main()
