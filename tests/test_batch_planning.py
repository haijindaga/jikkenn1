import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from panda_handover.batch_planning import (
    candidate_chunks,
    compare_batch_to_sequential,
    pad_candidate_indices,
    reduce_batch_success,
)


class BatchPlanningTests(unittest.TestCase):
    def test_candidate_chunks_preserve_order_and_remainder(self):
        chunks = [list(value) for value in candidate_chunks(10, 4)]
        self.assertEqual(chunks, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_reduce_batch_success_ignores_padding_and_reduces_seeds(self):
        values = np.array(
            [[False, True], [False, False], [True, False], [True, True]]
        )
        self.assertEqual(reduce_batch_success(values, 3), [True, False, True])

    def test_partial_final_batch_is_padded_without_reordering_real_candidates(self):
        self.assertEqual(pad_candidate_indices([32, 33], 16)[:2], [32, 33])
        self.assertEqual(len(pad_candidate_indices([32, 33], 16)), 16)
        self.assertEqual(set(pad_candidate_indices([32, 33], 16)[2:]), {33})

    def test_padding_rejects_empty_or_oversized_chunks(self):
        with self.assertRaises(ValueError):
            pad_candidate_indices([], 16)
        with self.assertRaises(ValueError):
            pad_candidate_indices([0, 1, 2], 2)

    def test_comparison_never_claims_full_pipeline_equivalence(self):
        report = compare_batch_to_sequential(
            [
                {"source_candidate_index": 7, "batch_grasp_lift_success": True},
                {"source_candidate_index": 9, "batch_grasp_lift_success": False},
            ],
            [
                {
                    "source_candidate_index": 7,
                    "plan_status": "success",
                    "planner_feasible": True,
                },
                {
                    "source_candidate_index": 9,
                    "plan_status": "planning_failed",
                    "planner_feasible": False,
                },
            ],
        )
        self.assertEqual(report["status"], "feasibility_equivalent")
        self.assertFalse(report["full_pipeline_equivalence_established"])
        self.assertTrue(report["remaining_equivalence_gates"])

    def test_comparison_fails_visible_on_mismatch(self):
        report = compare_batch_to_sequential(
            [{"source_candidate_index": 7, "batch_grasp_lift_success": False}],
            [
                {
                    "source_candidate_index": 7,
                    "plan_status": "success",
                    "planner_feasible": True,
                }
            ],
        )
        self.assertEqual(report["status"], "feasibility_mismatch")
        self.assertEqual(len(report["mismatches"]), 1)

    def test_handover_failure_still_counts_as_sequential_grasp_lift_success(self):
        report = compare_batch_to_sequential(
            [{"source_candidate_index": 21, "batch_grasp_lift_success": True}],
            [
                {
                    "source_candidate_index": 21,
                    "plan_status": "handover_transport_planning_failed",
                    "planner_feasible": False,
                }
            ],
        )
        self.assertEqual(report["status"], "feasibility_equivalent")
        candidate = report["candidates"][0]
        self.assertTrue(candidate["sequential_grasp_lift_success"])
        self.assertFalse(candidate["sequential_full_plan_feasible"])

    def test_unknown_sequential_status_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unclassified"):
            compare_batch_to_sequential(
                [{"source_candidate_index": 1, "batch_grasp_lift_success": True}],
                [
                    {
                        "source_candidate_index": 1,
                        "plan_status": "missing_report",
                        "planner_feasible": False,
                    }
                ],
            )

    def test_batch_route_is_diagnostic_and_does_not_replace_baseline(self):
        batch_source = (
            PROJECT / "scripts" / "curobo_batch_grasp_lift_diagnostic.py"
        ).read_text(encoding="utf-8")
        trial_source = (
            PROJECT / "scripts" / "curobo_plan_grasp_lift_trials.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"',
            batch_source,
        )
        self.assertIn("BatchMotionPlanner", batch_source)
        self.assertIn('"diagnosis_only": True', batch_source)
        self.assertIn('"replayable_trajectory_saved": False', batch_source)
        self.assertNotIn("curobo_batch_grasp_lift_diagnostic", trial_source)


if __name__ == "__main__":
    unittest.main()
