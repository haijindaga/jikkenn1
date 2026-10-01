import unittest

import numpy as np

from panda_handover.grasp_visualization import (
    CUROBO_PLAN_REJECTED,
    CUROBO_PLAN_SUCCESS,
    PLANNING_ERROR,
    STATIC_COLLISION_FREE,
    STATIC_COLLISION_REJECTED,
    classify_candidate_states,
    resolve_saved_gripper_identity,
    state_counts,
    verify_saved_world_grasps,
)


class GraspCandidateVisualizationTests(unittest.TestCase):
    def test_legacy_reports_resolve_to_historical_franka_default(self) -> None:
        self.assertEqual(
            resolve_saved_gripper_identity(
                {"parameters": {"gripper": "franka_panda"}},
                {"parameters": {"collision_threshold_m": 0.005}},
            ),
            "franka_panda",
        )

    def test_explicit_gripper_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "grippers disagree"):
            resolve_saved_gripper_identity(
                {"gripper": "robotiq_2f_85"},
                {"gripper": "franka_panda"},
            )

    def test_classifies_static_and_curobo_results_without_conflation(self) -> None:
        states = classify_candidate_states(
            np.array([False, True, True, True, True]),
            [
                {
                    "source_candidate_index": 1,
                    "return_code": 2,
                    "plan_status": "preflight_failed",
                    "available_for_physical_trial": False,
                },
                {
                    "source_candidate_index": 2,
                    "return_code": 0,
                    "plan_status": "success",
                    "available_for_physical_trial": True,
                },
                {
                    "source_candidate_index": 3,
                    "return_code": 1,
                    "plan_status": "missing_report",
                    "available_for_physical_trial": False,
                },
            ],
        )
        self.assertEqual(
            states.tolist(),
            [
                STATIC_COLLISION_REJECTED,
                CUROBO_PLAN_REJECTED,
                CUROBO_PLAN_SUCCESS,
                PLANNING_ERROR,
                STATIC_COLLISION_FREE,
            ],
        )
        self.assertEqual(sum(state_counts(states).values()), 5)

    def test_rejects_planning_attempt_for_statically_rejected_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "statically rejected"):
            classify_candidate_states(
                np.array([False]),
                [
                    {
                        "source_candidate_index": 0,
                        "return_code": 2,
                        "plan_status": "preflight_failed",
                        "available_for_physical_trial": False,
                    }
                ],
            )

    def test_verifies_saved_camera_to_world_transform(self) -> None:
        camera = np.repeat(np.eye(4)[None], 2, axis=0)
        camera[1, :3, 3] = [0.1, -0.2, 0.3]
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        world = np.einsum("ij,njk->nik", transform, camera)
        verify_saved_world_grasps(camera, world, transform)
        world[0, 0, 3] += 0.01
        with self.assertRaisesRegex(ValueError, "disagree"):
            verify_saved_world_grasps(camera, world, transform)


if __name__ == "__main__":
    unittest.main()
