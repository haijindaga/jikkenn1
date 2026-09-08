import unittest

import numpy as np

from panda_handover.handover import (
    generate_affordance_handover_goals,
    receive_clear_score_order,
)


class HandoverGeometryTests(unittest.TestCase):
    def test_receive_clear_gate_precedes_stable_graspgenx_score_order(self):
        order = receive_clear_score_order(
            np.array([0.7, 0.9, 0.9, 0.8]),
            np.array([True, False, True, True]),
        )
        np.testing.assert_array_equal(order, [2, 3, 0])

    def test_receive_part_is_placed_at_receiver_and_axis_points_to_human(self):
        grasp_transform = np.eye(4)
        grasp_transform[:3, 3] = [0.4, 0.0, 0.2]
        grasp_part = np.array(
            [[0.39, -0.01, 0.2], [0.41, 0.01, 0.2], [0.4, 0.0, 0.2]]
        )
        receive_part = grasp_part + np.array([0.2, 0.0, 0.0])
        receiver = np.array([0.55, -0.3, 0.65])
        human_direction = np.array([0.0, -2.0, 0.0])

        goals, report = generate_affordance_handover_goals(
            grasp_transform,
            grasp_part,
            receive_part,
            receiver,
            human_direction,
        )

        self.assertEqual(goals.shape, (6, 4, 4))
        grasp_center_hand = np.asarray(report["grasp_part_center_panda_hand_m"])
        receive_center_hand = np.asarray(report["receive_part_center_panda_hand_m"])
        expected_direction = human_direction / np.linalg.norm(human_direction)
        for goal in goals:
            np.testing.assert_allclose(
                goal[:3, :3] @ receive_center_hand + goal[:3, 3],
                receiver,
                atol=1e-6,
            )
            presented_axis = goal[:3, :3] @ (
                receive_center_hand - grasp_center_hand
            )
            presented_axis /= np.linalg.norm(presented_axis)
            np.testing.assert_allclose(presented_axis, expected_direction, atol=1e-6)
            np.testing.assert_allclose(
                goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-6
            )
            self.assertAlmostEqual(np.linalg.det(goal[:3, :3]), 1.0, places=6)

    def test_degenerate_part_axis_fails_closed(self):
        points = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        with self.assertRaisesRegex(ValueError, "grasp-to-receive"):
            generate_affordance_handover_goals(
                np.eye(4), points, points, [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]
            )


if __name__ == "__main__":
    unittest.main()
