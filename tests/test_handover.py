import unittest

import numpy as np

from panda_handover.handover import (
    generate_affordance_handover_goals,
    generate_gravity_tilt_preserving_handover_goals,
    generate_orientation_preserving_handover_goal,
    gravity_tilt_deviation_rad,
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

    def test_orientation_preserving_goal_keeps_rotation_and_places_receive_part(self):
        grasp_transform = np.eye(4)
        grasp_transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        grasp_transform[:3, 3] = [0.4, 0.1, 0.2]
        receive_part = np.array(
            [[0.5, 0.09, 0.2], [0.5, 0.11, 0.2], [0.5, 0.1, 0.2]]
        )
        receiver = np.array([0.55, -0.3, 0.65])

        goals, report = generate_orientation_preserving_handover_goal(
            grasp_transform, receive_part, receiver
        )

        self.assertEqual(goals.shape, (1, 4, 4))
        np.testing.assert_allclose(
            goals[0, :3, :3], grasp_transform[:3, :3], atol=1e-7
        )
        receive_center_hand = np.asarray(report["receive_part_center_panda_hand_m"])
        np.testing.assert_allclose(
            goals[0, :3, :3] @ receive_center_hand + goals[0, :3, 3],
            receiver,
            atol=1e-6,
        )
        self.assertFalse(report["human_direction_alignment_enforced"])

    def test_gravity_tilt_goals_keep_tilt_allow_yaw_and_place_receive_part(self):
        angle = np.deg2rad(20.0)
        grasp_transform = np.eye(4)
        grasp_transform[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        grasp_transform[:3, 3] = [0.4, 0.1, 0.2]
        receive_part = np.array(
            [[0.5, 0.09, 0.2], [0.5, 0.11, 0.2], [0.5, 0.1, 0.2]]
        )
        receiver = np.array([-0.3, 0.0, 0.35])

        goals, report = generate_gravity_tilt_preserving_handover_goals(
            grasp_transform,
            receive_part,
            receiver,
            yaw_degrees=(0.0, 90.0, 180.0, 270.0),
        )

        self.assertEqual(goals.shape, (4, 4, 4))
        receive_center_hand = np.asarray(report["receive_part_center_panda_hand_m"])
        for goal in goals:
            np.testing.assert_allclose(
                goal[:3, :3] @ receive_center_hand + goal[:3, 3],
                receiver,
                atol=1e-6,
            )
        deviations = gravity_tilt_deviation_rad(
            goals[:, :3, :3], grasp_transform[:3, :3]
        )
        np.testing.assert_allclose(deviations, 0.0, atol=1e-7)
        self.assertEqual(report["yaw_degrees"], [0.0, 90.0, 180.0, 270.0])

    def test_gravity_tilt_deviation_rejects_changed_tilt_but_not_world_yaw(self):
        reference = np.eye(3)
        yaw = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        angle = np.deg2rad(10.0)
        tilted = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        deviations = gravity_tilt_deviation_rad(
            np.stack((reference, yaw, tilted)), reference
        )
        np.testing.assert_allclose(deviations[:2], 0.0, atol=1e-7)
        self.assertAlmostEqual(deviations[2], angle, places=7)


if __name__ == "__main__":
    unittest.main()
