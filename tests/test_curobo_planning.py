import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from panda_handover.curobo_planning import (
    CUROBO_COMMIT,
    CUROBO_VOXEL_PATCH_SHA256,
    classify_pregrasp_failure,
    load_backend_a_esdf,
    load_conservative_esdf,
    load_singleview_observed_pointcloud,
    prepare_pregrasp_goalset,
    rotation_matrix_to_quaternion_wxyz,
    summarize_ik_result_arrays,
    validate_voxel_fix_report,
)


def _backend_a_report(shape=(2, 3, 4), voxel=0.1):
    extent = np.asarray(shape, dtype=float) * voxel
    center = np.array([0.5, 0.0, 0.75])
    return {
        "status": "success",
        "backend": "A_nvblox_torch_dense_edt",
        "grid": {
            "shape_xyz": list(shape),
            "voxel_size_m": voxel,
            "extent_m": extent.tolist(),
            "center_robot_base_m": center.tolist(),
            "min_corner_robot_base_m": (center - 0.5 * extent).tolist(),
            "index_order": "x_slowest_z_fastest",
            "sdf_sign": "positive_free_negative_blocked",
        },
        "parameters": {"target_clear_applied": False},
        "automatic_checks": {"all": True},
        "unknown_environment_contract": {
            "only_sensor_observed_space_can_be_free": True,
            "unobserved_space_is_blocked": True,
            "distance_to_unknown_boundary_recomputed": True,
            "target_is_currently_blocked": True,
        },
        "safe_to_plan": False,
    }


def _backend_a_optimistic_report(shape=(2, 3, 4), voxel=0.1):
    report = _backend_a_report(shape=shape, voxel=voxel)
    report["parameters"].update(
        {"unknown_policy": "free", "planning_mode": "optimistic_sim"}
    )
    report["unknown_environment_contract"] = {
        "policy": "free",
        "only_sensor_observed_space_can_be_free": False,
        "unobserved_space_is_blocked": False,
        "unobserved_space_is_free": True,
        "distance_to_unknown_boundary_recomputed": True,
        "target_is_currently_blocked": False,
    }
    report["experiment_scope"] = {
        "simulation_only": True,
        "environment_visually_reviewed_by_operator": False,
        "safe_for_unknown_real_environment": False,
        "trajectory_execution_authorized": False,
    }
    return report


class CuroboPlanningTests(unittest.TestCase):
    def test_observed_pointcloud_loader_requires_reviewed_single_view_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "capture" / "camera_0"
            capture.mkdir(parents=True)
            points = np.array(
                [[0.4, 0.0, 0.7], [0.6, 0.1, 0.71]], dtype=np.float32
            )
            np.save(root / "occupied_points_robot_base.npy", points)
            report = {
                "status": "success",
                "reference": {
                    "apis": [
                        "RobotSegmenter",
                        "FilterDepth",
                        "Mapper.compute_esdf",
                    ]
                },
                "frames": {"map": "franka robot base"},
                "parameters": {
                    "input_frames": 1,
                    "voxel_size_m": 0.01,
                    "extent_m": [1.6, 1.6, 1.6],
                    "grid_center_robot_base_m": [0.5, 0.0, 0.75],
                },
                "counts": {"occupied_surface_voxels": 2},
                "views": [
                    {
                        "capture": str(capture),
                        "robot_mask_pixels": 20,
                        "target_mask_pixels": 10,
                    }
                ],
                "automatic_checks": {"all": True},
                "safe_to_plan": False,
                "unknown_environment_contract": {
                    "isaac_semantic_labels_used": False,
                    "isaac_ground_truth_obstacle_geometry_used": False,
                    "target_removed_with_sam3_mask": True,
                    "robot_removed_with_curobo_kinematics": True,
                    "unobserved_space_proven_occupied": False,
                },
            }
            (root / "esdf_check.json").write_text(json.dumps(report))

            loaded = load_singleview_observed_pointcloud(root, capture)

            np.testing.assert_array_equal(loaded.points_robot_base_m, points)
            self.assertEqual(loaded.voxel_size_m, 0.01)

            report["parameters"]["input_frames"] = 2
            (root / "esdf_check.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "exactly one view"):
                load_singleview_observed_pointcloud(root, capture)

    def test_observed_pointcloud_loader_rejects_failed_removal_or_count_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "capture"
            capture.mkdir()
            np.save(
                root / "occupied_points_robot_base.npy",
                np.array([[0.5, 0.0, 0.7]], dtype=np.float32),
            )
            report = {
                "status": "success",
                "reference": {
                    "apis": ["RobotSegmenter", "FilterDepth", "Mapper.compute_esdf"]
                },
                "frames": {"map": "franka robot base"},
                "parameters": {
                    "input_frames": 1,
                    "voxel_size_m": 0.01,
                    "extent_m": [1.6, 1.6, 1.6],
                    "grid_center_robot_base_m": [0.5, 0.0, 0.75],
                },
                "counts": {"occupied_surface_voxels": 2},
                "views": [
                    {
                        "capture": str(capture),
                        "robot_mask_pixels": 0,
                        "target_mask_pixels": 10,
                    }
                ],
                "automatic_checks": {"all": True},
                "safe_to_plan": False,
                "unknown_environment_contract": {
                    "isaac_semantic_labels_used": False,
                    "isaac_ground_truth_obstacle_geometry_used": False,
                    "target_removed_with_sam3_mask": True,
                    "robot_removed_with_curobo_kinematics": True,
                    "unobserved_space_proven_occupied": False,
                },
            }
            (root / "esdf_check.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "robot pixels"):
                load_singleview_observed_pointcloud(root, capture)
            report["views"][0]["robot_mask_pixels"] = 10
            (root / "esdf_check.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "count"):
                load_singleview_observed_pointcloud(root, capture)

    def test_optimistic_loader_requires_exact_evidence_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = _backend_a_optimistic_report()
            (root / "esdf_check.json").write_text(json.dumps(report))
            observed = np.zeros((2, 3, 4), dtype=bool)
            observed[0, 0, 0] = True
            sensor_free = np.zeros_like(observed)
            planning_free = ~observed
            features = np.where(planning_free, 0.05, -0.05).astype(np.float32)
            np.save(root / "observed_mask.npy", observed)
            np.save(root / "sensor_known_free_mask.npy", sensor_free)
            np.save(root / "known_free_mask.npy", planning_free)
            np.save(root / "esdf_features.npy", features)
            loaded = load_backend_a_esdf(root, expected_unknown_policy="free")
            self.assertEqual(loaded.unknown_policy, "free")
            planning_free[0, 0, 0] = True
            np.save(root / "known_free_mask.npy", planning_free)
            np.save(
                root / "esdf_features.npy",
                np.where(planning_free, 0.05, -0.05).astype(np.float32),
            )
            with self.assertRaisesRegex(ValueError, "observed obstacles"):
                load_backend_a_esdf(root, expected_unknown_policy="free")

    def test_loader_rejects_unknown_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "esdf_check.json").write_text(
                json.dumps(_backend_a_optimistic_report())
            )
            with self.assertRaisesRegex(ValueError, "expected blocked"):
                load_backend_a_esdf(root, expected_unknown_policy="blocked")

    def test_ik_summary_counts_official_result_fields(self):
        summary = summarize_ik_result_arrays(
            np.array([[False, True, True]]),
            feasible=np.array([[True, True, False]]),
            position_error=np.array([[0.2, 0.003, np.inf]]),
            rotation_error=np.array([[0.4, 0.02, 0.1]]),
            goalset_index=np.array([[4, 2, 7]]),
        )
        self.assertEqual(summary["returned_seed_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["feasible_count"], 2)
        self.assertEqual(summary["successful_goalset_indices"], [2, 7])
        self.assertAlmostEqual(summary["minimum_position_error_m"], 0.003)

    def test_failure_classifier_separates_world_collision_from_geometric_ik(self):
        self.assertEqual(
            classify_pregrasp_failure(
                planner_success=False,
                world_ik_success_count=0,
                free_world_ik_success_count=2,
                start_penetrating_sphere_count=3,
                planner_returned_result=False,
            ),
            "world_collision_rejects_ik_and_start_state_penetrates_map",
        )
        self.assertEqual(
            classify_pregrasp_failure(
                planner_success=False,
                world_ik_success_count=0,
                free_world_ik_success_count=0,
                start_penetrating_sphere_count=0,
                planner_returned_result=False,
            ),
            "pregrasp_ik_fails_even_without_world_collision",
        )

    def test_backend_a_loader_enforces_sign_and_grid_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = _backend_a_report()
            (root / "esdf_check.json").write_text(json.dumps(report))
            known = np.zeros((2, 3, 4), dtype=bool)
            known[0] = True
            features = np.where(known, 0.05, -0.05).astype(np.float32)
            np.save(root / "known_free_mask.npy", known)
            np.save(root / "esdf_features.npy", features)
            loaded = load_conservative_esdf(root)
            self.assertEqual(loaded.shape_xyz, (2, 3, 4))
            np.testing.assert_array_equal(loaded.known_free, known)

    def test_backend_a_loader_rejects_unknown_marked_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "esdf_check.json").write_text(json.dumps(_backend_a_report()))
            known = np.zeros((2, 3, 4), dtype=bool)
            features = np.full((2, 3, 4), -0.05, dtype=np.float32)
            features[1, 1, 1] = 0.05
            np.save(root / "known_free_mask.npy", known)
            np.save(root / "esdf_features.npy", features)
            with self.assertRaisesRegex(ValueError, "blocked voxels"):
                load_conservative_esdf(root)

    def test_pregrasp_is_negative_tool_z_and_score_ordered(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [0.1, 0.2, 0.3]
        goalset = prepare_pregrasp_goalset(
            poses,
            np.array([0.2, 0.9, 0.5]),
            np.eye(4),
            approach_offset_m=0.15,
            max_candidates=2,
            candidate_indices=np.array([10, 11, 12]),
        )
        np.testing.assert_array_equal(goalset.candidate_indices, [11, 12])
        np.testing.assert_allclose(goalset.pregrasp_robot_base[:, 2, 3], -0.15)
        np.testing.assert_allclose(goalset.grasp_robot_base[:, 0, 3], [0.2, 0.3])

    def test_pregrasp_offset_rotates_with_tool(self):
        pose = np.eye(4)[None]
        pose[0, :3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        goalset = prepare_pregrasp_goalset(
            pose, np.array([1.0]), np.eye(4), approach_offset_m=0.1
        )
        np.testing.assert_allclose(
            goalset.pregrasp_robot_base[0, :3, 3], [0.0, 0.1, 0.0], atol=1e-7
        )

    def test_rotation_conversion_uses_wxyz(self):
        rotations = np.stack(
            [
                np.eye(3),
                np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            ]
        )
        quaternions = rotation_matrix_to_quaternion_wxyz(rotations)
        np.testing.assert_allclose(quaternions[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quaternions[1], [0.0, 1.0, 0.0, 0.0])

    def test_voxel_fix_report_requires_exact_source_hash(self):
        report = {
            "status": "success",
            "reference": {
                "curobo_commit": CUROBO_COMMIT,
                "patched_source_sha256": CUROBO_VOXEL_PATCH_SHA256,
            },
            "automatic_checks": {"blocked": True, "free": True},
            "safe_to_load_real_esdf": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "check.json"
            path.write_text(json.dumps(report))
            self.assertEqual(validate_voxel_fix_report(path), report)
            report["reference"]["patched_source_sha256"] = "0" * 64
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "unreviewed"):
                validate_voxel_fix_report(path)


if __name__ == "__main__":
    unittest.main()
