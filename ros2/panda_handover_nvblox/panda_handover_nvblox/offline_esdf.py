"""Publish saved filtered depth to Isaac ROS nvblox and save its official ESDF response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from geometry_msgs.msg import Point, TransformStamped, Vector3
from nvblox_msgs.srv import EsdfAndGradients
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from panda_handover_nvblox.grid_contract import (
    fingerprint_files,
    inclusive_voxel_center_aabb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--prepared-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--extent", type=float, nargs=3, default=(1.6, 1.6, 1.6))
    parser.add_argument("--grid-center", type=float, nargs=3, default=(0.5, 0.0, 0.75))
    parser.add_argument("--publish-count", type=int, default=20)
    parser.add_argument("--publish-period", type=float, default=0.1)
    parser.add_argument("--service-timeout", type=float, default=30.0)
    return parser.parse_args()


def quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix without adding a SciPy dependency."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(
        matrix.T @ matrix, np.eye(3), atol=1e-5
    ):
        raise ValueError("camera rotation must be orthonormal")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ]
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def validate_prepared_view_order(
    prepared_map: Path, captures: list[Path]
) -> list[Path]:
    """Require the exact filtered views recorded by curobo_map_capture.py."""
    report_path = prepared_map / "esdf_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing prepared-map report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    views = report.get("views")
    if not isinstance(views, list) or len(views) != len(captures):
        raise ValueError("prepared-map view count does not match --capture count")
    depth_paths = []
    for index, (capture, view) in enumerate(zip(captures, views)):
        recorded = view.get("capture") if isinstance(view, dict) else None
        if not isinstance(recorded, str) or Path(recorded).resolve() != capture.resolve():
            raise ValueError(
                f"prepared-map view {index} does not match capture {capture}"
            )
        depth_path = (
            prepared_map / "views" / f"camera_{index}" / "mapping_depth_m.npy"
        )
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth_paths.append(depth_path)
    return depth_paths


class OfflineEsdfNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("panda_handover_offline_esdf")
        self.args = args
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.depth_publishers = []
        self.info_publishers = []
        self.depth_arrays = []
        self.intrinsics = []
        self.transforms = []
        self.frame_ids = []
        self.static_broadcaster = StaticTransformBroadcaster(self)
        depth_paths = validate_prepared_view_order(args.prepared_map, args.capture)

        for index, (capture, depth_path) in enumerate(zip(args.capture, depth_paths)):
            depth = np.load(depth_path).astype(np.float32, copy=False)
            intrinsics = np.load(capture / "intrinsics.npy").astype(
                np.float64, copy=False
            )
            transform = np.load(capture / "T_robot_base_camera.npy").astype(
                np.float64, copy=False
            )
            if depth.ndim != 2 or intrinsics.shape != (3, 3) or transform.shape != (4, 4):
                raise ValueError(f"camera_{index}: malformed saved geometry")
            self.depth_arrays.append(np.ascontiguousarray(depth))
            self.intrinsics.append(intrinsics)
            self.transforms.append(transform)
            self.frame_ids.append(f"saved_camera_{index}_optical")
            self.depth_publishers.append(
                self.create_publisher(Image, f"camera_{index}/depth/image", qos)
            )
            self.info_publishers.append(
                self.create_publisher(
                    CameraInfo, f"camera_{index}/depth/camera_info", qos
                )
            )
        self.client = self.create_client(
            EsdfAndGradients, "/nvblox_node/get_esdf_and_gradient"
        )

    def publish_static_transforms(self) -> None:
        messages = []
        stamp = self.get_clock().now().to_msg()
        for frame_id, transform in zip(self.frame_ids, self.transforms):
            quaternion = quaternion_wxyz_from_rotation(transform[:3, :3])
            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = "panda_link0"
            message.child_frame_id = frame_id
            message.transform.translation.x = float(transform[0, 3])
            message.transform.translation.y = float(transform[1, 3])
            message.transform.translation.z = float(transform[2, 3])
            message.transform.rotation.w = float(quaternion[0])
            message.transform.rotation.x = float(quaternion[1])
            message.transform.rotation.y = float(quaternion[2])
            message.transform.rotation.z = float(quaternion[3])
            messages.append(message)
        self.static_broadcaster.sendTransform(messages)

    def publish_views_once(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for index, (depth, intrinsics, frame_id) in enumerate(
            zip(self.depth_arrays, self.intrinsics, self.frame_ids)
        ):
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = frame_id
            image.height, image.width = depth.shape
            image.encoding = "32FC1"
            image.is_bigendian = False
            image.step = int(depth.shape[1] * depth.dtype.itemsize)
            image.data = depth.tobytes()

            info = CameraInfo()
            info.header = image.header
            info.height, info.width = depth.shape
            info.distortion_model = "plumb_bob"
            info.d = []
            info.k = intrinsics.reshape(-1).tolist()
            info.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
            info.p = [
                float(intrinsics[0, 0]), 0.0, float(intrinsics[0, 2]), 0.0,
                0.0, float(intrinsics[1, 1]), float(intrinsics[1, 2]), 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            self.info_publishers[index].publish(info)
            self.depth_publishers[index].publish(image)

    def request_esdf(self):
        if not self.client.wait_for_service(timeout_sec=self.args.service_timeout):
            raise RuntimeError("nvblox ESDF service was not available")
        extent = np.asarray(self.args.extent, dtype=np.float64)
        center = np.asarray(self.args.grid_center, dtype=np.float64)
        # nvblox's dense-grid conversion includes both AABB endpoints.  Ask
        # for the first and last voxel centres so an N-voxel physical extent
        # produces exactly N cells, as documented by EsdfAndGradients.srv.
        _, minimum_center, center_span = inclusive_voxel_center_aabb(
            center, extent, self.args.voxel_size
        )
        request = EsdfAndGradients.Request()
        request.update_esdf = True
        request.visualize_esdf = False
        request.use_aabb = True
        request.frame_id = "panda_link0"
        request.aabb_min_m = Point(
            x=float(minimum_center[0]),
            y=float(minimum_center[1]),
            z=float(minimum_center[2]),
        )
        request.aabb_size_m = Vector3(
            x=float(center_span[0]),
            y=float(center_span[1]),
            z=float(center_span[2]),
        )
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.args.service_timeout
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("nvblox ESDF service timed out")
        return future.result()


def corner_probe_indices_outside_all_views(
    args: argparse.Namespace, origin_m: np.ndarray, shape: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    """Return AABB corner voxels that cannot project into any saved camera."""
    probes = []
    for index in (
        (x, y, z)
        for x in (0, shape[0] - 1)
        for y in (0, shape[1] - 1)
        for z in (0, shape[2] - 1)
    ):
        point_robot = origin_m + (np.asarray(index, dtype=np.float64) + 0.5) * args.voxel_size
        visible = False
        for capture in args.capture:
            intrinsics = np.load(capture / "intrinsics.npy").astype(
                np.float64, copy=False
            )
            transform = np.load(capture / "T_robot_base_camera.npy").astype(
                np.float64, copy=False
            )
            depth_shape = np.load(capture / "depth_m.npy", mmap_mode="r").shape
            point_camera = (
                transform[:3, :3].T @ (point_robot - transform[:3, 3])
            )
            if point_camera[2] <= 0.0:
                continue
            u = intrinsics[0, 0] * point_camera[0] / point_camera[2] + intrinsics[0, 2]
            v = intrinsics[1, 1] * point_camera[1] / point_camera[2] + intrinsics[1, 2]
            if 0.0 <= u < depth_shape[1] and 0.0 <= v < depth_shape[0]:
                visible = True
                break
        if not visible:
            probes.append(index)
    return probes


def save_response(args: argparse.Namespace, response) -> Path:
    if not response.success:
        raise RuntimeError("nvblox returned success=false")
    dimensions = response.esdf_and_gradients.layout.dim
    if len(dimensions) != 3 or [dimension.label for dimension in dimensions] != ["x", "y", "z"]:
        raise ValueError("unexpected nvblox ESDF layout")
    shape = tuple(int(dimension.size) for dimension in dimensions)
    esdf = np.asarray(response.esdf_and_gradients.data, dtype=np.float32).reshape(shape)
    expected_shape = tuple(round(value / args.voxel_size) for value in args.extent)
    if shape != expected_shape:
        raise ValueError(f"nvblox grid shape {shape} does not match expected {expected_shape}")
    unknown_sentinel = -1000.0
    origin_m = np.asarray(
        [response.origin_m.x, response.origin_m.y, response.origin_m.z],
        dtype=np.float64,
    )
    (
        requested_origin_m,
        requested_minimum_center_m,
        requested_center_span_m,
    ) = inclusive_voxel_center_aabb(
        args.grid_center, args.extent, args.voxel_size
    )
    depth_paths = validate_prepared_view_order(args.prepared_map, args.capture)
    fingerprint_inputs = [("prepared_map_report", args.prepared_map / "esdf_check.json")]
    for index, (capture, depth_path) in enumerate(zip(args.capture, depth_paths)):
        fingerprint_inputs.extend(
            [
                (f"camera_{index}_mapping_depth", depth_path),
                (f"camera_{index}_intrinsics", capture / "intrinsics.npy"),
                (
                    f"camera_{index}_robot_base_camera",
                    capture / "T_robot_base_camera.npy",
                ),
            ]
        )
    input_fingerprint = fingerprint_files(fingerprint_inputs)
    unobserved_probes = corner_probe_indices_outside_all_views(args, origin_m, shape)
    probe_distances = np.asarray(
        [esdf[index] for index in unobserved_probes], dtype=np.float32
    )
    checks = {
        "service_success": bool(response.success),
        "voxel_size_matches_request": bool(
            np.isclose(response.voxel_size_m, args.voxel_size, atol=1e-7)
        ),
        "shape_matches_requested_aabb": shape == expected_shape,
        "origin_matches_requested_aabb": bool(
            np.allclose(origin_m, requested_origin_m, atol=1e-7, rtol=0.0)
        ),
        "esdf_is_finite": bool(np.isfinite(esdf).all()),
        "no_unobserved_sentinel": bool(
            ~np.isclose(esdf, unknown_sentinel, atol=1e-2).any()
        ),
        "has_proven_unobserved_corner_probes": bool(unobserved_probes),
        "proven_unobserved_probes_are_nonpositive": bool(
            probe_distances.size and np.all(probe_distances <= 0.0)
        ),
        "grid_has_positive_known_free": bool((esdf > 0.0).any()),
        "grid_has_nonpositive_blocked": bool((esdf <= 0.0).any()),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "esdf_features.npy", esdf)
    np.save(output / "known_free_mask.npy", esdf > 0.0)
    report = {
        "status": "success" if all(checks.values()) else "failed_checks",
        "backend": "B_isaac_ros_nvblox_kOccupied",
        "reference": {
            "isaac_ros_nvblox_tag": "v4.5-0",
            "isaac_ros_nvblox_commit": "a0dbb2a06475dc8fa0dbdf5b919ec53973843d17",
            "service": "nvblox_msgs/srv/EsdfAndGradients",
            "policy": "static_mapper.unobserved_esdf_policy=occupied",
        },
        "grid": {
            "shape_xyz": list(shape),
            "voxel_size_m": float(response.voxel_size_m),
            "extent_m": list(args.extent),
            "center_robot_base_m": list(args.grid_center),
            "min_corner_robot_base_m": requested_origin_m.tolist(),
            "origin_robot_base_m": [
                float(response.origin_m.x),
                float(response.origin_m.y),
                float(response.origin_m.z),
            ],
            "service_aabb_minimum_voxel_center_m": requested_minimum_center_m.tolist(),
            "service_aabb_center_span_m": requested_center_span_m.tolist(),
            "index_order": "x_slowest_z_fastest",
            "sdf_sign": "positive_free_negative_blocked",
        },
        "counts": {
            "total_voxels": int(esdf.size),
            "known_free_voxels": int((esdf > 0.0).sum()),
            "blocked_voxels": int((esdf <= 0.0).sum()),
            "zero_distance_sites": int(np.isclose(esdf, 0.0, atol=1e-7).sum()),
            "proven_unobserved_corner_probes": len(unobserved_probes),
        },
        "input_fingerprint_sha256": input_fingerprint,
        "proven_unobserved_corner_probes": [
            {
                "index_xyz": list(index),
                "distance_m": float(esdf[index]),
            }
            for index in unobserved_probes
        ],
        "automatic_checks": checks,
        "unknown_environment_contract": {
            "unobserved_space_is_blocked": True,
            "official_kOccupied_policy": True,
            "requested_aabb_was_fully_materialized": True,
            "target_is_currently_blocked": True,
        },
        "safe_to_plan": False,
        "next_gate": "Compare with backend A, then choose a target-clear proxy before planning",
    }
    report_path = output / "esdf_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"official nvblox checks failed; inspect {report_path}")
    return report_path


def main() -> int:
    args = parse_args()
    if len(args.capture) != 3:
        raise ValueError("the pinned backend-B launch currently requires exactly 3 captures")
    rclpy.init()
    node = OfflineEsdfNode(args)
    try:
        node.publish_static_transforms()
        for _ in range(args.publish_count):
            node.publish_views_once()
            rclpy.spin_once(node, timeout_sec=args.publish_period)
            time.sleep(args.publish_period)
        response = node.request_esdf()
        report_path = save_response(args, response)
        print(f"saved: {report_path}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
