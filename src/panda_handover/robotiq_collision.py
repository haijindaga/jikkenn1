"""Installed open-gripper geometry and isolated cuRobo draft configuration.

The open snapshot is rigid relative to the hand only while the fingers stay
open. It is not a gripper-closing, lift or attached-object model.
"""

from copy import deepcopy

import numpy as np

from .gripper_swap import ARM_JOINTS
from .robotiq_model import body_transform, installed_mount


def snapshot_meshes_in_hand(geometry, evidence):
    installed_mount(evidence)
    if (geometry.get("status") != "authored_collision_geometry_exported"
            or geometry.get("frame") != "installed Robotiq base_link"):
        raise ValueError("Expected installed gripper-base collision geometry")
    names = list(evidence["joint_names"])
    measured = np.asarray(evidence["joint_positions"], dtype=float)
    if (geometry.get("joint_names") != names
            or measured.shape != (len(names),) or len(set(names)) != len(names)
            or not np.all(np.isfinite(measured))
            or not np.array_equal(np.asarray(geometry["joint_positions"], dtype=float), measured)):
        raise ValueError("Collision snapshot and measured model evidence disagree")
    if names.count("finger_joint") != 1 or abs(measured[names.index("finger_joint")]) > 1e-3:
        raise ValueError("Only the measured open gripper snapshot is supported")
    if not set(ARM_JOINTS).issubset(names) or any(abs(q) > 1e-3 for name, q in zip(names, measured) if name not in ARM_JOINTS):
        raise ValueError("Expected seven Panda arm joints and reopened gripper joints")
    if any(name.startswith("panda_finger_joint") for name in names):
        raise ValueError("Original Panda finger DOFs remain")
    transform = np.linalg.inv(body_transform(evidence, "panda_hand")) @ body_transform(evidence, "base_link")
    bodies = {b["path"] for b in evidence["rigid_bodies"]}
    base_path = next(b["path"] for b in evidence["rigid_bodies"] if b["name"] == "base_link")
    gripper_root = base_path.rsplit("/", 1)[0] + "/"
    pieces, paths = [], set()
    for mesh in geometry["meshes"]:
        path, body = mesh["path"], mesh["body"]
        if path in paths or not path.startswith(gripper_root) or body not in bodies or not body.startswith(gripper_root):
            raise ValueError(f"Invalid or duplicate installed gripper mesh: {path}")
        paths.add(path)
        if mesh.get("physics_approximation") != "convexHull":
            raise ValueError(f"Unsupported PhysX approximation: {path}")
        vertices = np.asarray(mesh["vertices_gripper_base_m"], dtype=float)
        counts = np.asarray(mesh["face_vertex_counts"])
        indices = np.asarray(mesh["face_vertex_indices"])
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or not np.all(np.isfinite(vertices)) or counts.ndim != 1 or len(counts) == 0
                or not np.all(counts == 3) or indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer) or len(indices) != 3 * len(counts)
                or np.any(indices < 0) or np.any(indices >= len(vertices))):
            raise ValueError(f"Invalid triangle mesh: {path}")
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        pieces.append({"path": path, "vertices": vertices, "faces": indices.reshape(-1, 3)})
    expected = {c["path"] for c in evidence["collisions"]
                if c["path"].startswith(gripper_root) and c.get("enabled") is not False}
    if not pieces or paths != expected:
        raise ValueError("Snapshot does not cover every enabled installed gripper collider")
    return pieces


def validated_spheres(spheres):
    result = []
    for sphere in spheres:
        center = np.asarray(sphere["center"], dtype=float)
        radius = float(sphere["radius"])
        if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Official sphere fitter returned an invalid sphere")
        result.append({"center": center.tolist(), "radius": radius})
    if not result:
        raise ValueError("Official sphere fitter returned no positive spheres")
    return result


def vertex_coverage(vertices, spheres):
    """Inspection metric only, not proof of volume/triangle coverage."""
    spheres = validated_spheres(spheres)
    centers = np.asarray([s["center"] for s in spheres])
    radii = np.asarray([s["radius"] for s in spheres])
    values = []
    for start in range(0, len(vertices), 256):
        distances = np.linalg.norm(vertices[start:start + 256, None] - centers[None], axis=2) - radii
        values.extend(np.min(distances, axis=1).tolist())
    return {"vertex_count": len(values), "outside_vertex_count": int(np.count_nonzero(np.asarray(values) > 1e-6)),
            "maximum_vertex_outside_distance_m": max(0.0, float(max(values))),
            "numerical_tolerance_m": 1e-6, "volume_coverage_proven": False}


def open_snapshot_config(stock, urdf_path, spheres):
    """Reuse official arm spheres/cspace/pairs; replace only the old hand model."""
    cfg = deepcopy(stock)
    kin = cfg["robot_cfg"]["kinematics"]
    arm_links = [f"panda_link{i}" for i in range(8)]
    links = arm_links + ["panda_hand"]
    kin["collision_spheres"] = {name: deepcopy(kin["collision_spheres"][name]) for name in arm_links}
    kin["collision_spheres"]["panda_hand"] = validated_spheres(spheres)
    kin["collision_link_names"] = links
    kin["mesh_link_names"] = arm_links  # The composed hand has no old Panda mesh.
    kin["grasp_contact_link_names"] = []  # Not a contact/closure model.
    kin["extra_links"] = {}
    kin["extra_collision_spheres"] = {}
    kin["lock_joints"] = None
    kin["tool_frames"] = ["panda_hand"]
    kin["urdf_path"] = str(urdf_path)
    kin["asset_root_path"] = str(urdf_path.parent)
    kin["self_collision_ignore"] = {
        name: [other for other in others if other in links]
        for name, others in kin["self_collision_ignore"].items() if name in links
    }
    # Preserve wrist/hand adjacency exclusions, but do not blanket-ignore arm/hand.
    kin["self_collision_buffer"] = {name: kin["self_collision_buffer"].get(name, 0.0) for name in arm_links}
    kin["self_collision_buffer"]["panda_hand"] = 0.0  # New fitted geometry, no old hand's 20 mm padding.
    cspace = kin["cspace"]
    old_names = cspace["joint_names"]
    keep = [old_names.index(name) for name in ARM_JOINTS]
    for field in ("cspace_distance_weight", "null_space_weight", "default_joint_position"):
        cspace[field] = [cspace[field][i] for i in keep]
    cspace["joint_names"] = list(ARM_JOINTS)
    for field in ("max_acceleration", "max_jerk"):
        if isinstance(cspace[field], list):
            cspace[field] = [cspace[field][i] for i in keep]
    cfg["jikkenn1_scope"] = "draft, open gripper snapshot only; not executable or a grasp/lift profile"
    return cfg
