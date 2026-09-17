"""Installed USD inventory and synchronized PhysX poses, not a new robot model."""

import numpy as np

from .geometry import matrix_from_pose


def collect_model_evidence(stage, robot_path, joint_names, joint_positions,
                           rigid_prim_factory):
    from pxr import Usd, UsdPhysics

    names = list(joint_names)
    q = np.asarray(joint_positions, dtype=float).reshape(-1)
    if len(names) != len(q) or len(set(names)) != len(names) or not np.all(np.isfinite(q)):
        raise ValueError("Invalid measured joint inventory")
    bodies, joints, collisions = [], [], []
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise ValueError("Robot root does not exist")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # Only wrap existing rigid bodies; never author mass, transforms or drives.
            body = rigid_prim_factory(prim_path=path, name=f"model_evidence_{len(bodies)}",
                                      reset_xform_properties=False)
            body.initialize()
            position, quaternion = body.get_world_pose()
            transform = matrix_from_pose(np.asarray(position), np.asarray(quaternion))
            if not np.all(np.isfinite(transform)):
                raise ValueError(f"Non-finite runtime body pose: {path}")
            bodies.append({"path": path, "name": prim.GetName(),
                           "T_world_body": transform.tolist()})
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append({
                "path": path, "type": prim.GetTypeName(),
                "body0": list(map(str, joint.GetBody0Rel().GetTargets())),
                "body1": list(map(str, joint.GetBody1Rel().GetTargets())),
                "attributes": {str(a.GetName()): str(a.Get()) for a in prim.GetAttributes()
                               if str(a.GetName()).startswith(("physics:", "drive:", "physxMimicJoint:"))},
                "applied_schemas": list(map(str, prim.GetAppliedSchemas())),
            })
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append({"path": path, "type": prim.GetTypeName(),
                               "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()})
    if not bodies:
        raise ValueError("No runtime rigid bodies found")
    return {
        "status": "inventory_collected", "robot_prim": robot_path,
        "joint_names": names, "joint_positions": q.tolist(),
        "rigid_bodies": bodies, "joints": joints, "collisions": collisions,
        "pose_source": "SingleRigidPrim.get_world_pose after physics step, not USD xforms",
        "safety": {"profile_ready": False, "grasp_to_tool_verified": False,
                   "collision_geometry_equivalence_verified": False, "trajectory_saved": False},
    }


def arm_poses_in_base(evidence):
    """Select actual rigid bodies, avoiding same-named visual geometry."""
    selected = {}
    for index in range(8):
        name = f"panda_link{index}"
        matches = [b for b in evidence["rigid_bodies"] if b["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one rigid {name}, found {len(matches)}")
        value = np.asarray(matches[0]["T_world_body"], dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid body transform: {name}")
        selected[name] = value
    inverse_base = np.linalg.inv(selected["panda_link0"])
    return {name: inverse_base @ value for name, value in selected.items()}


def compare_arm_poses(observed, predicted, translation_tolerance=0.005,
                      rotation_tolerance=np.deg2rad(2)):
    if set(observed) != set(predicted):
        raise ValueError("FK and observed arm frame names disagree")
    results = {}
    for name, value in observed.items():
        prediction = np.asarray(predicted[name], dtype=float)
        if prediction.shape != (4, 4) or not np.all(np.isfinite(prediction)):
            raise ValueError(f"Invalid FK transform: {name}")
        translation = float(np.linalg.norm(value[:3, 3] - prediction[:3, 3]))
        cosine = (np.trace(value[:3, :3].T @ prediction[:3, :3]) - 1) / 2
        rotation = float(np.arccos(np.clip(cosine, -1, 1)))
        results[name] = {"translation_error_m": translation, "rotation_error_rad": rotation,
                         "passed": bool(translation <= translation_tolerance and rotation <= rotation_tolerance)}
    return results


def collect_gripper_collision_geometry(stage, evidence):
    """Export authored collision topology with synchronized rigid-body transforms.

    Mesh-local transforms come from USD, body poses from PhysX. We do not replace
    runtime body poses with potentially stale USD articulation xforms. Authored
    meshes are not a claim of equivalence to PhysX's cooked convex shapes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from .robotiq_model import body_transform

    base_entries = [b for b in evidence["rigid_bodies"] if b["name"] == "base_link"]
    if len(base_entries) != 1:
        raise ValueError("Expected one installed Robotiq base rigid body")
    root_path = str(stage.GetPrimAtPath(base_entries[0]["path"]).GetParent().GetPath())
    inverse_base = np.linalg.inv(body_transform(evidence, "base_link"))
    runtime = {b["path"]: np.asarray(b["T_world_body"], dtype=float)
               for b in evidence["rigid_bodies"]}
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Unsupported installed collision geometry: {prim.GetPath()}")
        body = prim
        while body.IsValid() and str(body.GetPath()) not in runtime:
            body = body.GetParent()
        if not body.IsValid():
            raise ValueError(f"Collision mesh has no measured parent body: {prim.GetPath()}")
        relative, reset_stack = cache.ComputeRelativeTransform(prim, body)
        if reset_stack:
            raise ValueError(f"Mesh resets its transform outside its parent body: {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                or not np.all(np.isfinite(points)) or counts.ndim != 1
                or len(counts) == 0 or np.any(counts < 3) or indices.ndim != 1
                or counts.sum() != len(indices) or np.any(indices < 0) or np.any(indices >= len(points))):
            raise ValueError(f"Invalid collision mesh topology: {prim.GetPath()}")
        transform = inverse_base @ runtime[str(body.GetPath())] @ np.asarray(relative, dtype=float).T
        points_base = points @ transform[:3, :3].T + transform[:3, 3]
        approximation = prim.GetAttribute("physics:approximation")
        meshes.append({"path": str(prim.GetPath()), "body": str(body.GetPath()),
                       "vertices_gripper_base_m": points_base.tolist(),
                       "face_vertex_counts": counts.tolist(), "face_vertex_indices": indices.tolist(),
                       "T_gripper_base_mesh": transform.tolist(),
                       "physics_approximation": str(approximation.Get()) if approximation.IsValid() else None})
    if not meshes:
        raise ValueError("No enabled installed gripper collision meshes found")
    return {"status": "authored_collision_geometry_exported", "frame": "installed Robotiq base_link",
            "joint_names": evidence["joint_names"], "joint_positions": evidence["joint_positions"],
            "meshes": meshes, "source": "authored USD topology + runtime rigid-body poses",
            "safety": {"source_assets_modified": False, "profile_ready": False,
                       "physx_cooked_geometry_exported": False,
                       "collision_geometry_equivalence_verified": False}}
