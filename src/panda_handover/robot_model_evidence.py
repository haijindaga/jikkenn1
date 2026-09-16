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
