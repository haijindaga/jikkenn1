# UR10e + Robotiq 2F-140 robot profile

## Scope

The first non-Franka profile is `ur10e_robotiq_2f_140`. It keeps the existing
RGB-D, VLM, SAM3, point-cloud, candidate-ranking, cuRobo, and result-report
stages. Robot-dependent values are selected by `--robot-profile`; the Franka
pipeline is not copied or replaced.

This profile reuses:

- NVIDIA Isaac ROS cuMotion release-3.2's validated
  `ur10e_robotiq_2f_140.urdf` + `.xrdf` pair at pinned commit
  `dbaa7e8264f6314f8baca516511414186ad1105d`.
- GraspGenX's official `robotiq_2f_140` grasp model.
- Isaac Sim 5.1's assembled `ur_gripper.usd` asset and its
  `robotiq_2f_140` variant.
- cuRobo's standard IK, trajectory optimization, robot segmentation, and
  collision checking, plus cuRobo's own XRDF-to-native-config converter.

For RGB-D capture, the arm is parked at the initial joint pose published by
Isaac Lab's `UR10e_ROBOTIQ_GRIPPER_CFG`:

```text
[pi, -pi/2, pi/2, -pi/2, -pi/2, 0]
```

This is an observation pose, not a replacement for cuRobo's retract/seed
configuration. The captured joint state remains the trajectory start state.
Before either pre-grasp replay or a physical grasp/lift trial, Isaac restores
every articulation DOF from that named capture state through its default-state
reset API. During settling, position targets hold only the planned arm joints
and the profiled gripper master joint; passive Robotiq joints remain governed
by the asset's coupling. The replay then verifies the measured arm state
against the capture before issuing the first trajectory command.

The official XRDF supplies the arm/gripper collision spheres, self-collision
exclusions, `grasp_frame`, and `attached_object` frame. The local adapter does
not refit or edit those values. It only adds the contact-link list consumed by
the existing grasp planner and reserves four disabled spheres that cuRobo's
official attachment manager replaces with the held-object geometry at runtime.

The Isaac USD exposes `robotiq_arg2f_base_link`, whereas the official planning
model uses `grasp_frame`. Capture observes the former rigid body and composes
the official fixed +0.20 m local-Z transform before comparing Isaac and cuRobo
forward kinematics. Both the directly observed and composed transforms are
saved, so this frame bridge is explicit and auditable.

UR10 (non-e) is intentionally not aliased to UR10e. It needs a separate
profile and matching official robot model because its kinematics and assets
must not be silently mixed with UR10e.

## One-time cuRobo profile preparation

Run from the GraspGenX environment. The command verifies the vendored official
URDF/XRDF hashes, invokes the pinned cuRobo XRDF converter, then writes a
separate `.jikkenn1.yml` runtime adapter and provenance JSON file.

```bash
cd /home/suzutaro/GraspGenX

uv run --no-sync python \
  /home/suzutaro/projects/jikkenn1/scripts/prepare_ur10e_robot_profile.py \
  --graspgenx-root /home/suzutaro/GraspGenX \
  --overwrite
```

Expected generated files:

```text
/home/suzutaro/GraspGenX/end2end/curobo_assets/ur10e_robotiq_2f_140.jikkenn1.yml
/home/suzutaro/GraspGenX/end2end/curobo_assets/ur10e_robotiq_2f_140.jikkenn1.json
```

Use `--overwrite` only when intentionally rebuilding those two local adapter
files after an upstream or configuration change.

## Create a UR10e tabletop scene

This example reuses the existing RoboLab hammer. Choose a new scene filename.

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

python scripts/isaac_edit_tabletop_scene.py \
  --robot-profile ur10e_robotiq_2f_140 \
  --target-usd /home/suzutaro/RoboLab/assets/objects/handal/hammer.usd \
  --target-center-xy 0.50 0.00 \
  --output scenes/hammer_ur10e_01.usda \
  --exit-after-save
```

The scene creator fails if the official assembled asset, expected gripper
variant, or profiled joints are absent.

## Run the shared end-to-end pipeline

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

python scripts/run_sim_grasp_pipeline.py \
  --robot-profile ur10e_robotiq_2f_140 \
  --scene-usd scenes/hammer_ur10e_01.usda \
  --target-object hammer \
  --ollama-model gemma3:12b \
  --output outputs/hammer_ur10e_e2e_v1 \
  --allow-reviewed-support-contact-preflight
```

Use a new output directory for each independent experiment. `--resume` is only
for resuming the same profile and same run.

## Mandatory alignment gate

The capture stage saves the composed planning-frame transform as
`capture/camera_0/T_robot_base_tool.npy` and the directly observed Isaac rigid
body as `T_robot_base_observed_tool.npy`. Before IK, the pre-grasp stage
computes `grasp_frame` using the official cuRobo model.

Planning stops with `robot_model_alignment_failed` when either error exceeds:

- translation: 0.005 m
- rotation: 2 degrees

This check prevents an apparently valid UR10e trajectory from being replayed
on an Isaac asset with a different Robotiq mount. Do not increase the
tolerances to force a run through; inspect the official mount representations
first.

## Current verification boundary

Repository-level compilation and unit tests verify profile routing, legacy
Franka compatibility, generated-config adaptation, and the alignment gate.
An Isaac/CUDA machine must still run the commands above to establish the first
UR10e runtime result. Until that succeeds, this is implemented support with a
pending simulator integration gate, not a claimed successful UR10e grasp.

## Upstream references

- [Isaac ROS cuMotion robot descriptions](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/tree/release-3.2/isaac_ros_cumotion_robot_description)
- [GraspGenX robot profiles](https://github.com/NVlabs/GraspGenX/blob/main/end2end/robot_profiles.py)
- [Isaac Sim 5.1 manipulator assembly tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html)
