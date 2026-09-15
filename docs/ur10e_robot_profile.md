# UR10e + Robotiq 2F-140 robot profile

## Scope

The first non-Franka profile is `ur10e_robotiq_2f_140`. It keeps the existing
RGB-D, VLM, SAM3, point-cloud, candidate-ranking, cuRobo, and result-report
stages. Robot-dependent values are selected by `--robot-profile`; the Franka
pipeline is not copied or replaced.

This profile reuses:

- GraspGenX's official `UR10eRobotiq2F140Profile`, gripper model, URDF merger,
  and collision-sphere fitter.
- Isaac Sim 5.1's assembled `ur_gripper.usd` asset and its
  `robotiq_2f_140` variant.
- cuRobo's standard IK, trajectory optimization, robot segmentation, and
  collision checking.

The local adapter adds only the attached-object collision proxy contract used
by this project's handover planner. It does not edit the official meshes,
mount, joint limits, or Isaac drive gains.

UR10 (non-e) is intentionally not aliased to UR10e. It needs a separate
profile and matching official robot model because its kinematics and assets
must not be silently mixed with UR10e.

## One-time cuRobo profile preparation

Run from the GraspGenX environment. The command invokes NVIDIA's official
builder first, then writes a separate `.jikkenn1.yml` adapter and a provenance
JSON file. It leaves the official generated YAML unchanged.

```bash
cd /home/suzutaro/GraspGenX

uv run python \
  /home/suzutaro/projects/jikkenn1/scripts/prepare_ur10e_robot_profile.py \
  --graspgenx-root /home/suzutaro/GraspGenX
```

Expected files:

```text
/home/suzutaro/GraspGenX/end2end/curobo_assets/ur10e_robotiq_2f_140.urdf
/home/suzutaro/GraspGenX/end2end/curobo_assets/ur10e_robotiq_2f_140.yml
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

The capture stage saves the observed tool transform as
`capture/camera_0/T_robot_base_tool.npy`. Before IK, the pre-grasp stage
computes the same tool transform using the generated cuRobo model.

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

- [GraspGenX robot profiles](https://github.com/NVlabs/GraspGenX/blob/main/end2end/robot_profiles.py)
- [GraspGenX UR10e gripper builder](https://github.com/NVlabs/GraspGenX/blob/main/end2end/build_ur10e_gripper.py)
- [Isaac Sim 5.1 manipulator assembly tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html)
