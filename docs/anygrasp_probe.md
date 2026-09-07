# AnyGrasp candidate-generation probe

This probe uses the official `graspnet/anygrasp_sdk` without replacing the
working GraspGenX pipeline. It asks one bounded question first: does official
AnyGrasp generate visually useful parallel-jaw candidates from this project's
saved calibrated RGB-D point map and SAM3 target mask?

The current official SDK accepts an unorganized camera-frame point cloud and an
element-aligned boolean `region_steering` mask. The probe therefore supplies all
finite observed scene points for AnyGrasp collision detection and uses the SAM3
mask only to steer candidate generation toward the target. It does not use the
target USD, CAD mesh, or authored pose.

## Official SDK prerequisite

Use a separate checkout and environment; do not mix its compiled Torch,
MinkowskiEngine, or PointNet++ dependencies into Isaac Lab or GraspGenX.

```bash
cd /home/suzutaro
git clone https://github.com/graspnet/anygrasp_sdk.git
cd anygrasp_sdk
git rev-parse HEAD
```

Follow the current official README for PyTorch, the AnyGrasp fork of
MinkowskiEngine, PointNet++, `graspnetAPI`, and Python requirements. Select the
`gsnet` binary whose Python ABI matches the dedicated environment. Do not copy
a guessed binary name.

The official SDK also requires a machine-bound license. From
`anygrasp_sdk/grasp_detection`, after selecting the matching `gsnet` binary:

```bash
python -c "from gsnet import get_feature_id; print(get_feature_id())"
```

Submit that feature ID through the form linked by the official license
instructions. Keep the returned license directory inside the official SDK
checkout and do not commit it to this repository. Validate it with the official
`check_license` call before running this probe. Place the official detection
checkpoint where the SDK instructions require it.

## First run on an existing hammer capture

Run from the official detection directory with its dedicated environment:

```bash
cd /home/suzutaro/anygrasp_sdk/grasp_detection

python /home/suzutaro/projects/jikkenn1/scripts/anygrasp_infer_capture.py \
  --capture /home/suzutaro/projects/jikkenn1/outputs/hammer_vlm_part_e2e_v1/capture/camera_0 \
  --segmentation /home/suzutaro/projects/jikkenn1/outputs/hammer_vlm_part_e2e_v1/capture/sam3 \
  --checkpoint /home/suzutaro/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --anygrasp-root /home/suzutaro/anygrasp_sdk \
  --output /home/suzutaro/projects/jikkenn1/outputs/anygrasp_hammer_probe_v1 \
  --vis
```

If the experiment uses the optional part mask, replace `--segmentation` with
the exact reviewed `parts/grasp_part` directory. The whole-object mask is the
cleaner first controlled comparison against the existing GraspGenX run.

The probe saves scores, widths, depths, rotations, translations, homogeneous
GraspNet poses, and official gripper-tip positions. It also opens at most the
top 20 candidates with the official Open3D geometry conversion.

## Deliberate execution gate

`anygrasp_check.json` always records
`panda_hand_frame_transform_validated=false` and `safe_to_execute=false`.
AnyGrasp defines the approach direction as grasp-frame `+X`, closing as `+Y`,
and reports a grasp center rather than the Panda hand origin. The next stage is
to validate that robot-specific rigid transform using the displayed candidates;
only then should the backend feed cuRobo or Isaac replay.

Official references:

- <https://github.com/graspnet/anygrasp_sdk>
- <https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/USAGE.md>
- <https://github.com/graspnet/anygrasp_sdk/blob/main/license_registration/README.md>
