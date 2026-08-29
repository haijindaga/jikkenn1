# RoboLab / Isaac Lab grasp-retention baseline

This baseline tests one question without changing perception, grasp selection, or
cuRobo planning: does the same planned hammer grasp remain physically held when
the Panda finger actuator uses a published Isaac Lab configuration?

## Fixed experimental conditions

- Same RGB-D capture: `outputs/hammer_singleview_v2/camera_0`
- Same cuRobo plan: `outputs/curobo_grasp_lift_hammer_singleview_v1`
- Same authored scene: `scenes/hammer_01.usda`
- Same physical RoboLab HANDAL hammer: 0.5 kg, static/dynamic friction 2.0
- Same close target, trajectory, physics rate, and hold duration
- Robot-world and self-collision behavior is not disabled

The only intended difference is the Panda finger drive condition.

## Why these two conditions

`authored-usd` leaves the loaded Panda USD unchanged. It is the control condition
and remains the default.

`isaaclab-franka` applies the Panda hand values published by NVlabs RoboLab's
`FrankaCfg`, which states that it is adapted from Isaac Lab's
`FRANKA_PANDA_CFG`: max effort/linear-drive max force 200, stiffness 2000, and
damping 100. The current replay implementation maps those values to the existing
OpenUSD linear `DriveAPI` fields and records before/after values.

These are simulator actuator values. In particular, 200 and the legacy optional
`--finger-drive-max-force-n 70` value are not claims about calibrated total
Franka Hand grasp force. The 70 value is not a default or a named baseline.

Pinned source:
`NVlabs/RoboLab@9db0aaf09d9fe5d4f37b168320788258c7012463/robolab/robots/franka.py`.

## Verify the unchanged RoboLab hammer

If the RoboLab assets live in this repository:

```bash
python scripts/verify_robolab_baseline_assets.py \
  --asset-root . \
  --asset hammer
```

If they live in a separate RoboLab checkout, pass that checkout as
`--asset-root`. The verifier checks the pinned USD hash and the dataset license.
The manifest also pins the YCB scissors baseline for the next object test:
`config/robolab_baseline_assets.json`.

## Run the control

```bash
python scripts/isaac_replay_grasp_lift.py \
  --capture outputs/hammer_singleview_v2/camera_0 \
  --plan outputs/curobo_grasp_lift_hammer_singleview_v1 \
  --scene-usd scenes/hammer_01.usda \
  --output outputs/isaac_grasp_lift_hammer_authored_usd_v1 \
  --finger-drive-preset authored-usd \
  --simulation-only
```

## Run the Isaac Lab condition

```bash
python scripts/isaac_replay_grasp_lift.py \
  --capture outputs/hammer_singleview_v2/camera_0 \
  --plan outputs/curobo_grasp_lift_hammer_singleview_v1 \
  --scene-usd scenes/hammer_01.usda \
  --output outputs/isaac_grasp_lift_hammer_isaaclab_franka_v1 \
  --finger-drive-preset isaaclab-franka \
  --simulation-only
```

Omit `--headless` to watch the replay. A process exit code of 2 still means the
simulation completed and saved diagnostics, but the object did not remain lifted
by the existing physical-pick criterion.

## Compare the reports

```bash
python scripts/compare_grasp_lift_replays.py \
  --baseline outputs/isaac_grasp_lift_hammer_authored_usd_v1 \
  --candidate outputs/isaac_grasp_lift_hammer_isaaclab_franka_v1 \
  --output outputs/isaac_grasp_lift_hammer_drive_comparison_v1.json
```

The comparison rejects a silent capture, plan, scene, or target-mass change. It
reports peak lift, held lift, lift lost after the peak, and finger gaps. Planning
success is not reclassified as physical pickup success.

## Asset provenance and next object

The hammer already used by this project matches RoboLab's physics-ready HANDAL
USD rather than a locally invented mesh. HANDAL is CC BY-NC-SA 4.0. RoboLab's
YCB scissors is MIT licensed, physics-ready, 0.2 kg, and also pinned in the
manifest. Keep the original dataset `LICENSE` and texture directory beside any
copied USD; do not flatten or silently rewrite the asset.

After the same-hammer comparison is reviewed, the next controlled experiment is
to place the pinned scissors USD into a new authored scene and run the existing
RGB-D -> SAM3 -> masked point cloud -> GraspGenX -> cuRobo -> Isaac replay path.

The physics-ready target can be referenced, centred, placed 2 mm above the table,
validated, saved, and closed without GUI editing:

```bash
python scripts/isaac_edit_tabletop_scene.py \
  --output scenes/scissors_01.usda \
  --target-usd /home/suzutaro/RoboLab/assets/objects/ycb/scissors.usd \
  --exit-after-save
```

The saved scene contains a relative OpenUSD reference. The RoboLab source asset
is not flattened or modified. Capture-time physics settling closes the initial
2 mm clearance before RGB-D validation.
