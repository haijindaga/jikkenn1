# Official Panda / Robotiq variant trial

This is a separate **gripper replacement and open/close smoke test**. It is not
yet a new end-to-end robot profile or a hammer grasp-success experiment.

Isaac Sim 5.1's official Panda USD ships a `Gripper=Robotiq_2F_85` variant:

- [Official robot assets](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html)
- [Official variant API and exact names](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.utils/docs/index.html)
- [Robotiq drive/mimic-joint example](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html)

Use this already-mounted **85**, rather than assembling the previously suggested
140 onto Panda ourselves. Both grippers are supported by GraspGenX, but the
official Panda mounting is available for 85. No new mesh or mount is authored.

## Run

After transferring the new files to the Linux repository:

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

head_plan=outputs/hammer_head_handover_fixedjoint_e2e_v1/curobo_grasp_lift_trials/candidate_039
head_capture=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["capture"])' "$head_plan/grasp_lift_plan_check.json")

python scripts/isaac_try_panda_robotiq.py \
  --scene-usd scenes/hammer_01.usda \
  --capture "$head_capture" \
  --output "outputs/panda_robotiq85_$(date +%Y%m%d_%H%M%S)" \
  --simulation-only
```

The script opens, closes, and reopens the gripper (three seconds per phase), then
keeps the GUI open with **Open / Close** buttons. `--headless` runs the finite
smoke test and exits. It reads declared joint limits and drive gains; a
zero-stiffness velocity drive uses standard `ArticulationAction` velocity targets
at an explicitly recorded smoke-test speed (joint span / phase duration, capped
by the authored maximum velocity), stopping near the limit. This small limit-stop
adapter is not a claimed official Panda+Robotiq grasp controller. It does not import
the Panda `Franka` hand controller, alter force/stiffness/damping/friction,
attach an object, or run any old grasp trajectory. It records actual master
joint motion and arm tracking, not an assumed visual/grasp success.

## Non-destructive composition

The new output `scene.usda` sublayers the existing scene and selects the official
Robotiq variant **at the same `/World/Panda` path**. It does not create a second
robot, deactivate the original, author a mounting joint, or replace references.
Root-stage metadata is copied explicitly: USD sublayering alone does not inherit
the original up axis or units. A non-metre or non-Z-up source is rejected rather
than silently converted. Metadata, robot-root attributes, and non-robot prim
attributes/relationships are compared before simulation. Relative object
references remain anchored to the source layers.

The seven arm joint angles are read from the existing capture by **joint name**,
not guessed or inherited from a different home pose. Existing scene-level
overrides are retained, not silently removed. If original Panda finger DOFs remain
or additional gripper DOFs have independent drives, the motion test stops and
saves the inventory for review; it does not guess how to clean up the scene.

The table, object references, lights and camera are inherited unchanged. The
source scene hash is checked and source assets are never saved. The output
depends on the original source scene and official asset paths; it is not a
portable packaged asset. Output-directory reuse is refused. Scene physics settings
are retained by default. Optional `--replay-physics cpu` enables CPU PhysX / Fabric
OFF only in this test process. Normal Panda defaults remain unchanged.

The JSON `gripper_swap_check.json` records the asset, variant, joint inventory,
authored gains, USD joint/mimic schemas and attributes, physics settings, arm
angles, phase results and failure traceback. Joint inventory is written before
interpreting control conventions. The Isaac Sim 5.1 tensor DOF convention is
Rotation=0, Translation=1; the old smoke test incorrectly expected Rotation=1.
`success` means an open/close smoke test passed, **not that the hammer was picked**.
Unknown master-joint or drive conventions fail rather than guessing commands.

For an inventory-only pass add `--inspect-only`: the script loads the variant,
initializes/reset the simulator, saves the inventory and exits without sending
position/velocity commands. It is not a read-only simulator startup (World reset
still initializes physics), and `inspection_complete` does not mean open/close
or grasp success.

## Next gate

First confirm the replacement and opening/closing in the installed Isaac Sim.
Then prepare a matching Panda+Robotiq cuRobo model and grasp-frame transform,
regenerate GraspGenX candidates with `robotiq_2f_85`, and adapt capture/masking,
collision filtering and replay to that same profile. The original Panda
candidate 039 is not a valid interchangeable-gripper grasp test. Do not pass
this test scene into the current Panda-only end-to-end pipeline.

The local Windows environment cannot execute Isaac Sim. Pure-Python tests cover
control contracts and CLI defaults; additional real-USD composition tests run
when `pxr` is available and otherwise skip. Runtime and grasp compatibility remain
unverified until the Linux test is performed.

Metadata enumeration uses the root layer's `pseudoRoot.ListInfoKeys()` /
`GetInfo()` and copies values using `SetInfo()`, excluding composition/child
fields. `Usd.Stage.GetAllMetadata()` is not an available API. Regression tests
exercise a stage without that method, as well as real-USD variant composition,
metadata preservation and relative object references. A separate Windows
`usd-core` installation validates USD composition only, not Isaac Sim/PhysX.

## Robotiq connection preparation

Opening/closing has been visually confirmed on the user's installed model.
Candidate inference now accepts `--gripper-name robotiq_2f_85` (default remains
`franka_panda`). It uses the official `SweepVolumeParams.from_gripper_config`
and `infer_scene_pc` contract: gripper geometry is sent as sweep-volume parameters,
not an invented `gripper_name` RPC argument. Reports record the chosen gripper.
Static filtering infers the gripper from that report, requires its actual
`coll_mesh.obj`, rejects empty meshes and explicit gripper mismatches, and uses
the official filter. These checks do **not** establish equivalence with the USD.

Robotiq results deliberately do not contain `panda_hand_world.npy` or the Panda
90-degree grasp-frame offset. Existing Panda-only pregrasp and handover reranking
reject Robotiq candidates explicitly. This is not yet a full Robotiq pipeline:
robot masking, collision geometry, grasp-frame conversion and execution must all
use a matching verified profile first. Old Panda candidates/plans cannot simply
be reused with the new fingers.

The next experiment collects actual runtime rigid-body frames, measured joint
positions, joint body relationships, local-frame/mimic attributes and collision
inventory. It wraps only existing rigid bodies using `SingleRigidPrim` with
`reset_xform_properties=False`; it does not author new mounts or meshes. USD
xforms are not used as a substitute for PhysX runtime poses. The optional FK
check runs separately in the GraspGenX environment and reuses cuRobo's public
`Kinematics` API, with collision spheres disabled, rather than implementing FK.
It compares all eight arm frames at one measured posture. Passing that check
does not validate the Robotiq mount or grasp frame, nor authorize planning.

After committing/pushing these files from the same Windows repository and
pulling on Linux, run:

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab
head_plan=outputs/hammer_head_handover_fixedjoint_e2e_v1/curobo_grasp_lift_trials/candidate_039
head_capture=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["capture"])' "$head_plan/grasp_lift_plan_check.json")
test_run="outputs/panda_robotiq85_model_$(date +%Y%m%d_%H%M%S)"

python scripts/isaac_try_panda_robotiq.py \
  --scene-usd scenes/hammer_01.usda --capture "$head_capture" \
  --output "$test_run" --export-model-evidence --headless --simulation-only && {
  evidence_path="/home/suzutaro/projects/jikkenn1/$test_run/robot_model_evidence.json"
  cd /home/suzutaro/GraspGenX
  uv run --no-sync python \
    /home/suzutaro/projects/jikkenn1/scripts/check_panda_robotiq_arm_fk.py \
    --evidence "$evidence_path" \
    --output "${evidence_path%/*}/arm_fk_check.json"
}
```

Both commands use new output paths and refuse overwriting. The first command is
a finite open/close test, not a hammer grasp; remove `--headless` to see it (close
its GUI to proceed). Review both JSON reports before creating the full profile.
In particular, provide `robot_model_evidence.json` for matching the installed
mount and joint/geometry structure; `arm_alignment_passed` alone is insufficient.

### Rerun arm FK after the stock finger-lock error

Arm-only FK passes `lock_joints=None` and `extra_links={}` to the public cuRobo
configuration loader. The stock finger locks cannot be resolved in a kinematic
tree built only to arm frames; the stock attached-object extension also does not
belong to this diagnostic. These overrides are recorded in the result report.
The official YAML/URDF files, simulator joint locks, physical gripper and normal
Panda pipeline are not modified. This is not a workaround that bypasses frame
comparison: all eight arm frames and the existing error thresholds remain.

The existing evidence from the successful open/close run can be reused; do not
repeat the simulation. After transferring the fix, rerun:

```bash
cd /home/suzutaro/GraspGenX
uv run --no-sync python \
  /home/suzutaro/projects/jikkenn1/scripts/check_panda_robotiq_arm_fk.py \
  --evidence /home/suzutaro/projects/jikkenn1/outputs/panda_robotiq85_model_20260917_085346/robot_model_evidence.json \
  --output "/home/suzutaro/projects/jikkenn1/outputs/panda_robotiq85_model_20260917_085346/arm_fk_check_fixed_$(date +%Y%m%d_%H%M%S).json"
```

## Isolated source-model preparation and tool/mount FK

The installed arm FK has passed. The supplied runtime evidence also shows
`panda_hand` and the installed Robotiq `base_link` coincident, joined by the
official `AssemblerFixedJoint`. The preparation script checks both conditions,
then composes the **existing** cuRobo Panda arm URDF and GraspGenX 2F-85 URDF into
a **new FK-only file**. The arm and its 107 mm / minus-45-degree hand frame remain
unchanged. The old hand geometry, fingers and auxiliary old-hand TCP links are
removed only from the generated diagnostic model. The source Robotiq joint
origins, axes, scales, limits and mimic relationships are copied unchanged.
Mesh references are resolved to existing absolute paths; empty files, missing
files and Git LFS pointers fail. No mesh, collision sphere or physical mount is
created in Isaac Sim, and no source URDF/USD/config is overwritten.

The source gripper's fixed `world_joint` defines the canonical grasp frame:
`T_grasp_gripper_base = Rz(+1.5708)` with zero translation. Its `fingertip` value
of 136 mm is **not** an extra translation for the native base/hand frame. The
result is saved as `T_grasp_panda_hand_proposed.npy`; it is deliberately not
injected into grasp candidate or planner outputs before collision-mesh review.
Unexpected root transforms or additional `base_rotation` are rejected.

The extended FK checker compares eight arm links **plus** `panda_hand` and
`robotiq_arg2f_base_link` at the measured posture. It reuses the public cuRobo FK
loader with the generated URDF; no new FK solver is implemented. Its result is
`tool_mount_alignment_passed`, **not** `profile_ready`. Collision spheres are
disabled for this diagnostic only. There is no new collision-ready YAML yet,
and the normal Panda-only planner/rerank guards remain in place.

The USD export is needed because equivalent-looking gripper parts have different
body-frame conventions: the installed USD uses coincident open-state body origins
with mesh offsets in descendants, whereas the URDF uses joint/part-local origins.
Comparing same-named body origins is therefore not a geometry-equivalence test.
Optional `--export-collision-geometry` saves enabled authored gripper meshes in
the measured gripper-base frame, retaining face topology and collision
approximation attributes. It combines mesh-to-body USD transforms with runtime
body poses, not stale global USD articulation xforms. This is **not** an export
of PhysX's cooked convex hulls, and does not automatically declare equivalence.

After transferring these changes, the following block performs the finite
open/close + geometry export, arm FK, source composition and native mount FK.
It stops on any command failure and uses a fresh timestamped directory. It does
**not** attempt hammer grasping or reuse the old Panda trajectory.

```bash
(
  set -e
  cd /home/suzutaro/projects/jikkenn1
  conda activate env_isaaclab
  head_plan=outputs/hammer_head_handover_fixedjoint_e2e_v1/curobo_grasp_lift_trials/candidate_039
  head_capture=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["capture"])' "$head_plan/grasp_lift_plan_check.json")
  run="/home/suzutaro/projects/jikkenn1/outputs/panda_robotiq85_connection_$(date +%Y%m%d_%H%M%S)"

  python scripts/isaac_try_panda_robotiq.py \
    --scene-usd scenes/hammer_01.usda --capture "$head_capture" \
    --output "$run" --export-model-evidence --export-collision-geometry \
    --headless --simulation-only

  cd /home/suzutaro/GraspGenX
  uv run --no-sync python /home/suzutaro/projects/jikkenn1/scripts/check_panda_robotiq_arm_fk.py \
    --evidence "$run/robot_model_evidence.json" --output "$run/arm_fk_check.json"

  uv run --no-sync python /home/suzutaro/projects/jikkenn1/scripts/prepare_panda_robotiq_model.py \
    --graspgenx-root /home/suzutaro/GraspGenX \
    --evidence "$run/robot_model_evidence.json" --arm-fk-check "$run/arm_fk_check.json" \
    --output "$run/fk_model"

  uv run --no-sync python /home/suzutaro/projects/jikkenn1/scripts/check_panda_robotiq_arm_fk.py \
    --evidence "$run/robot_model_evidence.json" \
    --prepared-model "$run/fk_model/model_preparation_check.json" \
    --output "$run/tool_mount_fk_check.json"

  echo "Results: $run"
)
```

Remove `--headless` if watching opening/closing is useful; close the GUI to
continue the block. Inspect `tool_mount_fk_check.json`,
`fk_model/model_preparation_check.json` and `gripper_collision_geometry.json`.
All generated-model source paths/hashes are recorded; the FK checker rejects a
changed URDF or different evidence. Source meshes must remain at those paths.
Source URDFs/meshes retain their upstream licenses; this diagnostic composition
does not relicense or package those assets for redistribution.

Next gate: review actual USD vs source URDF/canonical collision geometry, then
build matching collision spheres/masking and an isolated Robotiq execution
adapter. Original Panda capture/planning/replay defaults, drive gains, friction,
physics settings and attachment policy are unchanged. Local tests cover contracts
and source composition, not Isaac/PhysX runtime or successful Robotiq grasping.

## Installed open-gripper collision snapshot draft

The user's native tool/mount FK passed. Comparing the exported open USD meshes
with GraspGenX's canonical `coll_mesh.obj`, after the source-derived positive
90-degree rotation, gave:

| Extent in grasp frame | Installed USD | GraspGenX |
| --- | --- | --- |
| Finger-opening direction | 154.48 mm | 148.43 mm |
| Depth | 75.00 mm | 75.00 mm |
| Height | 151.78 mm | 151.81 mm |

These are outer bounds, not jaw gaps. Bounds matching would not prove surface
or collision equivalence. The models are not rescaled to force agreement.

`prepare_panda_robotiq_collision.py` uses the **installed USD snapshot** instead
of assuming that the canonical gripper mesh matches it exactly. It checks that
the geometry and model evidence have identical measured joint states, the master
and follower joints are reopened, every enabled gripper collider is included,
and the previously checked model/source hashes are unchanged. It constructs a
convex hull for **each** source collider separately using standard trimesh,
matching the declared USD `convexHull` approximation in intent. This is not exact
PhysX cooking. It never hulls the entire gripper and therefore does not replace
the inter-finger space with one solid block.

Spheres are fitted with the existing cuRobo `fit_spheres_to_mesh`, VOXEL mode,
`sphere_density=1.0`, with no new fitting solver or guessed mounting offsets.
The positive sphere centers/radii are expressed in `panda_hand` and aggregated
as one **frozen open hand** collision group. Each hull and the combined canonical
open-gripper mesh are saved for inspection. Sphere coverage is reported per hull
as vertex outside-distance/count, with a 1 micrometre numerical tolerance. This
is an inspection metric, not a proof of triangle or solid-volume coverage.
Radii are not silently expanded or shrunk to pass.

The generated YAML reuses the stock eight arm links' spheres, arm motion limits,
distance/null-space weights and arm adjacency exclusions. It removes old Panda
finger/hand spheres, old finger locks and the attached-object extension from
this draft only. Stock wrist-to-hand adjacency exclusions remain recorded for
review; new hand/arm collision checks are not blanket-disabled. New hand
self-collision padding is zero, rather than copying the old hand's 20 mm padding
onto the new fitted geometry. Physical gains, friction and force limits are
untouched. Because this is an open snapshot, there are no contact-excluded links
and **closing, lift and handover are unsupported**.

`check_panda_robotiq_collision.py` loads this YAML with collision spheres enabled,
verifies ten FK frames again, checks sphere inventory/finite radii, and runs the
official cuRobo `SelfCollisionCost` at the measured posture. It saves runtime
spheres in robot-base coordinates and the checked sphere pairs. Neither command
starts Isaac Sim, moves the robot, creates attachments or saves a trajectory.

After committing/pushing and pulling these additions, run this block in Linux.
It reuses the user's successful FK run and creates a fresh collision output:

```bash
(
  set -e
  cd /home/suzutaro/GraspGenX
  run=/home/suzutaro/projects/jikkenn1/outputs/panda_robotiq85_connection_20260917_092827
  stamp=$(date +%Y%m%d_%H%M%S)
  collision="$run/collision_open_$stamp"

  uv run --no-sync python \
    /home/suzutaro/projects/jikkenn1/scripts/prepare_panda_robotiq_collision.py \
    --graspgenx-root /home/suzutaro/GraspGenX \
    --geometry "$run/gripper_collision_geometry.json" \
    --prepared-model "$run/fk_model_20260917_093353/model_preparation_check.json" \
    --tool-fk-check "$run/tool_mount_fk_check_20260917_093353.json" \
    --output "$collision"

  uv run --no-sync python \
    /home/suzutaro/projects/jikkenn1/scripts/check_panda_robotiq_collision.py \
    --collision-model "$collision/collision_model_check.json" \
    --output "$collision/runtime_preflight"

  echo "Collision outputs: $collision"
)
```

`open_snapshot_collision_draft_prepared` means spheres/YAML were generated.
`open_snapshot_preflight_passed` means FK, inventory and self collision passed at
one measured posture. It does **not** mean sphere coverage is sufficient or that
the model is executable: `profile_ready`, `safe_to_plan` and `safe_to_execute`
remain false. Review `colliders[].coverage` and the runtime report before adapting
capture/masking, candidate filtering and pregrasp execution. Do not feed this YAML
into the normal end-to-end pipeline or reuse old Panda grasp plans with Robotiq.

## Isolated physical pick diagnostic (not collision-aware deployment)

To inspect whether the replacement gripper can physically retain the hammer,
`run_robotiq_pick_diagnostic.py` reuses a saved capture and SAM3 mask, generates
**fresh Robotiq 2F-85** candidates with the existing official GraspGenX server,
stops that managed server to free GPU memory, and launches at most five separate
Isaac processes in original score order. Every attempt starts from a fresh scene
and the capture's named arm posture. A busy server port is not killed: stop that
server yourself or choose a different `--port`.

This deliberately does **not** make the incomplete collision draft executable.
Neither the 24 mm sphere undercoverage nor the canonical/installed geometry
differences are hidden. World/self-collision avoidance is **not checked** by this
diagnostic; normal PhysX contact remains active. Both `--simulation-only` and
`--allow-collision-unchecked-simulation` are mandatory. Do not use this runner
as the research's collision-aware planner or on a real robot.

`isaac_try_robotiq_pick.py` uses the installed official Panda/Robotiq USD variant
and Isaac Sim's **official Franka LulaKinematicsSolver**, explicitly targeting
`panda_hand`, not the stock `right_gripper` offset. The original Panda arm remains
unchanged. Before approach motion it verifies the measured native mount and the
official Lula FK against actual runtime hand poses. Source-derived Rz(+90 deg)
converts GraspGenX canonical poses to hand poses, without a guessed fingertip
translation, scale correction, mount offset, or original Panda candidate reuse.
This canonical conversion is an explicit diagnostic hypothesis, not proof of
canonical/installed collision-mesh equivalence.

The phases are:

1. Open/settle for 120 frames at the saved arm posture.
2. Solve a pregrasp 100 mm back along canonical grasp +Z, contact waypoints every
   5 mm, then a 150 mm world-Z lift, with official Lula IK using preceding warm
   starts. Reject failed IK, joint limits, or >0.35 rad branch jumps between
   contact/lift waypoints. The initial joint-space approach is not such a small
   Cartesian step and does not use that branch-jump check.
3. Generate all three trajectories before moving using official
   `LulaCSpaceTrajectoryGenerator` with its default limits; stretch time by 3x,
   not force/stiffness/damping. C-space splines interpolate Cartesian IK
   waypoints: this is not a guaranteed exact straight Cartesian path or a
   collision-aware trajectory optimization.
4. Approach and contact while open; settle 60 frames at each phase endpoint.
   Require runtime hand error <=10 mm and <=5 degrees before continuing.
5. Close only the installed `finger_joint` master for 180 frames; preserve its
   authored drive and follower mimic relationships. Lift, settle 60 frames,
   and hold for 180 frames. No fixed joint, surface gripper, friction alteration,
   force multiplier, gravity alteration, or target-pose override is applied.

Arm tracking error >0.35 rad, missing runtime bodies, incompatible models,
invalid samples, or endpoint tracking failure are **implementation/control
errors**: stop the batch immediately. Only expected candidate IK/trajectory
rejections or a completed physical pick failure advance to the next candidate.
Success means target height is >=50 mm above its settled baseline throughout
the final 180-frame hold, and reaches >=50 mm during lift. It is not a handover
success or a force-calibrated hardware result. The first observed pick stops
the batch.

The saved RGB-D contains the old Panda observation. It is useful for this
unchanged-object diagnostic but is **not a new Robotiq camera observation**.
The old plan is read only for its `inputs.capture` and `inputs.segmentation`
paths, never its joint trajectory, chosen grasp, or Panda tool offsets. This
mask may describe the whole object; the command below does not promise a
head-only grasp. Add `--segmentation /absolute/path/to/sam3/parts/grasp_part`
only if you have a verified intended part mask for that same capture.

After committing/pushing these additions, run in Linux:

```bash
cd /home/suzutaro/projects/jikkenn1
git pull --ff-only origin main
conda activate env_isaaclab

python scripts/run_robotiq_pick_diagnostic.py \
  --collision-model outputs/panda_robotiq85_connection_20260917_092827/collision_open_20260917_095618/collision_model_check.json \
  --reference-plan outputs/hammer_head_handover_fixedjoint_e2e_v1/curobo_grasp_lift_trials/candidate_039/grasp_lift_plan_check.json \
  --graspgenx-root /home/suzutaro/GraspGenX \
  --output "outputs/robotiq_pick_diagnostic_$(date +%Y%m%d_%H%M%S)" \
  --max-trials 5 \
  --simulation-only \
  --allow-collision-unchecked-simulation
```

No manual GraspGenX server startup or new environment installation is needed.
The wrapper uses the existing GraspGenX `.venv/bin/python` for inference and the
active Isaac environment's Python for motion. Each Isaac window closes after
its attempt; a successful attempt ends the batch. Without `--headless` you can
watch motion. Reports are `robotiq_pick_trials.json`, per-attempt
`trial_*/robotiq_pick_check.json`, and full server/inference/trial logs. Each
attempt saves measured joint states, commands, hand/target world transforms,
phase labels, planned IK waypoints, tracking errors, and target lift metrics.
Standalone `isaac_try_robotiq_pick.py --keep-open` can retain the GUI at the end;
it is not used by the multi-attempt runner.

Local tests exercise transforms, source binding, candidate provenance, official
API call contracts, phase ordering, hold criteria, and retry/error handling
using mocked Isaac/Lula. They do **not** establish actual pick success; runtime
verification must occur on the user's Linux Isaac Sim installation.
