# One-command RGB-D-to-grasp simulation

## Isaac Sim Surface Gripper compatibility preflight

Before adding another grasp-retention abstraction, inspect the exact Surface
Gripper API and bundled example installed with Isaac Sim. This check is
read-only: it does not create a stage, attach an object, or move the robot.

```bash
conda activate env_isaaclab
cd /home/suzutaro/projects/jikkenn1
python scripts/isaac_surface_gripper_preflight.py \
  --headless \
  --output outputs/isaac_surface_gripper_preflight.json
```

Surface Gripper is intended by NVIDIA for suction- or distance-based grippers.
For the Panda experiment it must therefore remain an explicit simulation-only
retention abstraction, compared against the existing FixedJoint baseline; it
must not be presented as a calibrated parallel-jaw contact model.

After the preflight succeeds, replay the same planned candidate with the
separate experimental mode. The runtime copies all nine D6 attachment points
from the bundled `SurfaceGripper_gantry.usda`, preserves their official 3 x 3
grid, and uses the Surface Gripper properties from NVIDIA's Isaac Sim 5.1 code
example. It does not introduce candidate-specific force limits or gains. This
mode keeps both Panda fingers at their open targets; retention comes only from
the explicit Surface Gripper abstraction, not from hidden parallel-jaw contact.
Isaac Sim 5.1 Surface Gripper is CPU-physics only, so this mode requires and
records `--replay-physics cpu`.

```bash
python scripts/isaac_replay_grasp_lift_trials.py \
  --capture outputs/hammer_handover_e2e_v1/capture/camera_0 \
  --plan-trials outputs/hammer_handover_e2e_v1/curobo_grasp_lift_trials \
  --scene-usd scenes/hammer_01.usda \
  --output outputs/hammer_handover_e2e_v1/isaac_grasp_lift_trials_surface_gripper_v1 \
  --max-physical-trials 1 \
  --finger-drive-preset isaaclab-franka \
  --grasp-retention-mode surface-gripper-attachment \
  --replay-physics cpu \
  --simulation-only
```

The one-command runner adds `--replay-physics cpu` automatically only for
`surface-gripper-attachment`. Other retention modes keep their existing physics
backend and finger behavior unchanged.

`scripts/run_sim_grasp_pipeline.py` preserves the existing reviewed stage
boundaries but executes them in order:

1. Isaac RGB-D capture
2. optional Ollama vision inference with strict `object`, `grasp_part`, and
   `receive_part` structured output
3. existing SAM3 multi-prompt segmentation
4. cuRobo observed-point-cloud map preparation with the whole target removed
5. managed GraspGenX server startup, grasp-part inference, and shutdown
6. GraspGenX static scene collision filtering
7. cuRobo pre-grasp and candidate-specific grasp/lift, plus optional attached
   transport planning
8. Isaac replay of at most five candidates, stopping at the first successful
   configured retention trial
9. generation and automatic display of `results.html`

The runner must be launched with the Isaac Lab environment's Python. It invokes
GraspGenX and cuRobo through `/home/suzutaro/GraspGenX/.venv/bin/python`.

## Handle-supported hammer-head diagnostic

To isolate tabletop clearance from grasp force, create a separate diagnostic
scene from the saved hammer-head segmentation.  The source scene is not
modified.  The support position is selected from observed handle points and is
required to leave the observed head projection clear.

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

python scripts/isaac_create_elevated_head_scene.py \
  --base-scene scenes/hammer_01.usda \
  --reference-capture outputs/hammer_head_handover_fixedjoint_e2e_v1/capture/camera_0 \
  --reference-segmentation outputs/hammer_head_handover_fixedjoint_e2e_v1/capture/sam3 \
  --output scenes/hammer_head_handle_supported_v2.usda \
  --head-clearance-m 0.06

python scripts/isaac_open_stage.py \
  --stage scenes/hammer_head_handle_supported_v2.usda \
  --view side
```

Before planning, visually confirm that the cyan block is below the handle and
that the head has air below it after physics settling.  Then run a fresh
pipeline with fixed manual part prompts so only the support condition changes:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/hammer_head_handle_supported_v2.usda \
  --prompt hammer \
  --grasp-part-prompt "hammer head" \
  --receive-part-prompt "hammer handle" \
  --output outputs/hammer_head_handle_supported_e2e_v2 \
  --allow-reviewed-support-contact-preflight
```

This is a simulation-only controlled diagnostic, not a proposed deployment
fixture.  Candidate generation, scores, collision thresholds, cuRobo settings,
and finger drives remain unchanged relative to the baseline command.

If visual review shows that the final grasp is shallow, rerun into a new output
with one common tool-frame depth refinement.  For example, the following adds
10 mm along every candidate's GraspGenX canonical approach axis; it does not
change the 150 mm pre-grasp approach distance or tune individual candidates:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/hammer_head_handle_supported_v2.usda \
  --prompt hammer \
  --grasp-part-prompt "hammer head" \
  --receive-part-prompt "hammer handle" \
  --output outputs/hammer_head_handle_supported_depth10mm_e2e_v1 \
  --grasp-depth-offset-m 0.010 \
  --allow-reviewed-support-contact-preflight
```

The default is `0.0`, so existing commands and artifacts keep their historical
behavior.  Each plan directory saves both the original and adjusted transforms
for a controlled comparison.  A depth-refined run must be replanned; an old
trajectory cannot be reused for this comparison.

The same geometry-derived fixture can be used for scissors without changing
candidate generation or planner settings. Here the segmented handles locate
the support rail and the segmented blade/pivot grasp region must remain clear:

```bash
python scripts/isaac_create_elevated_head_scene.py \
  --base-scene scenes/scissors_01.usda \
  --reference-capture outputs/scissors_vlm_e2e_v1/capture/camera_0 \
  --reference-segmentation outputs/scissors_vlm_e2e_v1/capture/sam3 \
  --output scenes/scissors_handles_supported_v1.usda \
  --target-clearance-m 0.06 \
  --object-label scissors \
  --grasp-part-label "scissors blade near the pivot" \
  --support-part-label "scissors handles"

python scripts/isaac_open_stage.py \
  --stage scenes/scissors_handles_supported_v1.usda \
  --view side
```

After visually confirming the support placement, run a fresh zero-depth
baseline so that support geometry is the only experimental change:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/scissors_handles_supported_v1.usda \
  --prompt scissors \
  --grasp-part-prompt "scissors blade near the pivot" \
  --receive-part-prompt "scissors handles" \
  --output outputs/scissors_handles_supported_e2e_v1 \
  --allow-reviewed-support-contact-preflight
```

## Static visual receiver

For presentation-only handover scenes, compose the official Isaac Sim 5.1 1X
NEO asset over an already saved tabletop scene. Receiver composition is kept
out of the running Isaac Sim authoring process because live composition of the
complex articulated asset caused a native Fabric shutdown. The overlay writer
is pure OpenUSD text composition: it neither opens nor changes either source.

```bash
python scripts/isaac_edit_tabletop_scene.py \
  --output scenes/mug_handover_base_v1.usda \
  --target-usd /home/suzutaro/RoboLab-current/assets/objects/hot3d/mug.usd \
  --exit-after-save

python scripts/create_receiver_overlay.py \
  --base-scene scenes/mug_handover_base_v1.usda \
  --output scenes/mug_handover_1x_neo_v1.usda \
  --receiver-root-xy -1.15 0.0 \
  --receiver-yaw-deg 0

python scripts/isaac_open_stage.py \
  --stage scenes/mug_handover_1x_neo_v1.usda
```

The default `side` view creates an upright, temporary camera that frames the
robot/table and receiver from the side. It uses the same world-axis camera pose
conversion as the RGB-D capture pipeline and is authored only in the USD
session layer, so it is never saved into the scene. Use `--view capture` to see
the exact authored `/World/camera_0` RGB-D view; use `--camera-prim PATH` with
that mode only when the authored camera has a different path.

The reviewed coordinate convention puts the robot base at the origin, the
table in world +X, and the visual receiver in world -X. Isaac Sim uses +X as
the world forward direction, so zero yaw is the documented convention-based
starting point and remains subject to visual confirmation of this asset's local
facing direction. The receiver Z is not chosen by eye: the checked-in evidence
records the measured official NEO lower bound as `-0.8406724618970056 m`, and
the generator applies `root_z = floor_z - measured_lower_bound_z`. The source
asset is not rescaled, its physics is not edited, and the resulting overlay is
not authorized for physical replay or human-safety evaluation.

For the reviewed scissors scene with Ollama part discovery, choose an installed
vision model explicitly:

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/scissors_01.usda \
  --target-object scissors \
  --ollama-model gemma3:12b \
  --output outputs/scissors_e2e_v1 \
  --allow-reviewed-support-contact-preflight
```

The target name is authoritative. The VLM must return exactly:

```json
{
  "object": "scissors",
  "grasp_part": "scissors blade near the pivot",
  "receive_part": "scissors handles",
  "transport_orientation_policy": "free"
}
```

Invalid JSON, extra fields, an altered object name, or ambiguous part phrases
fail closed. The exact Ollama request, response, model digest when available,
and input-image SHA-256 are saved under `vlm/vlm_part_discovery.json`.
`keep_alive=0` unloads the VLM before SAM3 starts.

An exploratory task instruction can be added without changing the four-field
output contract, for example `--task-instruction "Grasp near the estimated
center of mass."`. The VLM must translate abstract requests into a visible
semantic region suitable for SAM3; neither the VLM nor SAM3 output is treated
as a measured physical center of mass. The exact instruction and resulting
part phrase are saved in the VLM report for manual review.

`transport_orientation_policy` is a closed two-value decision, never a
VLM-generated angle. `free` retains the existing affordance-axis handover.
`keep_grasp_orientation` fixes the automatic handover goal to the selected
grasp rotation while still placing the receive-part median at the requested
receiver position. The saved lift and transport trajectories are then checked
with cuRobo FK at every waypoint against that grasp rotation. A candidate that
changes orientation by more than the project-wide 2 degree model-alignment
tolerance is rejected and the trial runner proceeds to the next candidate.
This is a goal constraint plus fail-closed path validation: the pinned cuRobo
V2 planner does not expose the legacy running pose-cost API. The report does
not claim a path-wide constrained optimizer. When orientation is preserved,
the potentially conflicting human-direction alignment is not enforced.

In VLM mode, only the saved `parts/grasp_part` mask is sent to GraspGenX for
candidate generation. The whole-object mask remains authoritative for removing
the target from the observed collision map, filtering candidates against the
surrounding scene, and constructing cuRobo's attached-object geometry. A
missing or undersampled grasp-part mask fails the inference stage; it never
silently falls back to whole-object grasp generation. `results.html` records
both mask paths and their distinct roles.

The previous manual whole-object mode remains available for controlled
comparisons and debugging:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/scissors_01.usda \
  --prompt scissors \
  --output outputs/scissors_manual_e2e_v1 \
  --allow-reviewed-support-contact-preflight
```

For an affordance-aware handover experiment, use VLM or manual part prompts and
specify (1) where the segmented receive-part representative point should be and
(2) the direction from the robot-held part toward the human, both in the Panda
base frame:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/hammer_01.usda \
  --target-object hammer \
  --ollama-model gemma3:12b \
  --output outputs/hammer_affordance_handover_e2e_v1 \
  --handover-receiver-position-robot-base-m X Y Z \
  --handover-human-direction-robot-base DX DY DZ \
  --allow-reviewed-support-contact-preflight
```

When the receiver height is not part of the task, add
`--handover-height-policy preserve-lift-end`. The requested receiver X/Y are
still used, while its Z is replaced by the measured receive-part height at the
end of the existing lift phase. This requests no additional vertical motion
during transport; it does not remove the initial table-clearance lift.

This mode first rejects candidates whose official Franka collision mesh enters
the observed receive-part clearance region (15 mm by default), then preserves
the original GraspGenX score order. It adds no weighted handover score. For each
remaining grasp, the grasp-part-to-receive-part axis is aligned with the human
direction, the receive-part median is placed at the requested position, and
cuRobo tests six fixed roll variants with the whole object attached. Candidate
planning continues until the configured number of complete attached transport
plans is available. This is a receive-region clearance proxy; it does not model
the human body, gaze, or true multi-view visibility.

The previous manual transport mode remains available when a fully reviewed
`panda_hand` pose is already known:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/hammer_01.usda \
  --prompt hammer \
  --output outputs/hammer_handover_e2e_v1 \
  --handover-goal-position-robot-base-m X Y Z \
  --allow-reviewed-support-contact-preflight
```

Supplying either handover mode makes `rigid-attachment` the replay default.
Before reset, the runner enables Isaac Sim's standard
`PhysxContactReportAPI` on the target. During closure it creates no attachment
until both Panda finger rigid bodies report target contact in the same physics
step and both finger joints have moved inward by at least 1 mm. `close_frames`
is the maximum time allowed for that state, not an unconditional delay. A
failed gate saves `fixed_joint_attachment_gate_failure.json` and does not lift.
After the gate passes, the runner creates a standard OpenUSD `FixedJoint`
between `panda_hand` and the target at their current simulated relative pose,
so neither body is deliberately snapped to a pre-authored frame. It holds that
state for five physics steps and verifies the relative pose before starting the
lift; an unstable joint is reported instead of being hidden as a failed grasp.
This ordering follows NVIDIA's recommended closure/contact-triggered runtime
FixedJoint pattern:
<https://forums.developer.nvidia.com/t/pick-and-place-in-space-zero-gravity-0g/370975>.
`physx-auto-attachment` remains available only as an experimental diagnostic:
the PhysX attachment schema is defined for an attachment containing at least
one deformable actor and did not constrain this rigid-body-to-rigid-body case.
`kinematic-pose-lock` is an explicit exact-following simulation fallback. It
clears the target's residual velocity, switches the target rigid body to the
standard USD kinematic state, and updates its world pose from the measured
post-close target-to-hand transform after every physics step. In this mode the
visible frame is rendered only after that pose update, keeping the viewport and
saved images synchronized with the measured trace. It creates no physics joint
and therefore does not apply attachment forces to the Panda.
This mode assumes grasp success; it is not contact-only grasp evidence.

No friction tuning is used by any attachment mode. Every attachment replay
saves the panda-hand pose and the translational and angular drift of the
target-to-hand transform at every physics sample. Attachment-mode success now
requires both lift retention and a maximum relative-pose drift of at most 5 mm
and 5 degrees; object height alone cannot produce a false success. The cuRobo
transport is still planned with the whole-object attached collision geometry.
In manual mode, if
`--handover-goal-quaternion-wxyz W X Y Z` is omitted, the selected grasp
orientation is preserved. Reports distinguish this explicit
no-slip grasp assumption from contact-only physical-pick evidence. Use
`--grasp-retention-mode physics` only when frictional retention itself is the
quantity being evaluated.

Use `--headless` to suppress Isaac windows. Defaults retain the current robust
candidate policy: 500 generated grasps, top 300 returned, 5 mm static collision
threshold, up to 100 pre-grasp candidates, and at most five physical replays.
All values remain explicit CLI options. The result page is generated after
both success and failure and opens in the default browser. `--headless` or
`--no-show-results` suppresses only automatic opening; `results.html` is still
saved. It embeds every saved image, provides expandable JSON previews, and
links every NPY, PLY, log, and report artifact.

## Inspect all grasp candidates in Viser

The read-only candidate viewer follows GraspGenX's official visualization
semantics without rerunning inference or inventing a new feasibility score:
red is rejected by the saved static collision filter, green passed that filter,
yellow was later rejected by cuRobo, blue has a successful cuRobo plan, and
magenta marks a planner/infrastructure error. A light-blue gripper mesh marks
the highest-score statically collision-free candidate. Scene points and grasp
poses are both transformed through the saved `T_world_camera` and displayed in
the Isaac world frame with +Z up, avoiding the former OpenCV-camera/world-up
ambiguity.

```bash
cd /home/suzutaro/GraspGenX

uv run --no-sync python \
  /home/suzutaro/projects/jikkenn1/scripts/visualize_grasp_candidates.py \
  --capture /home/suzutaro/projects/jikkenn1/outputs/hammer_handover_e2e_v1/capture/camera_0 \
  --segmentation /home/suzutaro/projects/jikkenn1/outputs/hammer_handover_e2e_v1/capture/sam3 \
  --candidates /home/suzutaro/projects/jikkenn1/outputs/hammer_handover_e2e_v1/graspgenx_candidates \
  --filtered /home/suzutaro/projects/jikkenn1/outputs/hammer_handover_e2e_v1/graspgenx_candidates_filtered \
  --plan-trials /home/suzutaro/projects/jikkenn1/outputs/hammer_handover_e2e_v1/curobo_grasp_lift_trials \
  --port 8081
```

Omit `--plan-trials` for the official-style red/green static-filter view.
The default shows all candidates. Use `--max-candidates N` only for a less
cluttered presentation view; selection is then deterministic score order.

For a controlled simulation-only high-drive diagnostic, reuse the same saved
capture and candidate plans with `scripts/isaac_replay_grasp_lift_trials.py`
and pass `--finger-drive-scale 5`. The scale multiplies the source-backed
`max_force` and `stiffness` by 5 and `damping` by `sqrt(5)`. Every candidate in
the run receives the same values, and reports explicitly mark the condition as
diagnostic and not hardware-force calibrated. Do not compare it to a baseline
generated from different candidate plans.

For a friction-only retention diagnostic, keep `--finger-drive-scale 1`, use
`--grasp-retention-mode physics`, and pass (for example)
`--fingertip-friction-coefficient 5`. The runner creates a runtime physics
material with static and dynamic friction both equal to the requested value,
restitution zero, and PhysX friction combine mode `max`. It binds that material
to the editable left and right Panda finger links. Standard USD material
inheritance carries the binding into instance-proxy collision geometry, and the
runner verifies the resolved material on every fingertip collider before
starting the replay. The target and table materials are not changed, and the
authored scene USD is not saved. This prevents the object/table settling
behavior from becoming a second experimental variable. Friction and
finger-drive diagnostics cannot be enabled together in one controlled run.

```bash
python scripts/isaac_replay_grasp_lift_trials.py \
  --capture outputs/hammer_vlm_part_e2e_v1/capture/camera_0 \
  --plan-trials outputs/hammer_vlm_part_e2e_v1/curobo_grasp_lift_trials \
  --scene-usd scenes/hammer_01.usda \
  --output outputs/hammer_vlm_part_e2e_v1/isaac_grasp_lift_trials_friction5_v1 \
  --max-physical-trials 5 \
  --finger-drive-preset isaaclab-franka \
  --finger-drive-scale 1 \
  --fingertip-friction-coefficient 5 \
  --grasp-retention-mode physics \
  --simulation-only
```

The coefficient is an intentionally nonphysical sensitivity test, not a rubber
calibration or a real-robot setting. Compare it only with a standard-friction
run using the same capture and candidate-plan manifest.

For a controlled FixedJoint solver diagnostic, first replay the same saved
candidate plans with the default solver settings, then use a new output name
and repeat with `--solver-position-iterations 64` and
`--solver-velocity-iterations 4`. The override is applied equally to the Panda
articulation and target rigid body and is recorded with before/after readback in
the JSON report. It does not alter the source USD, grasp candidates, or cuRobo
trajectory, and is not the pipeline default.

To generate or reopen the same report for an existing output directory:

```bash
python scripts/show_experiment_results.py \
  --output outputs/scissors_e2e_v1
```

The GraspGenX port (5556 by default) must be free before starting. The runner
starts the server only for inference and terminates only the child process it
created. It deliberately refuses to kill a pre-existing server.

If a stage fails, inspect `pipeline_status.json`. Correct the reported issue and
rerun the same command with `--resume`; only stages with a successful saved JSON
report are skipped. An incomplete stage directory is moved under
`failed_stage_outputs/` before that stage is retried, so its diagnostics are not
destroyed. Use a new output directory to start a genuinely new trial.

This command remains simulation-only. It does not authorize or command a real
robot.
