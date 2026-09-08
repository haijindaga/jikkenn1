# One-command RGB-D-to-grasp simulation

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
  "receive_part": "scissors handles"
}
```

Invalid JSON, extra fields, an altered object name, or ambiguous part phrases
fail closed. The exact Ollama request, response, model digest when available,
and input-image SHA-256 are saved under `vlm/vlm_part_discovery.json`.
`keep_alive=0` unloads the VLM before SAM3 starts.

An exploratory task instruction can be added without changing the three-field
output contract, for example `--task-instruction "Grasp near the estimated
center of mass."`. The VLM must translate abstract requests into a visible
semantic region suitable for SAM3; neither the VLM nor SAM3 output is treated
as a measured physical center of mass. The exact instruction and resulting
part phrase are saved in the VLM report for manual review.

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

Supplying either handover mode makes `rigid-attachment` the replay default. The
object's current pose relative to `panda_hand` is preserved immediately after
gripper closure with a runtime `UsdPhysics.FixedJoint`; no friction tuning is
used. The cuRobo transport is still planned with the whole-object attached
collision geometry. In manual mode, if
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

For a controlled simulation-only high-drive diagnostic, reuse the same saved
capture and candidate plans with `scripts/isaac_replay_grasp_lift_trials.py`
and pass `--finger-drive-scale 5`. The scale multiplies the source-backed
`max_force` and `stiffness` by 5 and `damping` by `sqrt(5)`. Every candidate in
the run receives the same values, and reports explicitly mark the condition as
diagnostic and not hardware-force calibrated. Do not compare it to a baseline
generated from different candidate plans.

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
