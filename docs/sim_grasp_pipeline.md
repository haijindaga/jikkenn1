# One-command RGB-D-to-grasp simulation

`scripts/run_sim_grasp_pipeline.py` preserves the existing reviewed stage
boundaries but executes them in order:

1. Isaac RGB-D capture
2. optional Ollama vision inference with strict `object`, `grasp_part`, and
   `receive_part` structured output
3. existing SAM3 multi-prompt segmentation
4. cuRobo observed-point-cloud map preparation
5. managed GraspGenX server startup, inference, and shutdown
6. GraspGenX static scene collision filtering
7. cuRobo pre-grasp and candidate-specific grasp/lift planning
8. Isaac physical replay of at most five candidates, stopping at the first pick
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
  --ollama-model qwen3-vl:4b \
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

The previous manual whole-object mode remains available for controlled
comparisons and debugging:

```bash
python scripts/run_sim_grasp_pipeline.py \
  --scene-usd scenes/scissors_01.usda \
  --prompt scissors \
  --output outputs/scissors_manual_e2e_v1 \
  --allow-reviewed-support-contact-preflight
```

Use `--headless` to suppress Isaac windows. Defaults retain the current robust
candidate policy: 500 generated grasps, top 300 returned, 5 mm static collision
threshold, up to 100 pre-grasp candidates, and at most five physical replays.
All values remain explicit CLI options. The result page is generated after
both success and failure and opens in the default browser. `--headless` or
`--no-show-results` suppresses only automatic opening; `results.html` is still
saved. It embeds every saved image, provides expandable JSON previews, and
links every NPY, PLY, log, and report artifact.

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
