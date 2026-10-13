# cuRobo batch-planning diagnostic

This route evaluates the pinned cuRobo `BatchMotionPlanner` without replacing
the reviewed sequential grasp/handover pipeline.

It keeps one robot model, observed collision world, and GPU planner alive, and
submits score-ordered grasp candidates in fixed-size batches. `--batch-size`
is only a GPU resource cap: it does not change candidate scores, constraints,
or thresholds.

The pinned cuRobo goal registry fixes its leading batch dimension during
warmup. If the final chunk is smaller than `--batch-size`, the diagnostic
duplicates its final real candidate to fill the tensor, then discards every
padded result row. This is shape padding only; it does not add, remove, or
reorder real candidate decisions.

The first gate covers strict Issue #663 preflight and approach/grasp/lift
feasibility. It does **not** yet save replayable trajectories, apply the
reviewed support-contact fallback, or batch the attached-object handover.

## First bounded run

Run a small prefix before evaluating every candidate:

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

root=outputs/mug_joint_handover_region_20261012_234616
tag=$(date +%Y%m%d_%H%M%S)

/home/suzutaro/GraspGenX/.venv/bin/python \
  scripts/curobo_batch_grasp_lift_diagnostic.py \
  --capture "$root/capture/camera_0" \
  --pregrasp-plan "$root/curobo_pregrasp" \
  --output "$root/curobo_batch_diagnostic_first8_$tag" \
  --batch-size 4 \
  --maximum-candidates 8
```

The batch size and prefix are diagnostic resource bounds, not candidate-level
tuning. If this completes without a CUDA allocation failure, rerun all
candidates using a new output directory and the same batch size, omitting
`--maximum-candidates`.

## Feasibility comparison

Compare an all-candidate batch report with the already completed sequential
manifest:

```bash
batch_dir="$root/curobo_batch_diagnostic_all_<timestamp>"

python scripts/compare_curobo_batch_sequential.py \
  --batch-report "$batch_dir/batch_grasp_lift_diagnostic.json" \
  --sequential-manifest \
    "$root/curobo_grasp_lift_trials/grasp_lift_trial_plans.json" \
  --output "$batch_dir/sequential_feasibility_comparison.json"
```

Even `feasibility_equivalent` does not authorize replacing the normal path.
The comparison uses the sequential candidate's recorded stage status, so a
candidate that completed grasp/lift but later failed handover still counts as
a grasp/lift success. It does not incorrectly compare batch grasp/lift against
the sequential end-to-end `planner_feasible` field.

Trajectory equality, full collision post-validation, attached-object transport,
receive direction, tilt, handover region, and final lexicographic selection
remain separate gates.
