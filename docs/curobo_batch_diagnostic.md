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

## Bounded full-handover run

The batch diagnostic above is not a replacement for the complete handover
planner.  To test the unchanged grasp/lift/attached-object transport path on a
fixed score-ordered prefix, use the normal pipeline option below:

```bash
--maximum-grasp-lift-candidates 8
```

This evaluates at most the first eight candidates in the pregrasp score order.
It does not change any IK, collision, receive-direction, tilt, handover-region,
or physics-replay threshold.  The manifest records both the complete candidate
pool size and the evaluated prefix size.  A selected plan is therefore the
best feasible plan within that prefix, not a claim of global optimality over
the unsearched candidates.  Omitting the option preserves the previous search
scope.

## Opt-in batch-prefiltered handover pipeline

For the mug only, the batch grasp/lift result can now be used as a conservative
prefilter before the unchanged complete handover planner. A candidate proceeds
only when both its strict Issue #663 pose preflight and official batch
grasp/lift result succeed. Survivors still run the normal per-candidate
attached-object transport, receive-direction, tilt, handover-region, trajectory
save, and physics-replay gates. The batch result itself is never replayed.

The opt-in route writes separate directories:

- `curobo_batch_grasp_lift_prefilter`
- `curobo_grasp_lift_trials_batch_prefiltered`
- `isaac_grasp_lift_trials_batch_prefiltered`

Use `--batch-grasp-lift-prefilter-size 8`. For the first controlled test, also
use `--maximum-grasp-lift-candidates 8`. The normal pipeline is unchanged when
the batch option is omitted.

This route intentionally refuses `--allow-reviewed-support-contact-preflight`.
The current batch planner does not reproduce that narrow simulation-only
exception, so supported hammer and scissors experiments remain on the reviewed
sequential route.
