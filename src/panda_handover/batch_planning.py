"""Pure helpers for evaluating cuRobo batch-planning diagnostics.

The actual GPU planner lives in the pinned cuRobo checkout.  These helpers
keep candidate ordering and comparison policy testable without CUDA.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any


SEQUENTIAL_GRASP_LIFT_SUCCESS_STATUSES = frozenset(
    {
        "success",
        "handover_receive_direction_incompatible",
        "handover_transport_planning_failed",
        "handover_orientation_constraint_failed",
    }
)
SEQUENTIAL_GRASP_LIFT_FAILURE_STATUSES = frozenset(
    {"preflight_failed", "planning_failed"}
)


def candidate_chunks(candidate_count: int, batch_size: int) -> Iterator[range]:
    """Yield deterministic score-ordered slices without dropping candidates."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, candidate_count, batch_size):
        yield range(start, min(start + batch_size, candidate_count))


def pad_candidate_indices(indices: Sequence[int], batch_size: int) -> list[int]:
    """Pad a final chunk to cuRobo's fixed configured batch dimension.

    Padded entries duplicate the final real candidate and are discarded from
    all reported results.  This changes only tensor shape, never the real
    candidate order, thresholds, or acceptance decisions.
    """

    values = [int(value) for value in indices]
    if not values:
        raise ValueError("candidate index chunk must not be empty")
    if batch_size <= 0 or len(values) > batch_size:
        raise ValueError("batch_size must cover the candidate index chunk")
    return values + [values[-1]] * (batch_size - len(values))


def reduce_batch_success(value: Any, active_count: int) -> list[bool]:
    """Reduce all non-batch dimensions of a tensor/array-like success field."""

    import numpy as np

    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        raise ValueError("batch success field must have a batch dimension")
    if active_count <= 0 or active_count > array.shape[0]:
        raise ValueError("active_count is outside the batch success field")
    active = array[:active_count]
    if active.ndim > 1:
        active = active.reshape(active_count, -1).any(axis=1)
    return [bool(item) for item in active]


def batch_prefilter_eligible_sources(report: dict[str, Any]) -> list[int]:
    """Return score-ordered candidates that passed both reviewed batch gates.

    The batch route is only a prefilter.  A candidate must pass the Issue #663
    pose preflight and the official batch grasp/lift planner before the trusted
    sequential planner is allowed to spend a fresh process on its complete
    attached-object handover plan.
    """

    if report.get("status") != "success":
        raise ValueError("batch prefilter report did not complete successfully")
    safety = report.get("safety")
    if not isinstance(safety, dict) or safety.get("diagnosis_only") is not True:
        raise ValueError("batch prefilter report is not a diagnosis-only result")
    if safety.get("replayable_trajectory_saved") is not False:
        raise ValueError("batch prefilter must not claim a replayable trajectory")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("batch prefilter report has no candidates")

    ordered = sorted(candidates, key=lambda item: int(item["original_goalset_rank"]))
    ranks = [int(item["original_goalset_rank"]) for item in ordered]
    if ranks != list(range(len(ordered))):
        raise ValueError("batch prefilter ranks must be contiguous from zero")
    sources = [int(item["source_candidate_index"]) for item in ordered]
    if len(set(sources)) != len(sources):
        raise ValueError("batch prefilter source candidate indices must be unique")
    return [
        source
        for source, item in zip(sources, ordered, strict=True)
        if bool(item.get("strict_issue_663_preflight_success"))
        and bool(item.get("batch_grasp_lift_success"))
    ]


def compare_batch_to_sequential(
    batch_candidates: Sequence[dict[str, Any]],
    sequential_attempts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare feasibility only for candidates evaluated by both planners.

    This intentionally does not claim full algorithmic equivalence: transport
    constraints, selected trajectories, and post-validation are separate gates.
    """

    def unique_by_source(
        values: Sequence[dict[str, Any]], name: str
    ) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for item in values:
            source = int(item["source_candidate_index"])
            if source in result:
                raise ValueError(f"duplicate source candidate {source} in {name}")
            result[source] = item
        return result

    batch = unique_by_source(batch_candidates, "batch report")
    sequential = unique_by_source(sequential_attempts, "sequential manifest")
    common = sorted(set(batch).intersection(sequential))
    rows = []
    mismatches = []
    for source in common:
        batch_ok = bool(batch[source]["batch_grasp_lift_success"])
        plan_status = str(sequential[source].get("plan_status", ""))
        if plan_status in SEQUENTIAL_GRASP_LIFT_SUCCESS_STATUSES:
            sequential_ok = True
        elif plan_status in SEQUENTIAL_GRASP_LIFT_FAILURE_STATUSES:
            sequential_ok = False
        else:
            raise ValueError(
                f"candidate {source} has unclassified sequential plan status "
                f"{plan_status!r}"
            )
        row = {
            "source_candidate_index": source,
            "batch_grasp_lift_success": batch_ok,
            "sequential_plan_status": plan_status,
            "sequential_grasp_lift_success": sequential_ok,
            "sequential_full_plan_feasible": bool(
                sequential[source]["planner_feasible"]
            ),
            "feasibility_matches": batch_ok == sequential_ok,
        }
        rows.append(row)
        if not row["feasibility_matches"]:
            mismatches.append(row)

    missing_from_batch = sorted(set(sequential).difference(batch))
    missing_from_sequential = sorted(set(batch).difference(sequential))
    complete_candidate_coverage = not missing_from_batch and not missing_from_sequential
    feasibility_matches = bool(common) and not mismatches
    return {
        "status": (
            "feasibility_equivalent"
            if complete_candidate_coverage and feasibility_matches
            else "partial_comparison"
            if feasibility_matches
            else "feasibility_mismatch"
        ),
        "common_candidate_count": len(common),
        "complete_candidate_coverage": complete_candidate_coverage,
        "feasibility_matches_on_common_candidates": feasibility_matches,
        "missing_from_batch": missing_from_batch,
        "missing_from_sequential": missing_from_sequential,
        "mismatches": mismatches,
        "candidates": rows,
        "full_pipeline_equivalence_established": False,
        "remaining_equivalence_gates": [
            "identical grasp/lift trajectories and collision post-validation",
            "identical attached-object transport feasibility",
            "identical receive-direction, tilt, and handover-region checks",
            "identical lexicographic candidate selection",
        ],
    }
