"""Slice assignment within a shard. Subject-aligned, not record-aligned.

The plan this project started from ordered *records* by recency and cut slices
at record boundaries. That is correct for textbook SISA, where the deletion unit
is one data point. It is wrong here, where the deletion unit is a subject who
owns many records.

Under record-level slicing a subject's rows scatter across slices, and the
rollback point is the MINIMUM slice index among them. For k records spread over
n slices, E[min] is about n/(k+1) -- at k=10, n=5 that is slice 0, a full-shard
retrain. The saving slicing exists to provide disappears exactly when subjects
own more than one record, which is always.

So slices are cut at subject boundaries: every record a subject owns lands in
one slice. `min_slice_idx == max_slice_idx` for every subject, and
`test_slicer.py` enforces it. That invariant is what makes the rollback point in
db/schema.sql meaningful.
"""

import numpy as np

from config.settings import NUM_SLICES


def assign_slices(subject_ids: np.ndarray, churn: np.ndarray, record_counts: np.ndarray,
                  num_slices: int = NUM_SLICES) -> np.ndarray:
    """Order subjects by churn ascending, then pack into equal-record-count slices.

    Ascending churn puts the subjects most likely to be deleted in the LAST
    slice, where rollback is cheapest. The plan used recency as a proxy for this
    because PaySim has no churn signal; we synthesise one in data/churn_score.py,
    so we key on it directly instead of on its proxy.

    Packing targets equal record counts rather than equal subject counts --
    subjects own wildly uneven numbers of records, and equal-subject slices
    would make training time per slice just as uneven.
    """
    if num_slices < 1:
        raise ValueError(f"num_slices must be >= 1, got {num_slices}")

    # Ties broken by subject_id so the ordering is total, not merely sorted.
    order = np.lexsort((subject_ids, churn))
    cumulative = np.cumsum(record_counts[order])
    total = cumulative[-1]

    # Right-open bucketing on the running record count. `- 1` because cumsum is
    # inclusive: a subject whose records complete a boundary belongs to the
    # slice it filled, not the next one.
    slice_of_ordered = np.minimum(((cumulative - 1) * num_slices) // total, num_slices - 1)

    out = np.empty(len(subject_ids), dtype=np.int64)
    out[order] = slice_of_ordered
    return out
