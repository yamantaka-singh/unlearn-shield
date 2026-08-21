"""Shard assignment. Runs once at ingest; the result is frozen.

Random assignment scatters deletions evenly, so almost every request triggers a
rebuild somewhere and sharding buys nothing. Assigning by expected deletion
likelihood concentrates churn-prone subjects into a few hot shards and leaves
the rest cold, so most rebuilds touch a small fraction of the model.
"""

from hashlib import blake2b

import numpy as np

from config.settings import HOT_SHARDS, HOT_THRESHOLD, NUM_SHARDS


def stable_hash(value: str) -> int:
    """Deterministic across processes and runs, which `hash()` is not.

    Python's `hash()` for str is salted by PYTHONHASHSEED. Using it here would
    reshuffle shard assignment between runs and silently invalidate every
    checkpoint's rollback point.
    """
    return int.from_bytes(blake2b(value.encode(), digest_size=8).digest(), "big")


def assign_shard(subject_id: str, churn_score: float,
                 num_shards: int = NUM_SHARDS, hot_shards: int = HOT_SHARDS,
                 hot_threshold: float = HOT_THRESHOLD) -> int:
    if not 0 < hot_shards < num_shards:
        raise ValueError(f"hot_shards must be in (0, {num_shards}), got {hot_shards}")
    if churn_score >= hot_threshold:
        return stable_hash(subject_id) % hot_shards
    return hot_shards + (stable_hash(subject_id) % (num_shards - hot_shards))


def assign_all(subject_ids: np.ndarray, churn: np.ndarray, **kwargs) -> np.ndarray:
    return np.array([assign_shard(s, c, **kwargs) for s, c in zip(subject_ids, churn)],
                    dtype=np.int64)
