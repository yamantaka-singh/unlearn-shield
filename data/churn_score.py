"""PLACEHOLDER. Synthesises a per-subject churn likelihood in [0, 1).

PaySim has no churn, consent-type, or tenure signal, and this project needs one
to decide which shard a subject belongs in. Everything here is invented.

A production deployment replaces this wholesale with a real signal -- consent
type, contract tenure, a churn model's score. What must NOT change is the
contract: one score per subject, in [0, 1), computed once at ingest and then
frozen. Recomputing it later and letting the routing table follow would
invalidate the rollback point of every existing checkpoint.
"""

import numpy as np


def churn_scores(subject_ids: np.ndarray, last_step: np.ndarray, max_step: int,
                 seed: int = 0) -> np.ndarray:
    """Recency-weighted score: recently-active subjects score higher.

    The stand-in premise is that recent arrivals churn sooner than long-tenured
    accounts. Real deployments should not inherit this premise -- it is here so
    the shard assignment has something non-uniform to key on, nothing more.
    """
    rng = np.random.default_rng(seed)
    recency = last_step / max(max_step, 1)
    noise = rng.beta(2.0, 2.0, size=len(subject_ids))
    return np.clip(0.65 * recency + 0.35 * noise, 0.0, 0.999)
