"""OPTIONAL: flag transactions the shards disagree sharply about.

Off unless DISAGREEMENT_THRESHOLD is set above 0. Nothing in the erasure
guarantee, the manifest, the proofs, or the signing path depends on this file
existing -- delete it and the rest of the system is unchanged. That separation
is deliberate: this is an experiment about fraud detection bolted alongside a
compliance system, and the two should not be able to break each other.

Why it might be worth having: the ensemble mean is what gets served, but the
*spread* across shards is an epistemic-uncertainty signal. When most shards
say "normal" and one says "suspicious", that disagreement can mean a pattern
only some shards have been trained on. Measured on this repo's frozen eval
corpus, spread scores AUC 0.574 as a fraud detector against the served mean's
0.515 -- weaker than a real fraud model, but genuinely more informative than
the number we currently serve, which is the surprising part.

What it deliberately does NOT record: transaction features. See
db/schema.sql's disagreement_reviews comment for the full reasoning -- in
short, PredictRequest carries no subject_id, so a row here can never be
reached by an erasure, and storing features would build a store of personal
data this system promises to be able to delete and could not.
"""

import numpy as np

from config.settings import DISAGREEMENT_THRESHOLD
from db.conn import pooled


def is_enabled() -> bool:
    return DISAGREEMENT_THRESHOLD > 0.0


def spread(shard_probabilities: np.ndarray) -> float:
    """Population std across shards for a single row.

    Population (ddof=0), not sample: these five shards are the entire ensemble,
    not a sample drawn from some larger population of shards, so there is no
    n-1 correction to make.
    """
    return float(np.std(shard_probabilities, ddof=0))


def record(model_version: str, shard_scores: np.ndarray, mean_score: float,
           spread_value: float) -> None:
    """Insert one review row. Runs in a FastAPI BackgroundTask, i.e. after the
    response has already gone back to the caller, so a slow or failing insert
    cannot delay or fail a prediction.

    Swallows its own exceptions for that same reason: this is an optional
    side-channel, and a broken review queue must never turn a successful
    prediction into a 500. The trade-off is that failures here are silent --
    acceptable for an off-by-default experiment, not acceptable if this ever
    becomes load-bearing, at which point it needs a real error counter.
    """
    try:
        with pooled() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO disagreement_reviews
                        (model_version, shard_scores, mean_score, spread, threshold)
                    VALUES (%s, %s, %s, %s, %s)
                """, (model_version, [float(s) for s in shard_scores],
                      float(mean_score), float(spread_value), DISAGREEMENT_THRESHOLD))
            conn.commit()
    except Exception:
        # ponytail: silent by design (see docstring). Wire to the metrics
        # counter alongside the Phase 7 drift alert if this stops being
        # off-by-default.
        pass
