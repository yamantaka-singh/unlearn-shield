"""Model serving. Never trains, never writes.

Checkpoint paths come from the DB's `checkpoints.file_path` -- the
content-addressed copy made at promotion (ADR 0006) -- and never from
`engine.train.checkpoint_path`, which is a conventional name later rebuilds
overwrite in place.

Weights are loaded once per promoted model_version and cached
(inference/batched_ensemble.py). The cache key is that version's tuple of
checkpoint hashes, so a rebuild that promotes new weights yields a new key and
the old entry stops being reachable. That property is what keeps an erasure
from being undone by the serving layer: a cache invalidated by hand would
eventually miss one, and the symptom -- a model that keeps scoring with erased
data in it, silently, while every job says `done` -- is exactly what this
project exists to prevent.
"""

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from config.settings import CHECKPOINT_DIR, DISAGREEMENT_THRESHOLD
from data.synth import TYPES
from db.conn import pooled
from gateway import disagreement
from gateway.auth import require_scope
from gateway.schemas import PredictRequest, PredictResponse
from inference.batched_ensemble import load_ensemble

router = APIRouter(prefix="/v1", tags=["predict"])


def _current_model(cur) -> tuple[str, dict]:
    cur.execute("""
        SELECT model_version, shard_checkpoints FROM model_versions
        ORDER BY promoted_at DESC LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        raise HTTPException(503, "no promoted model_version -- run scripts.load_routing after a build")
    model_version, shard_checkpoints = row

    cur.execute("SELECT shard, file_path FROM checkpoints WHERE checkpoint_hash = ANY(%s)",
                (list(shard_checkpoints.values()),))
    shard_paths = {str(shard): path for shard, path in cur.fetchall()}

    missing = set(shard_checkpoints) - set(shard_paths)
    if missing:
        # Serving a partial ensemble would quietly change what the score means
        # while still returning 200 and a model_version that claims otherwise.
        raise HTTPException(503, f"model_version {model_version} references checkpoints "
                                 f"with no file on record for shard(s) {sorted(missing)}")
    return model_version, shard_paths


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, background: BackgroundTasks,
            principal: str = Depends(require_scope("predict:invoke"))):
    with pooled() as conn:
        with conn.cursor() as cur:
            model_version, shard_paths = _current_model(cur)

    preproc_paths = {s: f"{CHECKPOINT_DIR}/shard{s}_preproc.json" for s in shard_paths}
    ensemble = load_ensemble(shard_paths, preproc_paths)

    record = {
        "step": np.array([body.step]),
        "amount": np.array([body.amount]),
        "oldbalanceOrg": np.array([body.oldbalanceOrg]),
        "newbalanceOrig": np.array([body.newbalanceOrig]),
        "oldbalanceDest": np.array([body.oldbalanceDest]),
        "newbalanceDest": np.array([body.newbalanceDest]),
        "type_idx": np.array([TYPES.index(body.type)]),
    }
    # Per-shard scores rather than predict_proba: same single forward pass,
    # and the mean below is exactly what predict_proba would have returned.
    # Taking them here is what lets the optional disagreement check cost one
    # np.std over n_shards floats instead of a second inference.
    shard_scores = ensemble.shard_probabilities(record, np.array([0]))[:, 0]
    probability = float(shard_scores.mean())

    if disagreement.is_enabled():
        spread = disagreement.spread(shard_scores)
        if spread > DISAGREEMENT_THRESHOLD:
            # BackgroundTasks runs after the response is sent, so the insert
            # is off the hot path entirely -- the caller is never waiting on
            # the review queue.
            background.add_task(disagreement.record, model_version,
                                shard_scores, probability, spread)

    return PredictResponse(fraud_probability=probability, model_version=model_version)
