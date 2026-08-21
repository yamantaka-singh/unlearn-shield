"""Model serving. Reads checkpoints from disk; never trains, never writes.

Ensembles shards sequentially -- one forward pass per shard, summed and
averaged. Phase 5's inference/batched_ensemble.py replaces the loop below with
one batched pass across stacked shard parameters; this route calls into it once
that module exists. Correctness first, the S-times-fewer-passes optimisation
after.
"""

import numpy as np
import torch
from fastapi import APIRouter, Depends

from config.settings import CHECKPOINT_DIR
from data.synth import TYPES
from db.conn import connect
from engine.model import build_model
from engine.preprocessing import ShardPreprocessor
from gateway.auth import require_scope
from gateway.schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/v1", tags=["predict"])


def _load_shard_model(file_path: str) -> torch.nn.Module:
    """Loads from the DB's recorded file_path -- the content-addressed copy
    worker/jobs.py::_promote makes at promotion time -- never from
    engine.train.checkpoint_path(shard, slice_idx) directly. That path is a
    fixed name engine/train.py overwrites on every rebuild of the shard, so
    reconstructing it here could silently load whatever the most recent
    rebuild left behind instead of the checkpoint this model_version actually
    promoted."""
    model = build_model()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()
    return model


def _load_preprocessor(shard: int) -> ShardPreprocessor:
    with open(f"{CHECKPOINT_DIR}/shard{shard}_preproc.json") as f:
        return ShardPreprocessor.from_json(f.read())


def _current_model_version(cur) -> tuple[str, dict]:
    cur.execute("""
        SELECT model_version, shard_checkpoints FROM model_versions
        ORDER BY promoted_at DESC LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("no promoted model_version -- run scripts/load_routing after a build")
    return row[0], row[1]


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, principal: str = Depends(require_scope("predict:invoke"))):
    conn = connect()
    try:
        with conn.cursor() as cur:
            model_version, shard_checkpoints = _current_model_version(cur)
            cur.execute("""
                SELECT shard, file_path FROM checkpoints WHERE checkpoint_hash = ANY(%s)
            """, (list(shard_checkpoints.values()),))
            shard_paths = dict(cur.fetchall())
    finally:
        conn.close()

    record = {
        "step": np.array([body.step]),
        "amount": np.array([body.amount]),
        "oldbalanceOrg": np.array([body.oldbalanceOrg]),
        "newbalanceOrig": np.array([body.newbalanceOrig]),
        "oldbalanceDest": np.array([body.oldbalanceDest]),
        "newbalanceDest": np.array([body.newbalanceDest]),
        "type_idx": np.array([TYPES.index(body.type)]),
    }
    rows = np.array([0])

    probs = []
    for shard_str, file_path in shard_paths.items():
        shard = int(shard_str)
        preproc = _load_preprocessor(shard)
        x = torch.from_numpy(preproc.transform(record, rows))
        with torch.no_grad():
            probs.append(torch.sigmoid(_load_shard_model(file_path)(x)).item())

    return PredictResponse(fraud_probability=float(np.mean(probs)), model_version=model_version)
