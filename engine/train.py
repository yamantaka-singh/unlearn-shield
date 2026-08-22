"""Build shards and train each one slice by slice, checkpointing as it goes.

Offline CLI. No network, no database -- the routing table is written as JSON and
loaded into subject_shard_map by Phase 4, which owns the DB.

    python -m engine.train --build       # partition and write shard files
    python -m engine.train               # train every shard
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from config.determinism import enforce_determinism, state_dict_digest
from config.settings import (BATCH_SIZE, CHECKPOINT_DIR, EPOCHS_PER_SLICE, LEARNING_RATE,
                             NUM_SHARDS, NUM_SLICES, SEED, SHARD_DIR, TENANT_ID, subject_ref)
from data.churn_score import churn_scores
from data.synth import generate
from engine.model import build_model
from engine.preprocessing import ShardPreprocessor
from engine.sharder import assign_all
from engine.slicer import assign_slices

COLUMNS = ("step", "type_idx", "amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest", "isFraud")


def shard_path(shard: int) -> str:
    return os.path.join(SHARD_DIR, f"shard{shard}.npz")


def checkpoint_path(shard: int, slice_idx: int, checkpoint_dir: str | None = None) -> str:
    return os.path.join(checkpoint_dir or CHECKPOINT_DIR, f"shard{shard}_slice{slice_idx}.pt")


def load_shard(shard: int) -> dict:
    with np.load(shard_path(shard), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_shard(shard: int, records: dict) -> None:
    os.makedirs(SHARD_DIR, exist_ok=True)
    np.savez(shard_path(shard), **records)


def build(n_subjects: int = 800, seed: int = SEED, max_step: int | None = None,
         raw: dict | None = None) -> dict:
    """Partition raw records into shard files plus a routing table.

    Runs once. Shard and slice assignment are frozen afterwards: recomputing
    churn later and letting the routing follow it would invalidate the rollback
    point of every checkpoint already on disk.

    `raw` overrides the synthetic generator with a pre-loaded records dict of
    the same schema (data.prepare.load's real-PaySim output, for instance).
    Everything past this point is schema-generic; only the row source changes.

    `max_step=None` (the default) derives the simulation horizon from whichever
    data is actually in play: 720 for the synthetic generator, `raw["step"].max()`
    for real data. Leaving that to a hardcoded 720 regardless of `raw` was a
    real bug, found by running real PaySim through this exact path:
    churn_score.py computes `recency = last_step / max_step`, and a real
    subsample rarely spans the full simulation -- 150k real rows span steps
    1-153, not 1-720. Recency then tops out around 0.21, which combined with
    the noise term can never reach HOT_THRESHOLD (0.6), so every subject lands
    in a cold shard and the entire hot/cold concentration this sharding scheme
    exists for produces nothing -- silently, no error, no warning. Pass
    `max_step` explicitly only to override the derived horizon.
    """
    if raw is None:
        raw = generate(n_subjects=n_subjects, seed=seed, max_step=max_step or 720)
        max_step = max_step or 720
    elif max_step is None:
        max_step = float(raw["step"].max())
    subjects, inverse = np.unique(raw["nameOrig"], return_inverse=True)
    counts = np.bincount(inverse)

    last_step = np.zeros(len(subjects))
    np.maximum.at(last_step, inverse, raw["step"])
    churn = churn_scores(subjects, last_step, max_step, seed=seed)

    shard_of_subject = assign_all(subjects, churn)
    slice_of_subject = np.full(len(subjects), -1, dtype=np.int64)
    for k in range(NUM_SHARDS):
        m = shard_of_subject == k
        slice_of_subject[m] = assign_slices(subjects[m], churn[m], counts[m])

    refs = np.array([subject_ref(s) for s in subjects])
    routing = {}
    for k in range(NUM_SHARDS):
        rows = np.flatnonzero(shard_of_subject[inverse] == k)
        records = {c: raw[c][rows] for c in COLUMNS}
        records["subject_ref"] = refs[inverse[rows]]
        records["slice_idx"] = slice_of_subject[inverse[rows]]
        save_shard(k, records)

    for i, ref in enumerate(refs):
        routing[ref] = {"tenant_id": TENANT_ID, "shard": int(shard_of_subject[i]),
                        "min_slice_idx": int(slice_of_subject[i]),
                        "record_count": int(counts[i])}
    save_routing(routing)
    return routing


def save_routing(routing: dict) -> None:
    """Write the routing table. Every writer goes through here (build above,
    engine/rebuild.py, engine/gbdt.py) so the format cannot drift between the
    initial write and the rewrites an erasure performs.

    Compact, and not sorted or indented. At 1.6M subjects the file is ~213MB
    either way -- long past the point anyone reads it -- and `sort_keys` alone
    cost 4.3s of the ~10s each erasure spent rewriting it, for ordering that
    was already redundant: the dict is built in a deterministic order and
    `pop` preserves the order of what remains, so successive writes are
    byte-stable without paying to re-sort 1.6M keys per erasure.

    Nothing hashes this file -- the manifest commits to the Merkle root over
    retained subject_refs, not to the routing table -- so its formatting is
    not load-bearing for any proof.
    """
    os.makedirs(SHARD_DIR, exist_ok=True)
    with open(os.path.join(SHARD_DIR, "routing.json"), "w") as f:
        json.dump(routing, f, separators=(",", ":"))


def load_routing() -> dict:
    with open(os.path.join(SHARD_DIR, "routing.json")) as f:
        return json.load(f)


def fit_preprocessor(records: dict) -> ShardPreprocessor:
    """Fit on slice 0 only. See engine/preprocessing.ShardPreprocessor."""
    return ShardPreprocessor.fit(records, np.flatnonzero(records["slice_idx"] == 0))


def train_shard(shard: int, records: dict | None = None, from_slice: int = 0,
                resume_state: dict | None = None, seed: int = SEED,
                checkpoint_dir: str | None = None) -> dict:
    """Train slices [from_slice, NUM_SLICES), checkpointing after each.

    Returns {slice_idx: state_dict_digest}. `from_slice > 0` requires
    `resume_state` -- the checkpoint taken after slice from_slice-1.

    `checkpoint_dir` overrides where checkpoints land. The reproducibility
    spot-check (worker/jobs.py) passes a temporary directory: it re-runs a
    rebuild that already happened, and writing to the real paths would
    overwrite the promoted checkpoint -- harmlessly with identical bytes when
    the check passes, and with DIVERGENT weights exactly when it fails, which
    is the moment you least want the next rebuild resuming from a file that
    matches no recorded hash.
    """
    out_dir = checkpoint_dir or CHECKPOINT_DIR
    enforce_determinism(seed)
    if records is None:
        records = load_shard(shard)
    preproc = fit_preprocessor(records)

    model = build_model()
    if from_slice > 0:
        if resume_state is None:
            raise ValueError(f"resuming at slice {from_slice} needs a checkpoint")
        model.load_state_dict(resume_state)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"shard{shard}_preproc.json"), "w") as f:
        f.write(preproc.to_json())

    digests = {}
    for slice_idx in range(from_slice, NUM_SLICES):
        # Cumulative, not just this slice: SISA trains on slices 0..i at step i.
        rows = np.flatnonzero(records["slice_idx"] <= slice_idx)
        x = torch.from_numpy(preproc.transform(records, rows))
        y = torch.from_numpy(records["isFraud"][rows].astype(np.float32)).unsqueeze(1)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed + slice_idx),
        )
        for _ in range(EPOCHS_PER_SLICE):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        torch.save(model.state_dict(), checkpoint_path(shard, slice_idx, out_dir))
        digests[slice_idx] = state_dict_digest(model.state_dict())
    return digests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="partition data into shards first")
    parser.add_argument("--subjects", type=int, default=800)
    args = parser.parse_args()

    if args.build:
        routing = build(n_subjects=args.subjects)
        print(f"built {NUM_SHARDS} shards, {len(routing)} subjects -> {SHARD_DIR}/")

    for k in range(NUM_SHARDS):
        digests = train_shard(k)
        print(f"shard {k}: " + " ".join(f"s{i}={d[:8]}" for i, d in sorted(digests.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
