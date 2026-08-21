"""Process one claimed batch: rebuild, write the manifest, promote, spot-check.

Everything in here runs OUTSIDE the row lock that claimed the jobs -- the
worker already committed the lease in worker/queue.py before this is called, so
a multi-minute rebuild never holds a Postgres transaction open.
"""

import hashlib
import hmac
import os
from itertools import groupby

import numpy as np
import psycopg2.extras

from config.settings import AUDIT_KEY, CODE_DIGEST, SPOT_CHECK_RATE
from data.eval_set import auc as compute_auc
from data.eval_set import load as load_eval_set
from engine.rebuild import rebuild_batch_by_ref
from engine.train import checkpoint_path
from inference.batched_ensemble import load_ensemble


def _should_spot_check(erasure_id: str) -> bool:
    """HMAC(erasure_id, audit_key), not a coin flip the worker controls -- so an
    operator cannot steer the sample away from jobs it would rather not have
    re-run, and an auditor holding audit_key can recompute the same sample and
    confirm it wasn't gamed. Threshold on the digest's leading bits, uniform in
    [0, 1) the same way any HMAC-as-PRF construction is.
    """
    digest = hmac.new(AUDIT_KEY, erasure_id.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < SPOT_CHECK_RATE


def _promote(cur, shard: int, result_weights: str) -> None:
    """Copy the finished checkpoint to a content-addressed path, write the
    checkpoints row, then a model_versions row carrying forward every other
    shard's current hash unchanged -- a rebuild of shard k must not silently
    drop shards it didn't touch from the promoted set.

    The copy matters: engine/train.py writes each slice to a FIXED path keyed
    only by (shard, slice_idx), because within one rebuild it needs a stable
    name to resume from. But that means the next rebuild of this shard
    overwrites those bytes -- so a checkpoints row from an earlier rebuild
    would end up pointing at a file that no longer contains what its recorded
    hash says it does. Copying into checkpoints/cas/{hash}.pt at promotion time
    is what makes DB history actually retrievable rather than just recorded.
    """
    import shutil
    from engine.train import CHECKPOINT_DIR, NUM_SLICES

    source = checkpoint_path(shard, NUM_SLICES - 1)
    cas_dir = os.path.join(CHECKPOINT_DIR, "cas")
    os.makedirs(cas_dir, exist_ok=True)
    cas_path = os.path.join(cas_dir, f"{result_weights}.pt")
    if not os.path.exists(cas_path):
        shutil.copyfile(source, cas_path)

    cur.execute("""
        INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
        VALUES (%s, %s, %s, %s, %s) ON CONFLICT (checkpoint_hash) DO NOTHING
    """, (result_weights, shard, NUM_SLICES - 1, cas_path, CODE_DIGEST))

    cur.execute("SELECT model_version, shard_checkpoints, eval_set_version FROM model_versions "
               "ORDER BY promoted_at DESC LIMIT 1")
    row = cur.fetchone()
    shard_checkpoints = dict(row[1]) if row else {}
    eval_set_version = row[2] if row else "v0"
    shard_checkpoints[str(shard)] = result_weights

    new_version = f"v-{result_weights[:12]}"
    cur.execute("""
        INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
        VALUES (%s, %s, %s) ON CONFLICT (model_version) DO NOTHING
    """, (new_version, psycopg2.extras.Json(shard_checkpoints), eval_set_version))
    record_eval(cur, new_version, shard_checkpoints)


def record_eval(cur, model_version: str, shard_checkpoints: dict) -> float:
    """Score the just-promoted ensemble against the frozen eval corpus and
    record the AUC. Used at every promotion (here) and once at bootstrap
    (scripts/load_routing.py), so the dashboard's delta chart has a baseline
    to compare the first real rebuild against.

    Real, not fabricated: it runs the actual promoted checkpoints through
    inference.batched_ensemble against data/eval_set.py's frozen corpus. A
    Phase 6 "accuracy chart" backed by invented numbers would be exactly the
    kind of thing this project exists to catch someone else doing.
    """
    from engine.train import CHECKPOINT_DIR

    cur.execute("SELECT shard, file_path FROM checkpoints WHERE checkpoint_hash = ANY(%s)",
               (list(shard_checkpoints.values()),))
    shard_paths = {str(shard): path for shard, path in cur.fetchall()}
    preproc_paths = {s: os.path.join(CHECKPOINT_DIR, f"shard{s}_preproc.json") for s in shard_paths}

    records = load_eval_set()
    rows = np.arange(len(records["step"]))
    ensemble = load_ensemble(shard_paths, preproc_paths)
    probs = ensemble.predict_proba(records, rows)
    score = compute_auc(records["isFraud"], probs)

    cur.execute("""
        INSERT INTO eval_results (model_version, auc, n_eval)
        VALUES (%s, %s, %s) ON CONFLICT (model_version) DO NOTHING
    """, (model_version, score, len(rows)))
    return score


# NOT IMPLEMENTED in Phase 4: re-running a sampled job and comparing weight
# digests needs the pre-purge shard state, which nothing currently snapshots --
# engine/rebuild.py purges in place. Writing a `reproducibility_checks` row
# without an actual second rebuild would fabricate a "matched" result for a
# check that never ran, which is worse than not having the check at all: it
# corrupts the one table Phase 7's drift alert reads. `_should_spot_check`
# below is real and tested (the HMAC-gated sample selection Phase 4c asks
# for); wire the re-run itself once shard snapshotting exists, and only then
# start writing to reproducibility_checks.


def process_claimed(cur, jobs: list[dict]) -> None:
    """`jobs` is what worker/queue.py::claim_batch returned -- already leased
    and committed. Groups by shard so subjects sharing a shard retrain once."""
    jobs = sorted(jobs, key=lambda j: j["shard"])
    for shard, group in groupby(jobs, key=lambda j: j["shard"]):
        group = list(group)
        refs = [j["subject_ref"] for j in group]
        try:
            result = rebuild_batch_by_ref(refs)
        except Exception as exc:
            # Nothing offline has happened yet at this point -- purge/retrain
            # is the first thing rebuild_batch_by_ref does -- so 'failed' here
            # is unambiguous and the job is safe to retry once the cause is
            # fixed (routing.json still has the subject).
            cur.execute("""
                UPDATE erasure_jobs SET status = 'failed', last_error = %s,
                    leased_by = NULL, lease_expires_at = NULL
                WHERE erasure_id = ANY(%s::uuid[])
            """, (str(exc)[:2000], [j["erasure_id"] for j in group]))
            continue

        try:
            _promote(cur, shard, result["result_weights"])
            for job in group:
                manifest = result["manifests"][job["subject_ref"]]
                cur.execute("""
                    INSERT INTO erasure_manifests
                        (erasure_id, shard, resumed_from, dataset_root, absence_proof,
                         code_digest, config_digest, result_weights, model_version,
                         signature, manifest_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (job["erasure_id"], manifest["shard"], manifest["resumed_from"],
                      manifest["dataset_root"], psycopg2.extras.Json(manifest["absence_proof"]),
                      manifest["code_digest"], manifest["config_digest"],
                      manifest["result_weights"], manifest["model_version"],
                      manifest["signature"], psycopg2.extras.Json(manifest)))

                cur.execute("""
                    UPDATE erasure_jobs SET status = 'done', completed_at = now(),
                        leased_by = NULL, lease_expires_at = NULL
                    WHERE erasure_id = %s
                """, (job["erasure_id"],))
        except Exception as exc:
            # Unlike the branch above: by now rebuild_batch_by_ref has ALREADY
            # purged these subjects from the shard file, retrained it, and
            # dropped them from routing.json -- on disk, the erasure genuinely
            # happened. A bookkeeping failure here (this exact bug: a schema
            # constraint that made the second rebuild of a shard impossible to
            # promote) must not leave the job silently stuck at 'processing'
            # for a full lease period and then fail again identically on
            # retry, because retrying would call rebuild_batch_by_ref for a
            # subject routing.json no longer lists -- a confusing KeyError
            # that hides the real, already-fixed problem. Fail loud and
            # immediately instead, and say plainly that the data-side work is
            # done and only the record of it is missing.
            cur.execute("""
                UPDATE erasure_jobs SET status = 'failed',
                    last_error = %s, leased_by = NULL, lease_expires_at = NULL
                WHERE erasure_id = ANY(%s::uuid[])
            """, (f"rebuild completed but bookkeeping failed ({exc}); "
                  f"subject(s) already purged from shard {shard} -- "
                  f"do not blindly retry, reconcile manually"[:2000],
                  [j["erasure_id"] for j in group]))
            continue

        for job in group:
            if _should_spot_check(job["erasure_id"]):
                pass  # selected for spot-check; re-run not wired yet, see note above
