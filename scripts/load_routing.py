"""Load engine/train.py's routing.json into subject_shard_map, and bootstrap
model_versions with the freshly built baseline.

    python -m scripts.load_routing

Run once after `python -m engine.train --build`. engine/ writes JSON rather
than touching Postgres directly because it is offline and network-free by
design (see engine/train.py); the DB is Phase 4's, and this script is the seam.
"""

import os
import shutil
from hashlib import sha256

from psycopg2.extras import Json, execute_values

from config.settings import CHECKPOINT_DIR, CODE_DIGEST, NUM_SHARDS, NUM_SLICES
from db.conn import connect
from engine.train import checkpoint_path, load_routing
from worker.jobs import record_eval


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return sha256(f.read()).hexdigest()


def _cas_copy(source: str, digest: str) -> str:
    """Same reasoning as worker/jobs.py::_promote: engine/train.py writes each
    slice to a fixed (shard, slice_idx) path that the next rebuild overwrites,
    so the DB's file_path must point at a content-addressed copy or it goes
    stale the moment this shard is rebuilt for the first time."""
    cas_dir = os.path.join(CHECKPOINT_DIR, "cas")
    os.makedirs(cas_dir, exist_ok=True)
    cas_path = os.path.join(cas_dir, f"{digest}.pt")
    if not os.path.exists(cas_path):
        shutil.copyfile(source, cas_path)
    return cas_path


def main() -> int:
    routing = load_routing()
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO subject_shard_map (subject_ref, tenant_id, shard, min_slice_idx, record_count)
                VALUES %s
                ON CONFLICT (subject_ref) DO UPDATE SET
                    shard = EXCLUDED.shard, min_slice_idx = EXCLUDED.min_slice_idx,
                    record_count = EXCLUDED.record_count
            """, [(ref, e["tenant_id"], e["shard"], e["min_slice_idx"], e["record_count"])
                  for ref, e in routing.items()])

            shard_checkpoints = {}
            for shard in range(NUM_SHARDS):
                path = checkpoint_path(shard, NUM_SLICES - 1)
                digest = _file_hash(path)
                cas_path = _cas_copy(path, digest)
                cur.execute("""
                    INSERT INTO checkpoints (checkpoint_hash, shard, slice_idx, file_path, code_digest)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (checkpoint_hash) DO NOTHING
                """, (digest, shard, NUM_SLICES - 1, cas_path, CODE_DIGEST))
                shard_checkpoints[str(shard)] = digest

            cur.execute("""
                INSERT INTO model_versions (model_version, shard_checkpoints, eval_set_version)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_version) DO NOTHING
            """, ("v0-baseline", Json(shard_checkpoints), "v0"))
            baseline_auc = record_eval(cur, "v0-baseline", shard_checkpoints)

        print(f"loaded {len(routing)} subjects, baseline model_version v0-baseline, "
              f"eval AUC {baseline_auc:.4f}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
