"""Load engine/train.py's routing.json into subject_shard_map, and bootstrap
model_versions with the freshly built baseline.

    python -m scripts.load_routing

Run once after `python -m engine.train --build`. engine/ writes JSON rather
than touching Postgres directly because it is offline and network-free by
design (see engine/train.py); the DB is Phase 4's, and this script is the seam.
"""

from hashlib import sha256

from psycopg2.extras import Json, execute_values

from config.settings import CODE_DIGEST, NUM_SHARDS, NUM_SLICES
from db.conn import connect
from engine import active
from engine.train import load_routing
from worker.jobs import record_eval


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return sha256(f.read()).hexdigest()


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
                # Whichever engine is active: its live model at the mutable
                # conventional path, hashed, then copied content-addressed so
                # the DB's file_path survives this shard's first rebuild.
                path, _ = active.live_model_path(shard)
                digest = _file_hash(path)
                cas_path = active.promote_artifact(shard, digest)
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
