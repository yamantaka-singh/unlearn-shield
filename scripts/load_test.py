"""Find the worker's throughput ceiling on the batch-rebuild path.

    python -m scripts.load_test --jobs 60

Assumption 2 of the plan puts real volume at tens to low hundreds of erasures
per day. This exists to answer "at what volume does the Postgres queue stop
being sufficient" with a measurement instead of a guess -- the plan's Phase 7
asks for a known breaking point rather than discovering one in production.

Not a pytest: it mutates a real corpus destructively and takes minutes. Run it
against a scratch database.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

from db.conn import connect
from gateway.idempotency import insert_or_get
from worker.jobs import process_claimed
from worker.queue import claim_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    conn = connect()
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT subject_ref, shard FROM subject_shard_map
            ORDER BY subject_ref LIMIT %s
        """, (args.jobs,))
        subjects = cur.fetchall()
    if len(subjects) < args.jobs:
        print(f"only {len(subjects)} subjects routed; run engine.train --build "
              f"with more subjects first")
        return 1

    deadline = datetime.now(timezone.utc) + timedelta(hours=720)
    enqueue_start = time.perf_counter()
    with conn, conn.cursor() as cur:
        for i, (ref, shard) in enumerate(subjects):
            insert_or_get(cur, subject_ref=ref, reason="fraud_excision", shard=shard,
                          idempotency_key=f"load-{i}-{enqueue_start}",
                          sla_deadline=deadline, requested_by="load-test")
    enqueue_ms = (time.perf_counter() - enqueue_start) * 1000
    print(f"enqueued {len(subjects)} jobs in {enqueue_ms:.0f} ms "
          f"({enqueue_ms/len(subjects):.1f} ms/job)")

    processed, passes, drain_start = 0, 0, time.perf_counter()
    batch_times = []
    while processed < len(subjects):
        with conn, conn.cursor() as cur:
            claimed = claim_batch(cur, "load-test", limit=args.batch)
        if not claimed:
            break
        t = time.perf_counter()
        with conn, conn.cursor() as cur:
            process_claimed(cur, claimed)
        batch_times.append(time.perf_counter() - t)
        processed += len(claimed)
        passes += 1
        print(f"  pass {passes}: {len(claimed)} jobs in {batch_times[-1]:.1f}s")

    drain = time.perf_counter() - drain_start
    conn.close()

    if not processed:
        print("nothing processed")
        return 1

    per_job = drain / processed
    per_pass = sum(batch_times) / len(batch_times)
    jobs_per_pass = processed / passes
    print(f"\ndrained {processed} jobs in {drain:.1f}s over {passes} worker passes")
    print(f"  {jobs_per_pass:.1f} jobs discharged per rebuild pass "
          f"(this is what Phase 4c's batching buys)")
    print(f"  {per_pass:.2f}s per pass, {per_job*1000:.0f} ms/job amortised")

    # The raw rate is dominated by THIS model's rebuild time, not by the queue,
    # and quoting it as a system ceiling would be meaningless -- a production
    # shard trains for minutes, not milliseconds. What transfers is the ratio:
    # the queue drains at (jobs per pass) / (rebuild duration), so the ceiling
    # is set by how long a real rebuild takes.
    print(f"\nThe queue is not the bottleneck here -- enqueue costs "
          f"{enqueue_ms/len(subjects):.1f} ms/job against a {per_pass:.2f}s rebuild. "
          f"Extrapolating to real rebuild durations at {jobs_per_pass:.0f} jobs/pass:")
    for minutes in (1, 5, 15):
        per_hour = (60 / minutes) * jobs_per_pass
        print(f"  {minutes:2d} min/rebuild -> {per_hour:6.0f} erasures/hour, "
              f"{per_hour*24:7.0f}/day")
    print("\nAssumption 2 puts real volume at tens to low hundreds per day, so "
          "the Postgres queue has substantial headroom at every row above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
