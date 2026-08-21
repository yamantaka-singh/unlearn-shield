"""Worker entrypoint. Separate deployment from the gateway, no ingress.

    python -m worker.main [--once]

Each iteration: reap expired leases, claim a batch, process it, commit,
sleep. Claim and process are separate transactions (see worker/queue.py) --
a rebuild never runs inside the transaction that claimed the jobs.
"""

import argparse
import os
import socket
import time

from db.conn import connect
from worker.jobs import process_claimed
from worker.queue import claim_batch, reap_expired_leases

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
POLL_INTERVAL_SECONDS = 10


def run_once() -> int:
    """One claim-process cycle. Returns the number of jobs processed."""
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            reap_expired_leases(cur)
            jobs = claim_batch(cur, WORKER_ID)
        if not jobs:
            return 0
        with conn, conn.cursor() as cur:
            process_claimed(cur, jobs)
        return len(jobs)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="process one batch and exit")
    args = parser.parse_args()

    if args.once:
        n = run_once()
        print(f"processed {n} job(s)")
        return 0

    while True:
        n = run_once()
        if n:
            print(f"processed {n} job(s)")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
