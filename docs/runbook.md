# Incident runbook

Three alerts, each with what fired, what it means, and what to do. Written
before the first page rather than after it, per Phase 7.

All commands assume `DATABASE_URL` is set and you are at the repo root.

---

## 1. Determinism drift

**Fires when:** `reproducibility_checks.matched = false`. A sampled erasure was
re-run and produced different weights than its manifest claims.

```sql
SELECT erasure_id, code_digest, expected_weights, observed_weights, checked_at
FROM reproducibility_checks WHERE NOT matched ORDER BY checked_at DESC;
```

### What it actually means

Every manifest is auditable because a rebuild is a pure function of its
inputs — an auditor re-runs it and compares digests. A drift means that
property has broken, so **every manifest issued under this `code_digest`
becomes unverifiable by re-run**, not just the one that failed. The erasures
themselves probably still happened; what is lost is the ability to prove it
this way.

This is the most serious alert here. It does not mean data was retained; it
means the evidence mechanism stopped working.

### Do

1. **Stop promoting.** Scale the worker to zero (`docker compose stop worker`).
   Further rebuilds under a drifting build add unverifiable manifests.
2. Identify what changed. Determinism is scoped to a `code_digest` (ADR 0003),
   so a differing digest between the manifest and the check is an expected,
   benign explanation — confirm first:
   ```sql
   SELECT r.code_digest AS check_digest, m.code_digest AS manifest_digest
   FROM reproducibility_checks r JOIN erasure_manifests m USING (erasure_id)
   WHERE NOT r.matched;
   ```
   Digests differ → not drift, a version skew. The check ran on a newer build
   than the manifest. Re-run on the matching image before escalating.
3. Digests match → real drift. Reproduce in isolation:
   ```bash
   docker compose build --no-cache determinism
   docker compose run --rm determinism python scripts/spot_check_determinism.py
   ```
   The failure message names the usual causes: thread count not pinned,
   `PYTHONHASHSEED` unset, a DataLoader worker without `worker_init_fn`, a
   floated dependency.
4. Check the pin has not silently moved:
   ```bash
   grep '^FROM' Dockerfile     # must be @sha256:..., never a bare tag
   grep -c '==' requirements.txt
   ```
5. Once fixed, the affected manifests need re-issuing under the corrected
   `code_digest` — the old ones cannot be made verifiable retroactively. This
   is a disclosure decision, not just an engineering one.

### Do not

Do not clear the failed rows to silence the alert. `reproducibility_checks` is
the audit trail; a period with no rows is indistinguishable from a period with
no drift, which is exactly the ambiguity Phase 4 refused to introduce by
writing fabricated passes.

---

## 2. Queue backlog exceeding SLA

**Fires when:** a queued job's `sla_deadline` is close or past.

```sql
SELECT erasure_id, shard, status, attempts, last_error,
       round(extract(epoch FROM sla_deadline - now())/3600) AS hours_left
FROM erasure_jobs WHERE status IN ('queued','processing')
ORDER BY sla_deadline LIMIT 20;
```

The dashboard shows the same thing with red/amber urgency banding.

### Triage by what the rows say

**`status = 'processing'`, lease expired.** A worker died mid-rebuild. The
reaper requeues these automatically (`worker/queue.py::reap_expired_leases`)
on the next poll — confirm a worker is actually running before intervening:

```bash
docker compose ps worker && docker compose logs worker --tail 50
```

**`status = 'failed'` with `last_error` mentioning "bookkeeping failed".** The
rebuild succeeded on disk but the database write did not. **Do not retry** —
the subject is already purged and gone from `routing.json`, so a retry raises a
confusing `KeyError` that hides the real, already-resolved cause (ADR 0006).
Reconcile manually: regenerate the manifest for the completed rebuild and mark
the job done.

**`status = 'failed'`, other errors.** Read `last_error`. These fail before any
offline mutation, so they are safe to requeue once the cause is fixed:

```sql
UPDATE erasure_jobs SET status='queued', last_error=NULL, leased_by=NULL,
       lease_expires_at=NULL
WHERE erasure_id = '...';
```

**Genuinely queued and not draining.** Throughput is set by rebuild duration,
not by the queue (`scripts/load_test.py`): at 20 jobs discharged per rebuild
pass, a 5-minute rebuild sustains ~5,760 erasures/day and a 15-minute rebuild
~1,920/day, against an assumed volume of tens to low hundreds. If the backlog
is real at that volume, rebuilds are slower than assumed or the worker is not
running — check the worker before assuming the queue is the problem.

Adding worker replicas is safe: `FOR UPDATE SKIP LOCKED` means two workers
never claim the same job. It buys nothing if all pending jobs are on one shard,
since they batch into a single rebuild anyway.

---

## 3. Minority-class regression

**Fires when:** `eval_results.auc` drops materially after a promotion.

```sql
SELECT model_version, auc, auc - lag(auc) OVER (ORDER BY computed_at) AS delta,
       computed_at
FROM eval_results ORDER BY computed_at DESC LIMIT 10;
```

### The rule: alert and proceed. Never block.

SISA concentrates accuracy loss on the minority class — fraud — and an erasure
that removes fraud examples from a shard will legitimately degrade it. **The
erasure is a legal obligation; the accuracy is not.** A gate that blocked
promotion on a metric regression would block a required deletion, which is the
wrong failure. Log, page, proceed.

### Do

1. Confirm the eval set did not move. `EVAL_SEED` is frozen at `999983`
   (`data/eval_set.py`); if that changed, the delta is measuring a different
   corpus, not a worse model. Compare `n_eval` across rows.
2. Attribute the drop to a shard. `model_versions.shard_checkpoints` shows
   which shard's hash changed between the two versions — only one moves per
   rebuild.
3. Check whether that shard lost a disproportionate share of its fraud rows.
   A shard that has absorbed many erasures has less data and less signal; that
   is the documented cost of the design, not a bug.
4. If degradation accumulates past what the business accepts, the response is
   a re-shard and full retrain — a planned operation, not an incident one, and
   one that invalidates existing rollback points (see the dynamic-resharding
   entry in `roadmap-assessment.md`).

---

## Appendix: what none of these alerts mean

None of the three imply a subject's data was retained. The erasure path is
synchronous with the rebuild and its result is recorded in
`erasure_manifests`; these alerts are about the evidence, the throughput, and
the accuracy cost respectively. If you need to answer "was this specific
subject actually erased", that is the certificate, not an alert:

```bash
curl -s -X POST localhost:8000/v1/erasure/attest \
  -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"subject_id": "..."}' > cert.json
python -m verify.verifier_cli cert.json
```

The verifier prints what the certificate does and does not prove on every
successful run.
