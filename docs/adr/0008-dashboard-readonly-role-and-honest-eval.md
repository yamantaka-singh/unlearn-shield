# 0008 — Dashboard reads through a role that cannot write, and the accuracy chart is a real number

**Status:** accepted, 2026-08-21

## Context

The plan's Phase 6 asks for four things: queue depth per shard, an SLA
countdown per pending job, a manifest viewer, and an accuracy-delta chart per
rebuild — plus a "force rebuild now" button that goes through the same path
the API uses, not a direct DB write.

Two of those five items had a real decision behind them worth recording.

## The dashboard cannot write to Postgres — enforced by the database, not the code

The plan says the dashboard reads through "a read-only role" and writes only
through the gateway's own route. It would have been easy to satisfy that by
convention — just don't write raw SQL in `dashboard/app.py` — and easy to
violate it the same way: one future edit adds a `cur.execute("UPDATE …")`
because it's faster than an HTTP round trip, and nothing stops it.

`db/schema.sql` creates `unlearnshield_readonly`, granted `SELECT` only, and
`dashboard/app.py` connects exclusively through it
(`db.conn.connect_readonly`). Verified directly, not assumed: an `INSERT`
attempted through that role raises `psycopg2.errors.InsufficientPrivilege`.
The boundary is real regardless of what the application code does.

The dashboard's one write path — "force rebuild now" — is a `POST` to the
gateway's own `/v1/erasure`, using stdlib `urllib.request` rather than adding
a `requests` dependency for one call. It goes through `Idempotency-Key` and
subject lookup exactly like any other caller, because a dashboard-specific
insert path would be a second, less-tested way to create the same row.

## The accuracy chart is a real number, not a plausible one

"Accuracy-delta chart per rebuild" has no data behind it in this codebase —
nothing computes or stores an accuracy figure anywhere. The two honest options
were: skip the chart and say why, or build the smallest real thing that
produces one.

Built the smaller of the two problems the original plan describes. That plan's
eval-set versioning assumes the eval set contains real subjects, so an erasure
has to be reflected in it too — a second purge-state to track, on top of the
one this project already has for shards. `data/eval_set.py` sidesteps that:
a frozen, synthetic, non-subject corpus (`EVAL_SEED = 999_983`, fixed forever)
that is never inserted into `subject_shard_map` and so can never be the target
of an erasure request in the first place.

AUC is computed by hand (`data/eval_set.py::auc`) rather than adding an
sklearn dependency for one function — it's a rank statistic, cross-checked
against an O(n²) brute-force pairwise count over 200 random trials to 1e-9,
tested at the degenerate cases (single-class inputs return 0.5, not `NaN`,
which would otherwise poison a mean or silently break a chart), and tied
scores are rank-averaged so array order can't shift the result.

`worker/jobs.py::record_eval` scores the just-promoted ensemble at every
promotion, using the same `inference.batched_ensemble` code path that serves
real traffic — not a separate, only-tested-here scoring routine. Verified
against an independent recomputation built from scratch in
`test_recorded_auc_matches_an_independent_recomputation`, and against a real
worker run: two real promotions produced `0.5151` and `0.5158`, not `1.0`,
not fabricated round numbers.

### What this does not solve

If the eval set ever needs to contain real subject behaviour rather than
synthetic rows, the harder version of this problem — eval-set purge-state,
tracked the same way shard state is — is still unsolved and still open. This
is the easier case, chosen because it is honest and buildable now, not
presented as having solved the harder one.

## Consequences

- `eval_results` is append-only history (`ON CONFLICT DO NOTHING`), same
  pattern as `checkpoints` — every promotion's score stays comparable to every
  other, nothing is overwritten.
- Bootstrapping (`scripts/load_routing.py`) also calls `record_eval`, so the
  chart has a baseline point before any real rebuild happens.
- A model that ships without ever promoting through `worker/jobs.py::_promote`
  (for instance, a hand-built model_version row in a test) has no eval score
  and the dashboard shows "no promotions recorded yet" rather than a fabricated
  zero.

---

## Amendment, 2026-08-21: Streamlit replaced by a served page

The security boundary above is unchanged; only the rendering layer moved.

Streamlit got a working console quickly, but an ops console is almost entirely
dense tables, status chips, and countdowns — the things it gives least control
over. Two concrete costs showed up:

- **Contrast was fought, not designed.** The SLA table's urgency highlight
  shipped as light-pink row backgrounds while Streamlit's dark theme kept cell
  text near-white, leaving the most urgent rows the least readable. The fix was
  a workaround (setting an explicit `color` alongside every `background-color`)
  rather than a design decision.
- **Thirteen streamlit-family packages** (altair, pydeck, pyarrow, tornado,
  gitpython, …) in an image whose other dependencies are all deliberate and
  pinned for determinism.

`dashboard/app.py` is now a small FastAPI app serving JSON plus a hand-written
page (`dashboard/static/`). No React, no bundler, no build step — one page with
five panels does not need a toolchain, and adding one would put a build
artifact into review.

What the change bought:

- Colour is now spent only on meaning: a 2px status rail on stat cards, a left
  rail on urgent rows, coloured *numbers* rather than tinted surfaces. The rule
  learned from the contrast bug is written into `app.css`: colour the signal,
  not the surface.
- Live re-verification became a real endpoint (`GET /api/certificates/{id}`)
  that runs `verify_certificate` per request, so it is testable with a plain
  `TestClient` instead of Streamlit's `AppTest`.
- All API output is escaped before reaching the DOM. `subject_ref` and
  `last_error` are attacker-influenced in principle, and an ops console that
  renders them as markup would be a stored-XSS hole in the tool used to
  investigate incidents.

`requirements.txt` loses `streamlit`; `docker-compose.yml`'s dashboard service
runs `uvicorn dashboard.app:app`. The read-only role, the gateway-proxied write
path, and the honest-AUC reasoning above all carry over unchanged.
