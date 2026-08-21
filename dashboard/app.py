"""Internal ops tool. Not the product, not customer-facing.

Reads Postgres through the role that cannot write (db/schema.sql). The one
write path -- "queue a rebuild now" -- goes through the gateway's own
POST /v1/erasure exactly as any other caller would, over HTTP, using stdlib
urllib rather than adding a requests dependency for one call. Nothing here
touches the database with anything but SELECT.

    streamlit run dashboard/app.py
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DASHBOARD_GATEWAY_TOKEN, DASHBOARD_GATEWAY_URL
from db.conn import connect_readonly

st.set_page_config(page_title="UnlearnShield Ops", layout="wide")

# A fixed status->color map, not a library, and colors chosen for 4.5:1+
# contrast against white -- pale, low-opacity chips are unreadable in light
# mode, one of the concrete complaints a UI review would raise.
STATUS_COLOR = {"queued": "#92400e", "processing": "#1d4ed8", "done": "#166534", "failed": "#991b1b"}
STATUS_BG = {"queued": "#fef3c7", "processing": "#dbeafe", "done": "#dcfce7", "failed": "#fee2e2"}


def status_pill(status: str) -> str:
    color, bg = STATUS_COLOR.get(status, "#374151"), STATUS_BG.get(status, "#f3f4f6")
    return (f'<span style="background:{bg};color:{color};padding:2px 10px;'
           f'border-radius:999px;font-size:0.85em;font-weight:600">{status}</span>')


@st.cache_resource
def _conn():
    return connect_readonly()


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    # A fresh cursor per call on a long-lived connection; st.cache_resource
    # keeps the connection itself from being reopened on every Streamlit rerun,
    # which happens on almost every widget interaction.
    return pd.read_sql(sql, _conn(), params=params)


st.title("UnlearnShield — Ops")
st.caption("Internal visibility only. Every write goes through the gateway's own API.")

if st.button("Refresh"):
    st.cache_resource.clear()
    st.rerun()

st.divider()

# ---------------------------------------------------------------- queue health
st.header("Queue health")

jobs = query("SELECT shard, status, count(*) AS n FROM erasure_jobs GROUP BY shard, status")
if jobs.empty:
    st.info("No erasure jobs yet.")
else:
    pivot = jobs.pivot(index="shard", columns="status", values="n").fillna(0)
    for col in ("queued", "processing", "done", "failed"):
        if col not in pivot.columns:
            pivot[col] = 0
    st.bar_chart(pivot[["queued", "processing", "done", "failed"]])

st.divider()

# ------------------------------------------------------------ pending erasures
st.header("Pending erasures")

pending = query("""
    SELECT erasure_id, subject_ref, shard, reason, status, sla_deadline, created_at
    FROM erasure_jobs WHERE status IN ('queued', 'processing')
    ORDER BY sla_deadline ASC
""")

if pending.empty:
    st.success("Nothing pending.")
else:
    now = pd.Timestamp.now(tz=timezone.utc)
    pending["sla_deadline"] = pd.to_datetime(pending["sla_deadline"], utc=True)
    remaining = pending["sla_deadline"] - now
    pending["hours_remaining"] = (remaining.dt.total_seconds() / 3600).round(1)
    pending["subject_ref"] = pending["subject_ref"].str[:16] + "…"

    def urgency_row(row):
        # Text color set explicitly alongside the background, not left to
        # inherit: this highlight is always a light chip, but Streamlit's
        # dark theme keeps default cell text near-white, which put light
        # gray text on light pink -- unreadable. Caught by actually looking
        # at the rendered page, not by reading the code.
        if row["hours_remaining"] < 24:
            return ["background-color: #fee2e2; color: #991b1b"] * len(row)
        if row["hours_remaining"] < 24 * 7:
            return ["background-color: #fef3c7; color: #92400e"] * len(row)
        return [""] * len(row)

    display = pending[["erasure_id", "subject_ref", "shard", "reason", "status", "hours_remaining"]]
    styled = display.style.apply(urgency_row, axis=1).format({"hours_remaining": "{:.1f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Red: under 24h to SLA deadline. Amber: under 7 days.")

st.divider()

# --------------------------------------------------------------- eval accuracy
st.header("Model accuracy (frozen eval set)")

evals = query("""
    SELECT model_version, auc, n_eval, computed_at FROM eval_results ORDER BY computed_at
""")
if evals.empty:
    st.info("No promotions recorded yet. Run scripts.load_routing after a build.")
else:
    evals["delta"] = evals["auc"].diff()
    st.line_chart(evals.set_index("computed_at")["auc"])
    latest = evals.iloc[-1]
    delta = latest["delta"] if pd.notna(latest["delta"]) else 0.0
    st.metric(f"Current: {latest['model_version']}", f"{latest['auc']:.4f} AUC",
             f"{delta:+.4f} vs previous promotion")
    st.caption(f"Scored against a frozen {int(latest['n_eval'])}-row synthetic eval set "
              f"with no subject_ref -- it cannot be the target of an erasure, so it needs "
              f"no purge-state of its own.")

st.divider()

# ------------------------------------------------------------- certificate viewer
st.header("Certificate viewer")

completed = query("""
    SELECT m.erasure_id, m.model_version, m.created_at
    FROM erasure_manifests m ORDER BY m.created_at DESC LIMIT 200
""")

if completed.empty:
    st.info("No certificates yet.")
else:
    completed["erasure_id"] = completed["erasure_id"].astype(str)
    options = completed["erasure_id"].tolist()
    labels = {row["erasure_id"]: f"{row['erasure_id']} — {row['model_version']} — {row['created_at']}"
             for _, row in completed.iterrows()}
    chosen = st.selectbox("Erasure ID", options, format_func=lambda x: labels[x])

    manifest_row = query(
        "SELECT manifest_json FROM erasure_manifests WHERE erasure_id = %(id)s",
        params={"id": chosen})
    manifest = manifest_row.iloc[0]["manifest_json"]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.json(manifest)
    with col2:
        st.subheader("Live re-verification")
        # Genuinely re-runs verify/verifier_cli against this certificate right
        # now -- not a cached "it passed once" flag. That is the entire point
        # of shipping a certificate: anyone, including this dashboard, can
        # check it independently.
        from verify.sign import load_public_key
        from verify.verifier_cli import verify_certificate

        ok, findings = verify_certificate(dict(manifest), load_public_key())
        (st.success if ok else st.error)("VERIFIED" if ok else "REJECTED")
        for line in findings:
            st.text(line)

st.divider()

# ------------------------------------------------------------ force rebuild now
st.header("Force rebuild now")
st.caption("Goes through POST /v1/erasure on the gateway -- the same path any "
          "other caller uses. This page never writes to Postgres directly.")

with st.form("force_rebuild"):
    subject_id = st.text_input("Subject ID")
    reason = st.selectbox("Reason", ["consent_revocation", "fraud_excision"])
    submitted = st.form_submit_button("Queue erasure")

if submitted:
    if not subject_id.strip():
        st.warning("Subject ID is required.")
    else:
        body = json.dumps({"subject_id": subject_id, "reason": reason}).encode()
        req = urllib.request.Request(
            f"{DASHBOARD_GATEWAY_URL}/v1/erasure", data=body, method="POST",
            headers={"Authorization": f"Bearer {DASHBOARD_GATEWAY_TOKEN}",
                    "Idempotency-Key": f"dashboard-{subject_id}-{datetime.now(timezone.utc).timestamp()}",
                    "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                st.success(f"Queued: {resp.read().decode()}")
        except urllib.error.HTTPError as e:
            st.error(f"Gateway rejected the request ({e.code}): {e.read().decode()}")
        except urllib.error.URLError as e:
            st.error(f"Could not reach the gateway at {DASHBOARD_GATEWAY_URL}: {e.reason}")
