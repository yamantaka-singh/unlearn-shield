/* Ops console front-end. No framework, no build step.
 *
 * One page, five panels, a handful of fetches. React plus a bundler would add
 * a toolchain to the repo and a build artifact to review, and would not make
 * any of this shorter. Plain DOM it is.
 */

const $ = (id) => document.getElementById(id);
const STATUSES = ["queued", "processing", "done", "failed"];

/* Every value from the API is treated as text, never as markup -- subject_ref
 * and last_error are attacker-influenced in principle, and an ops console that
 * renders them as HTML is a stored-XSS hole in the tool you use to investigate
 * incidents. */
const esc = (v) => String(v ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const short = (s, n = 12) => (s && s.length > n ? s.slice(0, n) + "…" : s ?? "");

async function get(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ render */

function renderStats(d) {
  const m = d.current_model;
  const hist = d.eval_history;
  let delta = null;
  if (hist.length >= 2) delta = hist[hist.length - 1].auc - hist[hist.length - 2].auc;

  const cards = STATUSES.map((s) => `
    <div class="stat ${s}">
      <div class="label">${s}</div>
      <div class="value">${d.counts[s]}</div>
    </div>`);

  cards.push(`
    <div class="stat accent">
      <div class="label">Subjects routed</div>
      <div class="value">${d.subjects_routed}</div>
    </div>`);

  if (m && m.auc != null) {
    const cls = delta == null ? "flat" : delta > 0 ? "up" : delta < 0 ? "down" : "flat";
    const sign = delta == null ? "" :
      `<span class="delta ${cls}">${delta >= 0 ? "+" : ""}${delta.toFixed(4)}</span>`;
    cards.push(`
      <div class="stat accent">
        <div class="label">Eval AUC</div>
        <div class="value">${m.auc.toFixed(4)}${sign}</div>
      </div>`);
  }

  $("stats").innerHTML = cards.join("");
  $("model-pill").textContent = m ? m.model_version : "no model promoted";

  $("drift-banner").innerHTML = d.drift_failures > 0 ? `
    <div class="banner">
      <strong>Determinism drift:</strong> ${d.drift_failures} reproducibility
      check(s) failed. Manifests under this code_digest cannot be independently
      re-verified until resolved — see docs/runbook.md.
    </div>` : "";
}

function renderShards(byShard) {
  const shards = new Map();
  for (const r of byShard) {
    if (!shards.has(r.shard)) shards.set(r.shard, {});
    shards.get(r.shard)[r.status] = r.n;
  }
  if (!shards.size) {
    $("shards").innerHTML = `<div class="empty">No jobs yet.</div>`;
    return;
  }

  const rows = [...shards.entries()].sort((a, b) => a[0] - b[0]).map(([shard, counts]) => {
    const total = STATUSES.reduce((t, s) => t + (counts[s] || 0), 0);
    const segs = STATUSES.filter((s) => counts[s])
      .map((s) => `<div class="seg ${s}" style="width:${(counts[s] / total) * 100}%"
                        title="${counts[s]} ${s}"></div>`).join("");
    return `<div class="shard-row">
              <span class="name">shard ${shard}</span>
              <span class="track">${segs}</span>
              <span class="total">${total}</span>
            </div>`;
  });

  $("shards").innerHTML = rows.join("") + `
    <div class="legend">${STATUSES.map((s) =>
      `<span><i class="seg ${s}"></i>${s}</span>`).join("")}</div>`;
}

function renderPending(jobs) {
  $("pending-count").textContent = jobs.length ? `${jobs.length}` : "";
  if (!jobs.length) {
    $("pending").innerHTML = `<div class="empty">Queue is clear.</div>`;
    return;
  }
  const body = jobs.map((j) => {
    const h = j.hours_remaining;
    const urg = h < 24 ? "urgent-high" : h < 168 ? "urgent-med" : "";
    const hcls = h < 24 ? "high" : h < 168 ? "med" : "";
    return `<tr class="${urg}">
      <td class="mono dim">${esc(short(j.erasure_id, 8))}</td>
      <td class="mono dim">${esc(short(j.subject_ref, 10))}</td>
      <td class="num">${j.shard}</td>
      <td class="dim">${esc(j.reason)}</td>
      <td><span class="chip ${esc(j.status)}">${esc(j.status)}</span></td>
      <td class="num hours ${hcls}">${h.toFixed(1)}h</td>
    </tr>`;
  }).join("");

  $("pending").innerHTML = `<table>
    <thead><tr><th>Erasure</th><th>Subject</th><th>Shard</th><th>Reason</th>
      <th>Status</th><th style="text-align:right">SLA left</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function renderFailed(jobs) {
  $("failed-panel").hidden = jobs.length === 0;
  if (!jobs.length) return;
  $("failed-count").textContent = `${jobs.length}`;
  $("failed").innerHTML = `<table>
    <thead><tr><th>Erasure</th><th>Shard</th><th>Attempts</th><th>Error</th></tr></thead>
    <tbody>${jobs.map((j) => `<tr>
      <td class="mono dim">${esc(short(j.erasure_id, 8))}</td>
      <td class="num">${j.shard}</td>
      <td class="num">${j.attempts}</td>
      <td class="dim" style="font-size:12px">${esc(j.last_error || "—")}</td>
    </tr>`).join("")}</tbody></table>`;
}

function renderAuc(hist) {
  if (!hist.length) {
    $("auc").innerHTML = `<div class="empty">No promotions recorded yet.</div>`;
    return;
  }
  if (hist.length === 1) {
    $("auc").innerHTML = `<div class="empty">
      One promotion so far (AUC ${hist[0].auc.toFixed(4)}). A trend needs a second.</div>`;
    return;
  }

  const W = 520, H = 60, PAD = 4;
  const vals = hist.map((h) => h.auc);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  // Guard a flat series: an identical min and max would divide by zero and
  // collapse every point onto one line.
  const span = hi - lo || 1;
  const x = (i) => PAD + (i / (hist.length - 1)) * (W - PAD * 2);
  const y = (v) => H - PAD - ((v - lo) / span) * (H - PAD * 2);

  const line = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const area = `${line}L${x(vals.length - 1).toFixed(1)},${H}L${x(0).toFixed(1)},${H}Z`;

  $("auc").innerHTML = `
    <svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         role="img" aria-label="Eval AUC across ${hist.length} promotions">
      <path class="area" d="${area}"></path>
      <path d="${line}"></path>
      <circle cx="${x(vals.length - 1).toFixed(1)}" cy="${y(vals[vals.length - 1]).toFixed(1)}" r="2.5"></circle>
    </svg>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-faint);margin-top:6px">
      <span class="mono">${lo.toFixed(4)}</span>
      <span>${hist.length} promotions · frozen ${hist[0].n_eval}-row eval set</span>
      <span class="mono">${hi.toFixed(4)}</span>
    </div>`;
}

/* Minimal JSON pretty-printer with class hooks -- avoids pulling a syntax
 * highlighter in for one <pre>. Escapes first, then colourises. */
function highlight(obj) {
  return esc(JSON.stringify(obj, null, 2))
    .replace(/&quot;([^&]+?)&quot;(\s*:)/g, '<span class="k">"$1"</span>$2')
    .replace(/:\s(&quot;.*?&quot;)/g, ': <span class="s">$1</span>')
    .replace(/:\s(-?\d+\.?\d*)/g, ': <span class="n">$1</span>');
}

function renderCertList(certs, selectedId) {
  $("cert-count").textContent = certs.length ? `${certs.length}` : "";
  if (!certs.length) {
    $("cert-list").innerHTML = `<div class="empty">No certificates yet.</div>`;
    return;
  }
  $("cert-list").innerHTML = `<table><tbody>${certs.map((c) => `
    <tr class="clickable ${c.erasure_id === selectedId ? "selected" : ""}"
        data-id="${esc(c.erasure_id)}">
      <td>
        <div class="mono">${esc(short(c.erasure_id, 13))}</div>
        <div class="dim" style="font-size:11px">shard ${c.shard} · ${esc(c.created_at.slice(0, 16).replace("T", " "))}</div>
      </td>
    </tr>`).join("")}</tbody></table>`;

  for (const row of $("cert-list").querySelectorAll("tr[data-id]")) {
    row.onclick = () => selectCert(row.dataset.id);
  }
}

async function selectCert(id) {
  const detail = $("cert-detail");
  detail.innerHTML = `<div class="empty">Verifying…</div>`;
  for (const r of $("cert-list").querySelectorAll("tr[data-id]")) {
    r.classList.toggle("selected", r.dataset.id === id);
  }
  try {
    const d = await get(`/api/certificates/${encodeURIComponent(id)}`);
    detail.innerHTML = `
      <div class="verdict ${d.verified ? "ok" : "bad"}">
        ${d.verified ? "VERIFIED" : "REJECTED"}
      </div>
      <ul class="findings ${d.verified ? "" : "bad"}">
        ${d.findings.map((f) => `<li>${esc(f)}</li>`).join("")}
      </ul>
      <div class="caveat">
        <b>Proves</b> the subject is absent from the record set whose root this names,
        and that it was signed by the holder of the private key.
        <b>Does not prove</b> the weights were trained on that record set — nothing binds
        the two. That rests on <span class="mono">code_digest</span> and on re-running
        sampled rebuilds.
      </div>
      <pre class="manifest">${highlight(d.manifest)}</pre>`;
  } catch (e) {
    detail.innerHTML = `<div class="empty">Could not load certificate: ${esc(e.message)}</div>`;
  }
}

/* ------------------------------------------------------------------- load */

let selected = null;

async function load() {
  const btn = $("refresh");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const [ov, pend, fail, certs] = await Promise.all([
      get("/api/overview"), get("/api/pending"), get("/api/failed"), get("/api/certificates"),
    ]);
    renderStats(ov);
    renderShards(ov.by_shard);
    renderAuc(ov.eval_history);
    renderPending(pend);
    renderFailed(fail);
    renderCertList(certs, selected);
    // Auto-select the newest certificate so the panel is never a dead end on
    // first load -- an operator opening this wants to see a verification.
    if (!selected && certs.length) {
      selected = certs[0].erasure_id;
      selectCert(selected);
    }
  } catch (e) {
    $("drift-banner").innerHTML =
      `<div class="banner"><strong>Load failed:</strong> ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh";
  }
}

$("refresh").onclick = load;

$("rebuild-form").onsubmit = async (e) => {
  e.preventDefault();
  const out = $("rebuild-result");
  const subject = $("subject").value.trim();
  if (!subject) {
    out.className = "result bad";
    out.textContent = "subject_id is required";
    return;
  }
  out.className = "result";
  out.textContent = "Submitting…";
  try {
    const r = await fetch("/api/rebuild", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ subject_id: subject, reason: $("reason").value }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
    out.className = "result ok";
    out.textContent = `queued ${d.response.erasure_id}`;
    $("subject").value = "";
    load();
  } catch (err) {
    out.className = "result bad";
    out.textContent = err.message;
  }
};

load();
