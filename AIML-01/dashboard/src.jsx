import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const amountFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const caseStatuses = { open: "Open", investigating: "Investigating", closed: "Closed" };

function AlertQueue({ base, refresh }) {
  const [filter, setFilter] = useState("");
  const [queue, setQueue] = useState({ counts: {}, cases: [] });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const query = filter ? `?status=${filter}` : "";
        const response = await fetch(`${base}/cases${query}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Could not load the alert queue.");
        setQueue(await response.json());
      } catch (err) {
        if (!controller.signal.aborted) setError(err.message);
      }
    }
    load();
    const timer = setInterval(load, 5000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [base, filter, refresh]);

  async function save(event, id) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    setSaving(id);
    setError("");
    try {
      const response = await fetch(`${base}/cases/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: fields.get("status"), label: fields.get("label") || null }),
      });
      const updated = await response.json();
      if (!response.ok) throw new Error(updated.detail || "Could not save case.");
      setQueue((previous) => {
        const original = previous.cases.find((item) => item.transaction_id === id);
        const counts = { ...previous.counts };
        if (original && original.status !== updated.status) {
          counts[original.status] = Math.max(0, (counts[original.status] || 0) - 1);
          counts[updated.status] = (counts[updated.status] || 0) + 1;
        }
        const cases = previous.cases.map((item) => item.transaction_id === id ? updated : item)
          .filter((item) => !filter || item.status === filter);
        return { counts, cases };
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(null);
    }
  }

  return <section className="panel queue" aria-labelledby="queue-title">
    <div className="queue-heading">
      <div>
        <p className="eyebrow">INVESTIGATIONS</p>
        <h2 id="queue-title">Alert queue</h2>
      </div>
      <label>Filter by status
        <select value={filter} onChange={(event) => setFilter(event.target.value)}>
          <option value="">All cases</option>
          {Object.entries(caseStatuses).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
    </div>
    <div className="case-counts">
      {Object.entries(caseStatuses).map(([value, label]) => <span key={value}><b>{queue.counts[value] || 0}</b> {label}</span>)}
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {!queue.cases.length && <p className="muted">No cases in this view. Transactions marked REVIEW appear here.</p>}
    {queue.cases.map((item) => <form key={`${item.transaction_id}:${item.updated_at}`} className="case-row"
      onSubmit={(event) => save(event, item.transaction_id)}>
      <div>
        <strong>{item.transaction_id}</strong>
        <p>Risk {item.risk_score}/100 · {caseStatuses[item.status]}</p>
        {item.ring_id && <small>Ring lead {item.ring_id} · {percent(item.ring_score)}</small>}
      </div>
      <label>Case status
        <select name="status" defaultValue={item.status}>
          {Object.entries(caseStatuses).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <label>Feedback label
        <select name="label" defaultValue={item.label || ""}>
          <option value="">Unlabeled</option>
          <option value="fraud">Fraud</option>
          <option value="legitimate">Legitimate</option>
        </select>
      </label>
      <button disabled={saving !== null}>{saving === item.transaction_id ? "Saving…" : "Save case"}</button>
    </form>)}
  </section>;
}

function App() {
  const [api, setApi] = useState(import.meta.env.VITE_API_URL || "http://localhost:8000");
  const [health, setHealth] = useState("Checking API…");
  const [drift, setDrift] = useState(null);
  const [driftMessage, setDriftMessage] = useState("Loading drift summary…");
  const [retraining, setRetraining] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const base = api.trim().replace(/\/+$/, "");

  useEffect(() => {
    const controller = new AbortController();
    const options = { signal: controller.signal };
    setHealth("Checking API…");
    setDrift(null);
    setRetraining(null);
    setDriftMessage("Loading drift summary…");

    async function checkHealth() {
      try {
        const response = await fetch(`${base}/health`, options);
        if (!response.ok) throw new Error();
        const data = await response.json();
        setHealth(data.model_loaded ? "API and model ready" : "API ready · Model missing");
      } catch {
        if (!controller.signal.aborted) setHealth("API unavailable");
      }
    }

    async function loadDrift() {
      try {
        const response = await fetch(`${base}/monitoring/drift`, options);
        if (response.status === 404) {
          setDriftMessage("Generate a drift report to compare transaction windows.");
          return;
        }
        if (!response.ok) throw new Error();
        setDrift(await response.json());
      } catch {
        if (!controller.signal.aborted) setDriftMessage("Could not load the drift report.");
      }
    }

    checkHealth();
    loadDrift();
    async function loadRetraining() {
      try {
        const response = await fetch(`${base}/monitoring/retraining`, options);
        if (response.ok) setRetraining(await response.json());
      } catch {
        if (!controller.signal.aborted) setRetraining(null);
      }
    }
    loadRetraining();
    const timer = setInterval(() => { loadDrift(); loadRetraining(); }, 30_000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [base]);

  async function score(event) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    const optionalNumber = (name) => fields.get(name) === "" ? null : Number(fields.get(name));
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${base}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transaction_id: `dashboard-${Date.now()}`,
          TransactionAmt: Number(fields.get("amount")),
          TransactionDT: Number(fields.get("time")),
          card1: optionalNumber("card"),
          addr1: optionalNumber("address"),
          DeviceInfo: fields.get("device").trim() || null,
          P_emaildomain: fields.get("email").trim() || null,
          account_id: fields.get("account_id").trim() || null,
          card_id: fields.get("card_id").trim() || null,
          device_id: fields.get("device_id").trim() || null,
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(body.detail)
          ? body.detail.map(({ loc, msg }) => `${loc.at(-1)} — ${msg}`).join(". ")
          : body.detail;
        throw new Error(detail || "Scoring request failed");
      }
      setResult(body);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return <main>
    <header>
      <div>
        <p className="eyebrow">RISK OPERATIONS</p>
        <h1>Fraud Analyst Console</h1>
        <p className="muted">Transaction scoring and feature drift</p>
      </div>
      <span className="status" role="status">{health}</span>
    </header>

    <section className="grid">
      <article className="panel">
        <h2>Score a transaction</h2>
        <label>API base URL
          <input type="url" value={api} disabled={busy} onChange={(event) => setApi(event.target.value)} />
        </label>
        <form onSubmit={score}>
          <div className="fields">
            <label>Transaction amount
              <input name="amount" type="number" min="0" step="0.01" defaultValue="125.50" required />
            </label>
            <label>Transaction time (IEEE-CIS seconds)
              <input name="time" type="number" min="0" step="1" defaultValue="14000000" required />
            </label>
            <label>Card proxy ID
              <input name="card" type="number" min="0" step="1" defaultValue="1000" />
            </label>
            <label>Address proxy ID
              <input name="address" type="number" min="0" step="1" />
            </label>
            <label>Device info<input name="device" /></label>
            <label>Purchaser email domain<input name="email" /></label>
            <label>Account ID<input name="account_id" /></label>
            <label>Card ID<input name="card_id" /></label>
            <label>Device ID<input name="device_id" /></label>
          </div>
          <button disabled={busy}>{busy ? "Scoring…" : "Get fraud score"}</button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
        {result && <div className="result" aria-live="polite">
          <p className="eyebrow">{result.model === "xgboost+ring-gnn" ? "ENSEMBLE DECISION" : "XGBOOST DECISION"}</p>
          <strong>{result.risk_score}/100</strong>
          <h3>{result.decision}</h3>
          <p>Fraud probability {percent(result.fraud_probability)}</p>
          <small>Transaction {result.transaction_id}</small>
          <h3 className="signals-heading">Supporting signals</h3>
          {result.xgb_probability != null && <div className="metric">
            <span>XGBoost probability</span><b>{percent(result.xgb_probability)}</b>
          </div>}
          <div className="metric">
            <span>GNN score</span>
            <b>{result.gnn_score == null ? "Unavailable" : result.gnn_score.toFixed(3)}</b>
          </div>
          <div className="metric">
            <span>Feast · Historical transactions</span>
            <b>{result.card_history
              ? result.card_history.transaction_count.toLocaleString()
              : "Unavailable"}</b>
          </div>
          {result.card_history && <div className="metric">
            <span>Historical mean amount</span>
            <b>{amountFormat.format(result.card_history.mean_amount)}</b>
          </div>}
          {result.ring_id && <div className="metric">
            <span>Ring lead {result.ring_id}</span><b>{percent(result.ring_score)}</b>
          </div>}
        </div>}
      </article>

      <article className="panel drift">
        <h2>Data drift</h2>
        {!drift ? <p className="muted" role="status">{driftMessage}</p> : <>
          <p className={drift.dataset_drift ? "error" : "status"}>
            {drift.dataset_drift ? "Dataset drift detected" : "Below dataset drift threshold"}
          </p>
          <div className="metric">
            <span>Drifted features</span>
            <b>{drift.drifted_columns} / {drift.compared_columns}</b>
          </div>
          <meter aria-label="Share of drifted features" min="0" max="1" value={drift.drift_share} />
          <dl className="windows">
            <dt>Reference window</dt>
            <dd>{drift.reference_window} · {drift.reference_rows.toLocaleString()} rows</dd>
            <dt>Current window</dt>
            <dd>{drift.current_window} · {drift.current_rows.toLocaleString()} rows</dd>
          </dl>
          <p className="muted">Generated {new Date(drift.generated_at).toLocaleString()}</p>
          <a href={`${base}/monitoring/drift/report`} target="_blank" rel="noreferrer">Open Evidently report ↗</a>
        </>}
        {retraining && <div className="retraining">
          <h3>Automatic retraining</h3>
          <p className="muted">{retraining.status.replaceAll("_", " ")}</p>
          {retraining.candidate_id && <p>Candidate {retraining.candidate_id} · AP {retraining.candidate_average_precision.toFixed(3)}</p>}
        </div>}
      </article>
    </section>
    <AlertQueue base={base} refresh={result?.transaction_id} />
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
