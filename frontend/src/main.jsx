import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

function Card({ title, children, className = "" }) {
  return <section className={`card ${className}`}><h3>{title}</h3>{children}</section>;
}

function Metric({ label, value, hint }) {
  return <div className="metric"><div className="metric-label">{label}</div><div className="metric-value">{value}</div>{hint && <div className="metric-hint">{hint}</div>}</div>;
}

function formatPct(v) {
  return v == null ? "—" : `${(Number(v) * 100).toFixed(1)}%`;
}
function formatCost(v) {
  return v == null ? "—" : Number(v).toFixed(2);
}

function MiniBar({ value }) {
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return <div className="bar"><i style={{ width: `${pct}%` }} /></div>;
}

function VariantCell({ variant, metric }) {
  const v = variant?.[metric];
  return <td>{v == null ? "—" : formatPct(v)}</td>;
}

function EvaluationPage({ evaluation, evalError }) {
  if (evalError) return <div className="loading">Failed to load evaluation: {evalError}</div>;
  if (!evaluation) return <div className="loading">Loading evaluation…</div>;

  const rounds = evaluation.rounds || [];
  const p4 = evaluation.phase4;
  const p5 = evaluation.phase5;
  const policies = ["max_ab", "disagreement", "risk_x_disagreement"];
  const label = p => p === "max_ab" ? "max(A,B)" : p === "risk_x_disagreement" ? "risk × disagreement" : "disagreement";

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">RESEARCH EVALUATION · PHASES 1–5</div>
          <h1>Adaptive Defense Evaluation</h1>
          <p className="muted">Saved experiment results — no retraining, retuning, threshold changes, or protocol reruns from the demo.</p>
        </div>
        <div className="status-pill">ALL SAVED ARTIFACTS LOADED</div>
      </div>

      <div className="metric-grid">
        <Metric label="Rounds evaluated" value={rounds.length} hint="Phase 1–3 held-out comparison" />
        <Metric label="Phase 4 budget" value={p4?.protocol?.total_reviewed_labels ?? "—"} hint="30 after R2 + 30 after R3" />
        <Metric label="Phase 5 policies" value={p5 ? policies.length : "—"} hint="Equal 60-label budget each" />
        <Metric label="Round-1 sanity" value={evaluation?.sanity_check_round1_matches_phase1 ? "PASS" : "—"} hint="Phase 2 vs Phase 1" />
      </div>

      <Card title="Phase 1–3: frozen model comparison">
        <div className="table-wrap">
          <table><thead><tr>
            <th>Round</th><th>Day</th><th>Model A adaptive recall</th><th>Model B adaptive recall</th>
            <th>Hard-negative FPR A</th><th>Hard-negative FPR B</th><th>Cost A</th><th>Cost B</th>
          </tr></thead><tbody>
            {rounds.map(r => <tr key={r.round}>
              <td><strong>{r.round.replace("round_", "Round ")}</strong></td><td>{r.day}</td>
              <td>{formatPct(r.model_a?.adaptive_abuse_recall)}</td><td className="emphasis">{formatPct(r.model_b?.adaptive_abuse_recall)}</td>
              <td>{formatPct(r.model_a?.hard_negative_fpr)}</td><td className="emphasis">{formatPct(r.model_b?.hard_negative_fpr)}</td>
              <td>{formatCost(r.model_a?.total_cost)}</td><td className="emphasis">{formatCost(r.model_b?.total_cost)}</td>
            </tr>)}
          </tbody></table>
        </div>
      </Card>

      <Card title="Phase 4 — equal-review-budget ablation">
        {p4 ? <>
          <div className="notice">
            <strong>Controlled 60-label comparison.</strong>
            <p>Validation split only · 30 labels after Round 2 + 30 after Round 3 · same selected customer IDs for A and A+B retraining · frozen thresholds remain locked · final test untouched.</p>
          </div>
          <div className="table-wrap">
            <table><thead><tr>
              <th>Round</th><th>Labels used</th><th>A frozen recall</th><th>A retrain recall</th><th>A+B frozen recall</th><th>A+B retrain recall</th><th>A+B retrain cost</th>
            </tr></thead><tbody>
              {p4.rounds.map(r => <tr key={r.round}>
                <td><strong>{r.round.replace("round_", "Round ")}</strong></td><td>{r.review_budget}</td>
                <VariantCell variant={r.adaptive_abuse_recall} metric="A_frozen" />
                <VariantCell variant={r.adaptive_abuse_recall} metric="A_retrain" />
                <td className="emphasis">{formatPct(r.adaptive_abuse_recall?.["A+B_frozen"])}</td>
                <td className="emphasis">{formatPct(r.adaptive_abuse_recall?.["A+B_retrain"])}</td>
                <td>{formatCost(r.variants?.["A+B_retrain"]?.total_cost)}</td>
              </tr>)}
            </tbody></table>
          </div>
        </> : <div className="notice">Phase 4 metrics are not present in the local eval/results directory.</div>}
      </Card>

      <div className="two-col">
        <Card title="Phase 4 adaptive-abuse recall">
          {p4?.rounds?.map(r => <div className="cost-row" key={r.round}>
            <span>{r.round.replace("round_", "R")}</span>
            <span>A frozen <b>{formatPct(r.adaptive_abuse_recall?.A_frozen)}</b></span>
            <span>A retrain <b>{formatPct(r.adaptive_abuse_recall?.A_retrain)}</b></span>
            <strong>A+B retrain {formatPct(r.adaptive_abuse_recall?.["A+B_retrain"])}</strong>
          </div>)}
        </Card>
        <Card title="Phase 4 false-positive impact">
          {p4?.rounds?.map(r => <div className="cost-row" key={r.round}>
            <span>{r.round.replace("round_", "R")}</span>
            <span>A retrain {formatPct(r.hard_negative_fpr?.A_retrain)}</span>
            <strong>A+B retrain {formatPct(r.hard_negative_fpr?.["A+B_retrain"])}</strong>
          </div>)}
        </Card>
      </div>

      <Card title="Phase 5 — review-policy ablation">
        {p5 ? <>
          <div className="notice">
            <strong>Equal 60-label budget per policy with noisy, unresolved and delayed outcomes.</strong>
            <p>Policies: max(A,B), disagreement, and risk × disagreement. Validation split only; outcome seed is shared across policies so differences come from routing.</p>
          </div>
          <div className="table-wrap">
            <table><thead><tr>
              <th>Policy</th><th>Reviewed</th><th>Abuse yield</th><th>Resolved</th><th>Unresolved</th><th>Noisy labels</th><th>Agreement</th>
            </tr></thead><tbody>
              {policies.map(policy => { const q = p5.review_quality_by_policy?.[policy]; return <tr key={policy}>
                <td><strong>{label(policy)}</strong></td><td>{q?.n_reviewed ?? "—"}</td><td>{formatPct(q?.abuse_yield)}</td><td>{q?.n_resolved ?? "—"}</td><td>{q?.n_unresolved ?? "—"}</td><td>{q?.n_noisy_labels ?? "—"}</td><td>{formatPct(q?.observed_vs_oracle_label_agreement)}</td>
              </tr>})}
            </tbody></table>
          </div>
        </> : <div className="notice">Phase 5 metrics are not present in the local eval/results directory.</div>}
      </Card>

      <Card title="Phase 5 adaptive-abuse recall by routing policy">
        {p5?.rounds?.map(r => <div key={r.round} className="policy-block">
          <h4>{r.round.replace("round_", "Round ")}</h4>
          {policies.map(policy => {
            const d = r.policies?.[policy]?.adaptive_abuse_recall || {};
            return <div className="bar-row" key={policy}>
              <span>{label(policy)}</span><MiniBar value={d["A+B_retrain"]} /><b>{formatPct(d["A+B_retrain"])}</b>
            </div>;
          })}
        </div>)}
      </Card>

      <Card title="Research integrity">
        <ul className="clean-list">
          <li>Reads the saved Phase 1–5 JSON metrics; the web demo does not rerun the experiments.</li>
          <li>Model A is the frozen behavioral baseline; Model B is the frozen behavioral + relational model.</li>
          <li>Phase 4 uses a fixed 30-after-R2 + 30-after-R3 review budget and compares frozen controls with retrained variants.</li>
          <li>Phase 5 gives each routing policy an independent equal 60-label budget and simulates noisy, unresolved and delayed review outcomes.</li>
          <li>Frozen thresholds and final-test isolation are preserved by the supplied experimental artifacts.</li>
          <li>Reviewer actions in the Customer Review page are product/demo state only; they do not modify research artifacts.</li>
        </ul>
      </Card>
    </div>
  );
}

function AnalyzePage({ customerId, setCustomerId, result, analyze, loading, onReview }) {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">CUSTOMER REVIEW</div>
          <h1>AdaptiveGuard</h1>
          <p className="muted">Evidence-first abuse-risk review using the frozen research models.</p>
        </div>
      </div>

      <div className="search-row">
        <input value={customerId} onChange={e=>setCustomerId(e.target.value.toUpperCase().trim())} placeholder="Customer ID e.g. C0000003" onKeyDown={e=>e.key==="Enter"&&analyze()} />
        <button type="button" onClick={analyze} disabled={loading}>{loading ? "Analyzing…" : "Analyze"}</button>
      </div>

      {!result && <div className="empty"><strong>Try C0000003</strong><span>It is a valid supplied test customer.</span></div>}

      {result && <>
        <div className="metric-grid">
          <Metric label="Customer" value={result.customer.customer_id} hint={result.customer.population_type} />
          <Metric label="Final decision" value={result.decision} hint="Deterministic product-layer adapter" />
          <Metric label="Model A score" value={Number(result.model_a.score).toFixed(4)} hint={`threshold ${Number(result.model_a.threshold).toFixed(4)}`} />
          <Metric label="Model B score" value={Number(result.model_b.score).toFixed(4)} hint={`threshold ${Number(result.model_b.threshold).toFixed(4)}`} />
        </div>

        <div className="two-col">
          <Card title="Behavioral evidence">
            <ul className="evidence">{result.behavioral_evidence.map((x,i)=><li key={i}>{x.label}<b>{x.value}</b></li>)}</ul>
          </Card>
          <Card title="Relational evidence">
            <ul className="evidence">{result.relational_evidence.map((x,i)=><li key={i}>{x.label}<b>{x.value}</b></li>)}</ul>
          </Card>
        </div>

        <Card title="Model comparison">
          <div className="comparison"><div><span>Model A</span><strong>{result.model_a.score.toFixed(4)}</strong></div><div><span>Model B</span><strong>{result.model_b.score.toFixed(4)}</strong></div><div><span>B − A</span><strong>{result.model_b.score - result.model_a.score >= 0 ? "+" : ""}{(result.model_b.score-result.model_a.score).toFixed(4)}</strong></div></div>
        </Card>

        <Card title="SHAP / top model evidence">
          {result.shap_available ? <ul className="evidence">{result.top_evidence.map((x,i)=><li key={i}>{x.feature}<b>{Number(x.contribution).toFixed(4)}</b></li>)}</ul> : <div className="notice">SHAP unavailable for this artifact; no explanation is fabricated.</div>}
        </Card>

        <Card title="Reviewer explanation">
          <p className="explanation">{result.reviewer_explanation}</p>
          <div className="actions">
            <button type="button" onClick={()=>onReview(result.customer.customer_id,"APPROVE")}>Approve</button>
            <button type="button" onClick={()=>onReview(result.customer.customer_id,"CONFIRM_ABUSE")}>Confirm Abuse</button>
            <button type="button" onClick={()=>onReview(result.customer.customer_id,"ESCALATE")}>Escalate</button>
          </div>
        </Card>
      </>}
    </div>
  );
}

async function submitReview(customerId, action) {
  const res = await fetch(`${API}/reviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId, action }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Review request failed (${res.status})`);
  }
}

function App() {
  const [tab, setTab] = useState("analyze");
  const [customerId, setCustomerId] = useState("C0000003");
  const [result, setResult] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evalError, setEvalError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/evaluation`)
      .then(r => r.json())
      .then(setEvaluation)
      .catch(e => setEvalError(e.message || "network error"));
  }, []);

  async function analyze() {
    if (!customerId) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/analyze/${encodeURIComponent(customerId)}`);
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (e) { alert(e.message); }
    finally { setLoading(false); }
  }

  async function onReview(customerId, action) {
    try {
      await submitReview(customerId, action);
      alert(`Review action saved: ${action}`);
    } catch (e) { alert(e.message); }
  }

  return <div className="app">
    <header className="topbar">
      <div className="brand">AdaptiveGuard <span>Razorpay Buildathon</span></div>
      <nav>
        <button type="button" className={tab==="analyze"?"active":""} onClick={()=>setTab("analyze")}>Customer Review</button>
        <button type="button" className={tab==="evaluation"?"active":""} onClick={()=>setTab("evaluation")}>Evaluation</button>
      </nav>
    </header>
    {tab==="analyze"
      ? <AnalyzePage {...{customerId,setCustomerId,result,analyze,loading,onReview}} />
      : <EvaluationPage evaluation={evaluation} evalError={evalError} />}
  </div>;
}
createRoot(document.getElementById("root")).render(<App />);