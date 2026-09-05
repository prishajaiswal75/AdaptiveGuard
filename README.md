# AdaptiveGuard

**Adaptive Abuse Defense with Behavioral + Relational Intelligence**
Razorpay Buildathon — Track 2: AI Risk Manager · Serial & coordinated return abuse

AdaptiveGuard tests one question: **given the same review budget, does adding relational context (shared devices, addresses, payment fingerprints) beat a behavioral-only model against *adaptive* return abuse?**

Two frozen models are compared — **Model A** (behavioral) and **Model B** (behavioral + relational) — across adaptive-abuse rounds, an equal-review-budget ablation, and different human-review routing policies. A reviewer-facing web app surfaces both models' evidence and uses **Gemini** to turn it into a plain-English investigation summary.

## Results at a glance

**Adaptive-abuse recall decays for Model A round over round; Model B holds:**

| Round | Model A recall | Model B recall | Total cost, A vs. B |
|---|---|---|---|
| 1 | 88.3% | 98.2% | 485 vs. 85 |
| 2 | 74.3% | 97.3% | 845 vs. 80 |
| 3 | 56.1% | 93.0% | 1,349 vs. 214 |
| 4 | 40.4% | 91.2% | 1,871 vs. 318 |

- **Equal review budget (Phase 4, 60 oracle-labeled reviews total):** retraining Model A on those labels leaves it stuck at ~40% recall by round 4 — the labels alone don't teach it what the relational signal already knows. Model B stays at 90–91% recall whether frozen or retrained.
- **Review-routing policy (Phase 5):** routing the same budget toward cases where A and B *disagree* lets a retrained Model A partially recover — 73.7% recall vs. 40.4% under default max-score routing. Model B stays well ahead regardless of policy (72–91% recall) and never needs retraining to stay effective.
- **Static test set (Phase 1, Model A alone):** precision 97.2%, recall 92.4%, F1 94.7%, AUC 0.979 — strong on i.i.d. data; the adaptive rounds above are where it breaks down.
- **Leakage check:** 0 of ~5,000 identifier clusters span train/val/test splits — Model B's advantage isn't coming from leaked future information.

*(Computed from `eval/results/phase1-5_*.json` on the provided synthetic dataset.)*

## How it works

```
Customer ID
     │
     ▼
Feature pipeline ──► Model A (behavioral) ─────┐
     │                                          ├─► Frozen thresholds ─► APPROVE / HOLD / BLOCK
     └──────────────► Model B (+ relational) ───┘                              │
                                                                                 ▼
                                                     Gemini turns model evidence into a
                                                     reviewer-readable investigation summary
```

**Model A — behavioral:** order volume/value, AOV, merchant/category diversity, return frequency/timing/rate, account age, recent activity.

**Model B adds — relational:** shared device/address/payment-fingerprint counts, identifier-cluster size & return rate, suspicious-neighbor ratio, merchant/category-relative deviation, recent burst score.

Both are **frozen research artifacts** — the app is a demo layer, not a retraining pipeline. Gemini only explains evidence; it never scores, sets thresholds, or overrides the decision.

## Quick start

```bash
git clone <repo-url> && cd adaptiveguard_demo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify.py                                    # checks data + frozen artifacts

# Terminal 1
uvicorn backend.main:app --reload --port 8000       # → http://127.0.0.1:8000 (health: /health)

# Terminal 2
cd frontend && npm install && npm run dev           # → http://localhost:5173
```

Optional: add `GEMINI_API_KEY` to a `.env` file at the project root to enable the Investigator panel — the rest of the app works without it.

## Try it

1. Open the app, enter test customer **`C0000003`**, click **Analyze**.
2. Compare Model A vs. Model B scores and the final decision.
3. Review behavioral, relational, and SHAP evidence.
4. Read the Gemini investigation summary.
5. Take a reviewer action — Approve / Confirm Abuse / Escalate.
6. Open **Evaluation** to see Phase 2–5 results (adaptive recall, equal-budget ablation, review-policy comparison).

## Reproducing the experiments

```bash
python eval/phase2_adaptive_eval.py
python eval/phase3_model_b_eval.py
python eval/phase4_equal_budget_ablation.py
python eval/phase5_review_policy_ablation.py
```

Results are written to `eval/results/*.json`, which is what the Evaluation tab reads — the app never reruns experiments at startup.

## Project structure

```
adaptiveguard_demo/
├── backend/main.py         FastAPI inference + evidence API
├── models/                 features_a/b.py, model_a/b.py + frozen artifacts/
├── eval/                   Phase 2–5 scripts + results/*.json
├── data_gen/                synthetic data generation + entity-aware splits
├── data/                    customers, orders, returns, products, merchants, splits
├── frontend/src/main.jsx    React reviewer + evaluation dashboard
├── configs/config.yaml      experiment configuration
└── verify.py                checks data, config, and frozen artifacts
```

## Design principles

- **Frozen research artifacts** — model weights, thresholds, splits, and feature definitions are never modified by the app.
- **Entity-aware, temporal splits** to prevent relational leakage (checked explicitly in Phase 3).
- **Equal-budget comparison** — Model B only counts as better if it holds under the same human-review capacity as Model A (Phase 4).
- **Demo decision adapter** exists only for the UI, not the research evaluation:
  `score ≥ B-threshold → BLOCK` · `A-threshold ≤ score < B-threshold → HOLD` · `score < A-threshold → APPROVE`

## Limitations & responsible use

- Synthetic/evaluation dataset; a research prototype, not production-ready.
- Gemini explains evidence — it is not the fraud decision, and its output should be human-verified.
- SHAP availability depends on installed dependencies.
- The adaptive-abuse harness is a closed, offline simulation — it does not touch production systems or generate real-world evasion guidance.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Frontend blank page | `cd frontend && npm install && npm run dev`, check browser console (F12) |
| Backend won't start | Run `uvicorn` from the project root, not from `backend/` |
| API connection error | Confirm backend (`:8000`) and frontend (`:5173`) are both running |
| No Gemini explanation | Check `GEMINI_API_KEY` in `.env` — core pipeline works without it |
| Evaluation page empty | Confirm `eval/results/*.json` (Phase 1–5) exist |
