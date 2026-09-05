from pathlib import Path
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.features import build_behavioral_features, BEHAVIORAL_FEATURE_COLUMNS
from models.features_b import build_relational_features, RELATIONAL_FEATURE_COLUMNS

DATA = ROOT / "data"
ART = ROOT / "models" / "artifacts"
RESULTS = ROOT / "eval" / "results"
CONFIG = ROOT / "configs" / "config.yaml"
REVIEWS = ROOT / "backend" / "reviews.json"

with open(CONFIG) as f:
    CFG = yaml.safe_load(f)

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
returns = pd.read_csv(DATA / "returns.csv")
products = pd.read_csv(DATA / "products.csv")
splits = pd.read_csv(DATA / "splits.csv")

meta_a = json.loads((ART / "model_a_meta.json").read_text())
meta_b = json.loads((ART / "model_b_meta.json").read_text())

booster_a = xgb.Booster()
booster_a.load_model(str(ART / "model_a.json"))
booster_b = xgb.Booster()
booster_b.load_model(str(ART / "model_b.json"))

A_COLS = meta_a["feature_columns"]
B_COLS = meta_b["feature_columns"]

# Frozen thresholds from the supplied trained-artifact metadata.
A_THRESHOLD = float(meta_a["threshold_cost_optimal"])
B_THRESHOLD = float(meta_b["threshold_cost_optimal"])

# The supplied repo does not contain a separate APPROVE/HOLD/BLOCK policy file.
# The demo therefore keeps the model decision deterministic and exposes the
# exact frozen B threshold; this adapter is intentionally isolated so a supplied
# policy can replace it without changing any experiment artifacts.
def deterministic_decision(score_b: float) -> str:
    if score_b >= B_THRESHOLD:
        return "BLOCK"
    # A score below the frozen B threshold but above the frozen A threshold
    # is treated as a review band in the product layer.
    if score_b >= A_THRESHOLD:
        return "HOLD"
    return "APPROVE"

def load_reviews():
    if REVIEWS.exists():
        try:
            return json.loads(REVIEWS.read_text())
        except Exception:
            return []
    return []

def save_reviews(rows):
    REVIEWS.write_text(json.dumps(rows, indent=2))

def safe_float(x):
    x = float(x)
    return None if not math.isfinite(x) else x

def top_contribs(booster, X, columns, k=7):
    try:
        dm = xgb.DMatrix(X, feature_names=columns)
        contrib = booster.predict(dm, pred_contribs=True)
        vals = contrib[0, :-1]
        items = [{"feature": c, "contribution": safe_float(v), "direction": "raises risk" if v > 0 else "lowers risk"}
                 for c, v in zip(columns, vals)]
        items.sort(key=lambda z: abs(z["contribution"]) if z["contribution"] is not None else -1, reverse=True)
        return items[:k]
    except Exception as e:
        return [{"feature": "SHAP unavailable", "contribution": None, "direction": str(e)}]

def feature_rows(feats, cols):
    row = feats.iloc[0]
    out = []
    for c in cols:
        v = safe_float(row[c])
        out.append({"feature": c, "value": v})
    return out

def analyze_customer(customer_id: str):
    customer_id = customer_id.strip()
    if customer_id not in set(customers["customer_id"]):
        raise HTTPException(status_code=404, detail="Customer ID not found")

    cust = customers[customers.customer_id == customer_id].iloc[0]
    # Use the latest supplied benchmark/test horizon for a real customer lookup.
    as_of_day = int(CFG["simulation"]["rounds"][1]["end"])

    beh = build_behavioral_features(
        orders, returns, products, customers, as_of_day=as_of_day, customer_ids={customer_id}
    )
    rel = build_relational_features(
        orders, returns, products, as_of_day=as_of_day, customer_ids={customer_id}
    )
    if beh.empty:
        raise HTTPException(status_code=422, detail="Customer has no observed orders by the benchmark cutoff")

    merged = beh.merge(rel, on="customer_id", how="left")
    for c in RELATIONAL_FEATURE_COLUMNS:
        merged[c] = merged[c].fillna(0.0)

    xa = merged[A_COLS].fillna(0.0)
    xb = merged[B_COLS].fillna(0.0)
    score_a = float(booster_a.predict(xgb.DMatrix(xa, feature_names=A_COLS))[0])
    score_b = float(booster_b.predict(xgb.DMatrix(xb, feature_names=B_COLS))[0])
    decision = deterministic_decision(score_b)

    # Evidence is derived from the exact feature values used by the frozen models.
    r = merged.iloc[0]
    behavioral = [
        {"label": "Return rate", "value": safe_float(r["return_rate"]), "display": f"{r['return_rate']:.1%}",
         "why": "Higher return frequency increases behavioral risk."},
        {"label": "Fast returns", "value": safe_float(r["pct_fast_returns"]), "display": f"{r['pct_fast_returns']:.1%}",
         "why": "Share of observed returns completed within the behavioral fast-return window."},
        {"label": "Returns", "value": safe_float(r["n_returns"]), "display": f"{int(r['n_returns'])}",
         "why": "Observed returns by the frozen benchmark cutoff."},
        {"label": "Orders", "value": safe_float(r["n_orders"]), "display": f"{int(r['n_orders'])}",
         "why": "Observed orders by the frozen benchmark cutoff."},
        {"label": "Returns / week", "value": safe_float(r["returns_per_week"]), "display": f"{r['returns_per_week']:.2f}",
         "why": "Return frequency normalized by observed account lifetime."},
    ]
    relational = [
        {"label": "Device sharing", "value": safe_float(r["device_sharing_count"]), "display": f"{int(r['device_sharing_count'])}",
         "why": "Other customers observed on the same device identifier."},
        {"label": "Address sharing", "value": safe_float(r["address_sharing_count"]), "display": f"{int(r['address_sharing_count'])}",
         "why": "Other customers observed on the same address identifier."},
        {"label": "Payment fingerprint sharing", "value": safe_float(r["payment_fp_sharing_count"]), "display": f"{int(r['payment_fp_sharing_count'])}",
         "why": "Other customers observed on the same payment fingerprint."},
        {"label": "Identifier cluster", "value": safe_float(r["identifier_cluster_size"]), "display": f"{int(r['identifier_cluster_size'])}",
         "why": "Size of the connected identifier cluster."},
        {"label": "Suspicious-neighbor ratio", "value": safe_float(r["suspicious_neighbor_ratio"]), "display": f"{r['suspicious_neighbor_ratio']:.1%}",
         "why": "Share of connected neighbors whose observed behavior is suspicious."},
        {"label": "Recent burst score", "value": safe_float(r["recent_burst_score"]), "display": f"{r['recent_burst_score']:.2f}",
         "why": "Recency-weighted return activity."},
    ]
    explanation = (
        f"Model A gives a behavioral risk score of {score_a:.3f}; Model B gives {score_b:.3f}. "
        f"The frozen Model B threshold is {B_THRESHOLD:.3f}. "
        + ("Relational evidence materially increases the risk view." if score_b > score_a
           else "The relational model does not increase the score relative to Model A.")
        + f" Final product decision: {decision}."
    )

    shap_items = top_contribs(booster_b, xb, B_COLS)
    shap_available = not (len(shap_items) == 1 and shap_items[0]["feature"] == "SHAP unavailable")

    return {
        "customer": {
            "customer_id": customer_id,
            "population_type": str(cust["population_type"]),
            "signup_day": int(cust["signup_day"]),
            "hard_negative_subtype": None if pd.isna(cust["hard_negative_subtype"]) else str(cust["hard_negative_subtype"]),
            "ring_id": None if pd.isna(cust["ring_id"]) else str(cust["ring_id"]),
        },
        "cutoff": {"as_of_day": as_of_day, "name": "round_1 benchmark/test horizon"},

        # Legacy/debug shape — kept in case other tooling reads it.
        "scores": {
            "model_a": score_a, "model_b": score_b,
            "model_a_threshold": A_THRESHOLD, "model_b_threshold": B_THRESHOLD,
            "delta_b_minus_a": score_b - score_a,
        },

        # Shape actually consumed by the frontend AnalyzePage.
        "model_a": {"score": score_a, "threshold": A_THRESHOLD},
        "model_b": {"score": score_b, "threshold": B_THRESHOLD},

        "decision": decision,
        "behavioral_evidence": [{"label": e["label"], "value": e["display"]} for e in behavioral],
        "relational_evidence": [{"label": e["label"], "value": e["display"]} for e in relational],
        "model_comparison": [
            {"model": "Model A", "scope": "behavioral", "score": score_a, "threshold": A_THRESHOLD},
            {"model": "Model B", "scope": "behavioral + relational", "score": score_b, "threshold": B_THRESHOLD},
        ],

        "shap_available": shap_available,
        "top_evidence": shap_items if shap_available else [],
        "shap": shap_items,  # legacy alias

        "behavioral_features": feature_rows(merged, A_COLS),
        "relational_features": feature_rows(merged, RELATIONAL_FEATURE_COLUMNS),

        "explanation": explanation,
        "reviewer_explanation": explanation,
    }

class Review(BaseModel):
    customer_id: str
    action: str
    note: str = ""

app = FastAPI(title="AdaptiveGuard Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_a_loaded": True,
        "model_b_loaded": True,
        "data_loaded": True,
        "thresholds": {"model_a": A_THRESHOLD, "model_b": B_THRESHOLD},
    }

@app.get("/api/customers/search")
def search_customers(q: str = "", limit: int = 10):
    q = q.strip().lower()
    rows = customers[customers.customer_id.str.lower().str.contains(q, regex=False)] if q else customers.head(limit)
    rows = rows.head(min(limit, 25))
    return rows[["customer_id", "population_type", "hard_negative_subtype"]].fillna("").to_dict("records")

@app.get("/api/analyze/{customer_id}")
def analyze(customer_id: str):
    return analyze_customer(customer_id)

@app.get("/api/evaluation")
def evaluation():
    """Return the saved research evaluation artifacts without rerunning them."""
    def load_json(name):
        path = RESULTS / name
        return json.loads(path.read_text()) if path.exists() else None

    p1 = load_json("phase1_model_a_test_metrics.json")
    p2 = load_json("phase2_adaptive_eval_metrics.json")
    p3 = load_json("phase3_model_b_eval_metrics.json")
    p4 = load_json("phase4_equal_budget_ablation_metrics.json")
    p5 = load_json("phase5_review_policy_metrics.json")

    rounds = []
    if p2 and p3:
        for label in ["round_1", "round_2", "round_3", "round_4"]:
            try:
                a = p2["rounds"]["round_1_resc" if label == "round_1" else label]
                b = p3["rounds"][label]
                cmp = p3["comparison_vs_model_a"][label]
            except KeyError:
                # A round is missing from one of the saved artifacts — skip it
                # rather than 500ing the whole evaluation endpoint.
                continue
            rounds.append({
                "round": label,
                "day": a["as_of_day"],
                "model_a": {
                    "adaptive_abuse_recall": a["by_population"]["adaptive_abuse"].get("recall"),
                    "hard_negative_fpr": a["by_population"]["hard_negative"].get("fpr"),
                    "total_cost": a["overall"]["total_cost"],
                },
                "model_b": {
                    "adaptive_abuse_recall": b["by_population"]["adaptive_abuse"].get("recall"),
                    "hard_negative_fpr": b["by_population"]["hard_negative"].get("fpr"),
                    "total_cost": b["overall"]["total_cost"],
                },
                "delta": {
                    "adaptive_recall_b_minus_a": cmp["adaptive_abuse_recall_B"] - cmp["adaptive_abuse_recall_A"],
                    "cost_b_minus_a": cmp["total_cost_B"] - cmp["total_cost_A"],
                }
            })

    phase4 = None
    if p4:
        phase4_rounds = []
        for key in ["round_1", "round_2", "round_3", "round_4"]:
            rr = p4.get("rounds", {}).get(key)
            if not rr:
                continue
            hero = p4.get("hero_comparison", {}).get(key, {})
            phase4_rounds.append({
                "round": key,
                "day": rr.get("as_of_day"),
                "review_budget": rr.get("review_budget_used_before_scoring", 0),
                "adaptive_abuse_recall": hero.get("adaptive_abuse_recall", {}),
                "hard_negative_fpr": hero.get("hard_negative_fpr", {}),
                "variants": hero,
            })
        phase4 = {
            "experiment": p4.get("experiment"),
            "protocol": p4.get("protocol", {}),
            "review_audit_counts": p4.get("review_audit_counts", {}),
            "rounds": phase4_rounds,
        }

    phase5 = None
    if p5:
        phase5_rounds = []
        hero_all = p5.get("hero_comparison", {})
        for key in ["round_1", "round_2", "round_3", "round_4"]:
            if key in hero_all:
                phase5_rounds.append({"round": key, "policies": hero_all[key]})
        phase5 = {
            "experiment": p5.get("experiment"),
            "protocol": p5.get("protocol", {}),
            "review_quality_by_policy": p5.get("review_quality_by_policy", {}),
            "rounds": phase5_rounds,
        }

    return {
        "rounds": rounds,
        "phase1": p1,
        "sanity_check_round1_matches_phase1": p2.get("sanity_check_round1_matches_phase1") if p2 else None,
        "phase4": phase4,
        "phase5": phase5,
        "research_artifacts": {
            "phase1": bool(p1), "phase2": bool(p2), "phase3": bool(p3),
            "phase4": bool(p4), "phase5": bool(p5),
        },
    }

@app.post("/api/reviews")
def review(payload: Review):
    if payload.action not in {"APPROVE", "CONFIRM_ABUSE", "ESCALATE"}:
        raise HTTPException(status_code=400, detail="Invalid review action")
    # Review records are product/demo state only; research artifacts are untouched.
    rows = load_reviews()
    rows.append({
        "customer_id": payload.customer_id,
        "action": payload.action,
        "note": payload.note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    save_reviews(rows)
    return {"ok": True, "saved": rows[-1]}