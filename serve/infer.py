"""
Phase 6 -- inference path for a single case.

Reuses, unmodified:
  - eval/phase4_equal_budget_ablation.py (data loading, feature builders,
    predict()) -- same import pattern eval/phase5_review_policy_ablation.py
    already uses.
  - models/artifacts/model_a.json, model_b.json (frozen, never retrained)
  - models/artifacts/model_a_meta.json, model_b_meta.json (locked
    thresholds/costs, same artifacts directory as the model files)

Does NOT retrain, retune, or refit anything. Does NOT touch the test split
except to read features for the one demo case being scored (identical to
how p4.score_test() already reads the test split for reporting).
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import phase4_equal_budget_ablation as p4  # Phase 1-4 code, imported only
from evidence import build_evidence, top_contributors


def load_frozen_models():
    """Identical pattern to phase5_review_policy_ablation.load_frozen_models()."""
    frozen_a = xgb.XGBClassifier()
    frozen_a.load_model(str(p4.MODELS_ARTIFACTS / "model_a.json"))
    frozen_b = xgb.XGBClassifier()
    frozen_b.load_model(str(p4.MODELS_ARTIFACTS / "model_b.json"))
    return frozen_a, frozen_b


def load_meta():
    """model_a_meta.json / model_b_meta.json live in models/artifacts/,
    alongside the frozen model_a.json / model_b.json -- NOT in configs/."""
    import json
    meta_a = json.loads((p4.MODELS_ARTIFACTS / "model_a_meta.json").read_text())
    meta_b = json.loads((p4.MODELS_ARTIFACTS / "model_b_meta.json").read_text())
    return meta_a, meta_b


def _contributions(model, X_row, cols):
    """Exact local TreeSHAP contributions via xgboost's built-in
    pred_contribs (no extra dependency, no approximation)."""
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_row[cols])
    contribs = booster.predict(dmat, pred_contribs=True)[0]  # len(cols)+1, last = bias
    return top_contributors(contribs, cols)


def decide_action(risk_a, risk_b, threshold_a, threshold_b):
    """
    3-tier action policy built only from the two already-locked thresholds.
    No new modeling.

    This is a separate Phase-6 demo/business-policy rule, NOT a reuse of
    Phase 5's review-routing policies -- it simply routes to human review
    whenever Model A (behavioral-only) and Model A+B (behavioral+relational)
    disagree about whether to flag this case. A+B is treated as the more
    robust model per Phases 3-5 and is authoritative for the block decision.
    """
    if risk_b >= threshold_b:
        return "block", (
            "Model A+B (behavioral+relational, frozen) risk is at/above its "
            "locked cost-optimal threshold."
        )
    if risk_a >= threshold_a:
        return "hold_for_review", (
            "Demo business-policy rule: Model A (behavioral-only) would flag "
            "this case while Model A+B does not. This model disagreement is "
            "routed to human review as a Phase-6 policy choice, distinct from "
            "the Phase 5 review-routing experiments."
        )
    return "approve", "Both frozen models score below their locked thresholds."


def score_case(data, customer_id, as_of_day, frozen_a, frozen_b, meta_a, meta_b):
    """Builds features for ONE customer at ONE as_of_day using the
    unmodified Phase 1-3 feature builders (via p4's thin wrappers), scores
    with both frozen models, decides an action, and returns a structured
    evidence dict. Returns None if the customer has no activity by
    as_of_day (features empty)."""
    beh = p4.build_behavioral(data, [customer_id], as_of_day)
    comb = p4.build_combined(data, [customer_id], as_of_day)
    if beh.empty or comb.empty:
        return None

    a_cols = p4.BEHAVIORAL_FEATURE_COLUMNS
    ab_cols = p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS

    risk_a = float(p4.predict(frozen_a, beh, a_cols)[0])
    risk_b = float(p4.predict(frozen_b, comb, ab_cols)[0])

    threshold_a = meta_a["threshold_cost_optimal"]
    threshold_b = meta_b["threshold_cost_optimal"]

    decision, basis = decide_action(risk_a, risk_b, threshold_a, threshold_b)

    contribs_a = _contributions(frozen_a, beh.fillna(0.0), a_cols)
    contribs_b = _contributions(frozen_b, comb.fillna(0.0), ab_cols)

    feature_values = {c: (None if pd_isna(comb.iloc[0][c]) else float(comb.iloc[0][c]))
                    for c in ab_cols}

    return build_evidence(
        customer_id=customer_id,
        as_of_day=as_of_day,
        feature_values=feature_values,
        risk_a=risk_a, risk_b=risk_b,
        threshold_a=threshold_a, threshold_b=threshold_b,
        contribs_a=contribs_a, contribs_b=contribs_b,
        decision=decision, decision_basis=basis,
    )


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)