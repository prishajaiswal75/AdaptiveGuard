
"""
Phase 4 — equal-review/label-budget ablation.

Protocol:
- Final test customers are NEVER reviewed or used for retraining.
- Validation customers (597) are the adaptation/review pool.
- Exactly 30 validation cases are reviewed after Round 2 and 30 after Round 3:
  60 reviewed labels total = 60/597 = 10.05% of the validation pool.
- The SAME reviewed customer IDs and oracle labels are supplied to both
  retraining variants.
- Selection is common and deterministic: at each review cutoff, rank the
  not-yet-reviewed validation cases by max(frozen Model-A score,
  frozen Model-B score), descending, then customer_id.
- Retraining occurs only after each review batch, so Round 2 is scored with
  the initial models, Round 3 after batch 1, and Round 4 after batch 2.
- Decision thresholds remain locked to the already validated frozen thresholds;
  no test labels or adaptive test outcomes are used for threshold tuning.
- Static A and frozen A+B are controls and are never modified.
- A_retrain uses behavioral features only.
- A+B_retrain uses behavioral + relational features.
- Initial training uses the original train split at day 89. Each reviewed
  validation case is appended using the feature snapshot computed as of the
  day it was actually reviewed (day 149 for the Round-2 batch, day 179 for
  the Round-3 batch). A customer's reviewed snapshot is fixed at its own
  review day and is NEVER recomputed at a later cutoff when it is carried
  into a subsequent round's training set alongside a newer batch.
- Metrics are evaluated ONLY on the fixed held-out test entities at each
  round cutoff.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from models.features import build_behavioral_features, BEHAVIORAL_FEATURE_COLUMNS
from models.features_b import build_relational_features, RELATIONAL_FEATURE_COLUMNS
from eval.metrics import compute_metrics

# Repo layout: this script lives at the repo root, alongside data/, models/,
# and eval/ as sibling directories.
DATA = ROOT / "data"                      # customers/orders/returns/products/splits.csv
MODELS = ROOT / "models"                  # frozen model_a.json / model_b.json (read-only)
MODELS_ARTIFACTS = MODELS / "artifacts"   # Phase 4's own retrained model outputs
CONFIGS = MODELS_ARTIFACTS                 # model_a_meta.json / model_b_meta.json (thresholds, costs, seed, train_as_of)
RESULTS = ROOT / "eval" / "results"       # Phase 4 metrics/audit/prediction outputs
MODELS_ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

POPULATIONS = ["normal", "hard_negative", "naive_abuse", "adaptive_abuse"]
ABUSIVE = {"naive_abuse", "adaptive_abuse"}
REVIEW_BATCH = 30

def load_data():
    return {
        "customers": pd.read_csv(DATA / "customers.csv"),
        "orders": pd.read_csv(DATA / "orders.csv"),
        "returns": pd.read_csv(DATA / "returns.csv"),
        "products": pd.read_csv(DATA / "products.csv"),
        "splits": pd.read_csv(DATA / "splits.csv"),
    }

def labels_for(customers, ids):
    c = customers.set_index("customer_id")
    return c.loc[list(ids), "population_type"].isin(ABUSIVE).astype(int).values

def customer_labels_map(customers):
    c = customers.set_index("customer_id")
    return c["population_type"].isin(ABUSIVE).astype(int).to_dict()

def build_behavioral(data, ids, day):
    f = build_behavioral_features(
        data["orders"], data["returns"], data["products"], data["customers"],
        as_of_day=day, customer_ids=set(ids)
    )
    return f

def build_combined(data, ids, day):
    beh = build_behavioral(data, ids, day)
    rel = build_relational_features(
        data["orders"], data["returns"], data["products"],
        as_of_day=day, customer_ids=set(ids)
    )
    f = beh.merge(rel, on="customer_id", how="left")
    for col in RELATIONAL_FEATURE_COLUMNS:
        f[col] = f[col].fillna(0.0)
    return f

def fit_model(X, y, seed):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed,
    )
    model.fit(X, y, verbose=False)
    return model

def predict(model, frame, cols):
    return model.predict_proba(frame[cols].fillna(0.0))[:, 1]

def population_breakdown(y_true, y_pred, pop):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "population_type": pop})
    out = {}
    for p in POPULATIONS:
        s = df[df.population_type == p]
        n = len(s)
        d = {"n": n}
        if n:
            tp = int(((s.y_pred == 1) & (s.y_true == 1)).sum())
            fp = int(((s.y_pred == 1) & (s.y_true == 0)).sum())
            fn = int(((s.y_pred == 0) & (s.y_true == 1)).sum())
            tn = int(((s.y_pred == 0) & (s.y_true == 0)).sum())
            d.update(tp=tp, fp=fp, fn=fn, tn=tn)
            if tp + fn: d["recall"] = tp / (tp + fn)
            if fp + tn: d["fpr"] = fp / (fp + tn)
        out[p] = d
    return out

def score_test(data, day, model_a, model_ab, threshold_a, threshold_ab, cost_fp, cost_fn):
    test_ids = data["splits"].loc[data["splits"].split == "test", "customer_id"].tolist()
    beh = build_behavioral(data, test_ids, day)
    comb = build_combined(data, test_ids, day)
    # Preserve exact test customer order from feature output.
    ids = beh["customer_id"].values
    y = labels_for(data["customers"], ids)
    pop = data["customers"].set_index("customer_id").loc[ids, "population_type"].values

    score_a = predict(model_a, beh, BEHAVIORAL_FEATURE_COLUMNS)
    score_ab = predict(model_ab, comb, BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS)

    return {
        "customer_id": ids, "y_true": y, "population_type": pop,
        "a": score_a, "ab": score_ab,
        "metrics_a": compute_metrics(y, score_a, threshold_a, cost_fp, cost_fn),
        "metrics_ab": compute_metrics(y, score_ab, threshold_ab, cost_fp, cost_fn),
    }

def selection_scores(data, day, frozen_a, frozen_ab):
    val_ids = data["splits"].loc[data["splits"].split == "val", "customer_id"].tolist()
    beh = build_behavioral(data, val_ids, day)
    comb = build_combined(data, val_ids, day)
    sa = predict(frozen_a, beh, BEHAVIORAL_FEATURE_COLUMNS)
    sb = predict(frozen_ab, comb, BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS)
    s = pd.DataFrame({"customer_id": beh.customer_id, "risk_a": sa, "risk_b": sb})
    s["selection_score"] = s[["risk_a","risk_b"]].max(axis=1)
    return s.sort_values(["selection_score","customer_id"], ascending=[False, True]).reset_index(drop=True)

def build_reviewed_snapshot(data, review_batches, kind):
    """Builds feature rows for reviewed validation customers, one batch at a
    time, each batch featurized at the as-of day it was actually reviewed on.
    review_batches: list of (customer_ids, review_day) tuples.

    This is the mechanism that prevents an earlier-reviewed customer from
    being silently recomputed with a later cutoff: each batch only ever sees
    its own review_day, and batches are concatenated (not re-derived from a
    single max day) when multiple batches are supplied together."""
    cols = BEHAVIORAL_FEATURE_COLUMNS if kind == "a" else BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS
    frames = []
    for ids, day in review_batches:
        if not ids:
            continue
        frame = build_behavioral(data, ids, day) if kind == "a" else build_combined(data, ids, day)
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["customer_id"] + cols)


def train_augmented(data, review_batches, base_day, kind, seed):
    """review_batches: list of (customer_ids, review_day) tuples, e.g.
    [(review_r2_ids, 149)] at Round 3, or
    [(review_r2_ids, 149), (review_r3_ids, 179)] at Round 4.
    Each batch is featurized at its own review_day -- a customer reviewed
    after Round 2 keeps its day-149 snapshot even when it is included in the
    Round 4 training set alongside the new day-179 batch."""
    train_ids = data["splits"].loc[data["splits"].split == "train", "customer_id"].tolist()
    if kind == "a":
        base = build_behavioral(data, train_ids, base_day)
        cols = BEHAVIORAL_FEATURE_COLUMNS
    else:
        base = build_combined(data, train_ids, base_day)
        cols = BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS

    extra = build_reviewed_snapshot(data, review_batches, kind)

    frames_X = [base[cols]]
    labels_list = [labels_for(data["customers"], base["customer_id"])]
    if len(extra):
        frames_X.append(extra[cols])
        labels_list.append(labels_for(data["customers"], extra["customer_id"]))

    X = pd.concat(frames_X, ignore_index=True).fillna(0.0)
    y = np.concatenate(labels_list)
    return fit_model(X, y, seed), len(X), int(y.sum()), cols

def evaluate_model_on_test(data, day, model, cols, threshold, cost_fp, cost_fn):
    test_ids = data["splits"].loc[data["splits"].split == "test", "customer_id"].tolist()
    frame = build_behavioral(data, test_ids, day) if cols == BEHAVIORAL_FEATURE_COLUMNS else build_combined(data, test_ids, day)
    y = labels_for(data["customers"], frame.customer_id)
    pop = data["customers"].set_index("customer_id").loc[frame.customer_id, "population_type"].values
    score = predict(model, frame, cols)
    m = compute_metrics(y, score, threshold, cost_fp, cost_fn)
    yp = (score >= threshold).astype(int)
    by = population_breakdown(y, yp, pop)
    return m, by, score

def main():
    data = load_data()
    meta_a = json.loads((CONFIGS / "model_a_meta.json").read_text())
    meta_b = json.loads((CONFIGS / "model_b_meta.json").read_text())
    threshold_a = meta_a["threshold_cost_optimal"]
    threshold_b = meta_b["threshold_cost_optimal"]
    cost_fp, cost_fn = meta_a["cost_fp"], meta_a["cost_fn"]
    seed = meta_a["seed"]

    # Frozen controls: load exact existing artifacts. Never overwrite them.
    frozen_a = xgb.XGBClassifier()
    frozen_a.load_model(str(MODELS_ARTIFACTS / "model_a.json"))
    frozen_b = xgb.XGBClassifier()
    frozen_b.load_model(str(MODELS_ARTIFACTS / "model_b.json"))

    rounds = {1:119, 2:149, 3:179, 4:209}

    # Initial retrain models are fresh fits on the original training split
    # (same feature families and day-89 cutoff as Model A / A+B), using this
    # script's own fit_model() hyperparameters. This guarantees an identical,
    # documented training procedure across A_retrain and A+B_retrain within
    # Phase 4 -- it is NOT a verified reproduction of whatever original
    # training/hyperparameter-selection procedure produced the frozen
    # model_a.json / model_b.json artifacts, since that procedure is not
    # available in this script. No review labels have been consumed yet.
    train_ids = data["splits"].loc[data["splits"].split == "train", "customer_id"].tolist()
    base_a = build_behavioral(data, train_ids, meta_a["train_as_of"])
    base_b = build_combined(data, train_ids, meta_b["train_as_of"])
    y_base = labels_for(data["customers"], base_a["customer_id"])
    retrain_a = fit_model(base_a[BEHAVIORAL_FEATURE_COLUMNS].fillna(0.0), y_base, seed)
    retrain_b = fit_model(base_b[BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS].fillna(0.0), y_base, seed)
    retrain_a.save_model(str(MODELS_ARTIFACTS / "A_retrain_reviewed0.json"))
    retrain_b.save_model(str(MODELS_ARTIFACTS / "AB_retrain_reviewed0.json"))

    # Common review protocol: 30 cases after R2 and 30 after R3.
    # We select using only frozen controls and validation customers.
    review_r2 = selection_scores(data, 149, frozen_a, frozen_b).head(REVIEW_BATCH)
    reviewed_so_far = set(review_r2.customer_id)
    remaining = selection_scores(data, 179, frozen_a, frozen_b)
    review_r3 = remaining[~remaining.customer_id.isin(reviewed_so_far)].head(REVIEW_BATCH)

    review_r2_ids = review_r2.customer_id.tolist()
    review_r3_ids = review_r3.customer_id.tolist()
    all_review_ids = review_r2_ids + review_r3_ids

    # Persist review audit.
    review_rows = []
    for batch, df in [(2, review_r2), (3, review_r3)]:
        for _, row in df.iterrows():
            cid = row.customer_id
            pop = data["customers"].set_index("customer_id").loc[cid, "population_type"]
            review_rows.append({
                "review_batch_after_round": batch,
                "review_day": 149 if batch == 2 else 179,
                "customer_id": cid,
                "frozen_a_risk": float(row.risk_a),
                "frozen_b_risk": float(row.risk_b),
                "selection_score": float(row.selection_score),
                "oracle_label": int(pop in ABUSIVE),
                "population_type_for_audit_only": pop,
            })
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(RESULTS / "phase4_review_audit.csv", index=False)

    # At each round:
    # R1/R2: no review labels consumed yet.
    # R3: after R2 review batch, retrain both.
    # R4: after R3 review batch, retrain again.
    # Frozen controls remain unchanged throughout.
    results = {}
    for r, day in rounds.items():
        if r == 3:
            batches = [(review_r2_ids, 149)]
            retrain_a, n_a, pos_a, cols_a = train_augmented(data, batches, meta_a["train_as_of"], "a", seed)
            retrain_b, n_b, pos_b, cols_b = train_augmented(data, batches, meta_b["train_as_of"], "b", seed)
            retrain_a.save_model(str(MODELS_ARTIFACTS / "A_retrain_reviewed30.json"))
            retrain_b.save_model(str(MODELS_ARTIFACTS / "AB_retrain_reviewed30.json"))
            retrain_state = {"reviewed_total": 30, "last_batch": 30, "train_rows_a": n_a, "train_rows_ab": n_b}
        elif r == 4:
            # The Round-2 batch keeps its day-149 snapshot; only the Round-3
            # batch is featurized at day 179. Earlier-reviewed cases are
            # never recomputed using a later cutoff.
            batches = [(review_r2_ids, 149), (review_r3_ids, 179)]
            retrain_a, n_a, pos_a, cols_a = train_augmented(data, batches, meta_a["train_as_of"], "a", seed)
            retrain_b, n_b, pos_b, cols_b = train_augmented(data, batches, meta_b["train_as_of"], "b", seed)
            retrain_a.save_model(str(MODELS_ARTIFACTS / "A_retrain_reviewed60.json"))
            retrain_b.save_model(str(MODELS_ARTIFACTS / "AB_retrain_reviewed60.json"))
            retrain_state = {"reviewed_total": 60, "last_batch": 30, "train_rows_a": n_a, "train_rows_ab": n_b}
        else:
            cols_a, cols_b = BEHAVIORAL_FEATURE_COLUMNS, BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS
            retrain_state = {"reviewed_total": 0, "last_batch": 0}

        frozen = score_test(data, day, frozen_a, frozen_b, threshold_a, threshold_b, cost_fp, cost_fn)
        ma_rt, by_a_rt, score_a_rt = evaluate_model_on_test(data, day, retrain_a, BEHAVIORAL_FEATURE_COLUMNS, threshold_a, cost_fp, cost_fn)
        mb_rt, by_b_rt, score_b_rt = evaluate_model_on_test(data, day, retrain_b, BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS, threshold_b, cost_fp, cost_fn)

        # The four variants:
        # A frozen, A retrain, A+B frozen, A+B retrain.
        results[f"round_{r}"] = {
            "as_of_day": day,
            "review_budget_used_before_scoring": retrain_state["reviewed_total"],
            "controls": {
                "A_frozen": {
                    "overall": frozen["metrics_a"],
                    "by_population": population_breakdown(frozen["y_true"], (frozen["a"] >= threshold_a).astype(int), frozen["population_type"]),
                },
                "A+B_frozen": {
                    "overall": frozen["metrics_ab"],
                    "by_population": population_breakdown(frozen["y_true"], (frozen["ab"] >= threshold_b).astype(int), frozen["population_type"]),
                },
            },
            "retrained": {
                "A_retrain": {"overall": ma_rt, "by_population": by_a_rt},
                "A+B_retrain": {"overall": mb_rt, "by_population": by_b_rt},
            },
        }

        pred = pd.DataFrame({
            "customer_id": frozen["customer_id"],
            "population_type": frozen["population_type"],
            "y_true": frozen["y_true"],
            "A_frozen_score": frozen["a"],
            "A_retrain_score": score_a_rt,
            "AB_frozen_score": frozen["ab"],
            "AB_retrain_score": score_b_rt,
        })
        pred.to_csv(RESULTS / f"phase4_predictions_round_{r}_day{day}.csv", index=False)

    # Add direct hero comparison and protocol metadata.
    out = {
        "experiment": "Phase 4 equal-review/label-budget ablation",
        "protocol": {
            "review_pool": "validation split only",
            "validation_pool_size": int((data["splits"].split == "val").sum()),
            "review_batch_size": REVIEW_BATCH,
            "review_batches": [
                {"after_round": 2, "day": 149, "labels": REVIEW_BATCH},
                {"after_round": 3, "day": 179, "labels": REVIEW_BATCH},
            ],
            "total_reviewed_labels": REVIEW_BATCH * 2,
            "budget_fraction_of_validation_pool": (REVIEW_BATCH * 2) / int((data["splits"].split == "val").sum()),
            "same_customer_ids_for_A_and_AB_retraining": True,
            "selection": "Common deterministic queue from frozen A/B: max(risk_A, risk_B), excluding previously reviewed cases; tie-break customer_id.",
            "label_source": "oracle true_abuse_label for this controlled Phase-4 experiment",
            "final_test_touched": False,
            "thresholds_locked": {"A": threshold_a, "A+B": threshold_b},
            "costs": {"cost_fp": cost_fp, "cost_fn": cost_fn},
            "frozen_controls_unchanged": True,
            "entity_temporal_controls": "existing entity-aware split; as-of feature construction at each round cutoff; no final-test labels or rows used for retraining",
            "paths": {
                "data_dir": str(DATA),
                "models_dir": str(MODELS),
                "models_artifacts_dir": str(MODELS_ARTIFACTS),
                "configs_dir": str(CONFIGS),
                "results_dir": str(RESULTS),
            },
        },
        "review_audit_counts": {
            "batch_after_round_2": len(review_r2_ids),
            "batch_after_round_3": len(review_r3_ids),
            "unique_total": len(set(all_review_ids)),
        },
        "rounds": results,
        "hero_comparison": {},
    }
    for r in rounds:
        rr = results[f"round_{r}"]
        def compact(x):
            return {k: x[k] for k in ["precision","recall","f1","fpr","fp","fn","fp_cost","fn_cost","total_cost"]}
        out["hero_comparison"][f"round_{r}"] = {
            "A_frozen": compact(rr["controls"]["A_frozen"]["overall"]),
            "A_retrain": compact(rr["retrained"]["A_retrain"]["overall"]),
            "A+B_frozen": compact(rr["controls"]["A+B_frozen"]["overall"]),
            "A+B_retrain": compact(rr["retrained"]["A+B_retrain"]["overall"]),
            "adaptive_abuse_recall": {
                "A_frozen": rr["controls"]["A_frozen"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                "A_retrain": rr["retrained"]["A_retrain"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                "A+B_frozen": rr["controls"]["A+B_frozen"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                "A+B_retrain": rr["retrained"]["A+B_retrain"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
            },
            "hard_negative_fpr": {
                "A_frozen": rr["controls"]["A_frozen"]["by_population"]["hard_negative"].get("fpr", 0.0),
                "A_retrain": rr["retrained"]["A_retrain"]["by_population"]["hard_negative"].get("fpr", 0.0),
                "A+B_frozen": rr["controls"]["A+B_frozen"]["by_population"]["hard_negative"].get("fpr", 0.0),
                "A+B_retrain": rr["retrained"]["A+B_retrain"]["by_population"]["hard_negative"].get("fpr", 0.0),
            }
        }
    (RESULTS / "phase4_equal_budget_ablation_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"protocol": out["protocol"], "hero_comparison": out["hero_comparison"]}, indent=2))

if __name__ == "__main__":
    main()
