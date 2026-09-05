"""
Train Model B = behavioral (Model A's 14 features) + relational (9 new
features from features_b.py), on the SAME train-as-of-day-89 /
val-as-of-day-104 protocol as Model A, using the SAME entity-aware split
and the SAME cost_fp/cost_fn as Model A's frozen meta.

This keeps the comparison to Model A fair: same customers, same training
cutoff, same cost assumptions. The only thing that changes is the feature
set. Model A itself is NOT touched/retrained here.

Run:
    python models/model_b.py --data data/ --config configs/config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_behavioral_features, BEHAVIORAL_FEATURE_COLUMNS
from models.features_b import build_relational_features, RELATIONAL_FEATURE_COLUMNS
from data_gen.schema import ABUSIVE_POPULATIONS
from eval.metrics import best_threshold_by_cost, supplementary_metrics

FEATURE_COLUMNS_B = BEHAVIORAL_FEATURE_COLUMNS + RELATIONAL_FEATURE_COLUMNS


def load_data(data_dir):
    data_dir = Path(data_dir)
    return {
        "customers": pd.read_csv(data_dir / "customers.csv"),
        "orders": pd.read_csv(data_dir / "orders.csv"),
        "returns": pd.read_csv(data_dir / "returns.csv"),
        "products": pd.read_csv(data_dir / "products.csv"),
        "splits": pd.read_csv(data_dir / "splits.csv"),
    }


def labels_for(customers_df, customer_ids):
    cust = customers_df.set_index("customer_id")
    return cust.loc[customer_ids, "population_type"].isin(ABUSIVE_POPULATIONS).astype(int).values


def build_full_features(data, as_of_day, customer_ids):
    beh = build_behavioral_features(
        data["orders"], data["returns"], data["products"], data["customers"],
        as_of_day=as_of_day, customer_ids=set(customer_ids),
    )
    rel = build_relational_features(
        data["orders"], data["returns"], data["products"],
        as_of_day=as_of_day, customer_ids=set(customer_ids),
    )
    merged = beh.merge(rel, on="customer_id", how="left")
    for c in RELATIONAL_FEATURE_COLUMNS:
        merged[c] = merged[c].fillna(0.0)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = load_data(args.data)

    meta_a = json.load(open("models/artifacts/model_a_meta.json"))
    train_as_of, val_as_of = meta_a["train_as_of"], meta_a["val_as_of"]
    cost_fp, cost_fn = meta_a["cost_fp"], meta_a["cost_fn"]

    train_customers = data["splits"].loc[data["splits"]["split"] == "train", "customer_id"]
    val_customers = data["splits"].loc[data["splits"]["split"] == "val", "customer_id"]

    print(f"=== Model B training (behavioral + relational, {len(FEATURE_COLUMNS_B)} features) ===")
    print(f"train_as_of={train_as_of}  val_as_of={val_as_of}  (same as Model A)")
    print(f"cost_fp={cost_fp}  cost_fn={cost_fn:.4f}  (reused from Model A meta -- same budget definition)")

    train_feats = build_full_features(data, train_as_of, train_customers)
    val_feats = build_full_features(data, val_as_of, val_customers)

    y_train = labels_for(data["customers"], train_feats["customer_id"].values)
    y_val = labels_for(data["customers"], val_feats["customer_id"].values)

    X_train = train_feats[FEATURE_COLUMNS_B].fillna(0.0)
    X_val = val_feats[FEATURE_COLUMNS_B].fillna(0.0)

    print(f"train: n={len(X_train)} positives={y_train.sum()} ({y_train.mean():.3f})")
    print(f"val:   n={len(X_val)} positives={y_val.sum()} ({y_val.mean():.3f})")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS_B)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS_B)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": cfg["seed"],
    }
    evals_result = {}
    booster = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30, evals_result=evals_result, verbose_eval=False,
    )
    best_iter = booster.best_iteration
    print(f"trees used (early-stopped): {best_iter + 1} / 500  "
          f"(val logloss={evals_result['val']['logloss'][best_iter]:.4f})")

    val_scores = booster.predict(dval, iteration_range=(0, best_iter + 1))
    train_scores = booster.predict(dtrain, iteration_range=(0, best_iter + 1))

    print("\n=== validation supplementary metrics ===")
    print(supplementary_metrics(y_val, val_scores))

    best = best_threshold_by_cost(y_val, val_scores, cost_fp, cost_fn)
    print("\n=== cost-optimal threshold selected on VALIDATION (day", val_as_of, ") ===")
    print(f"threshold={best['threshold']:.10f}  total_cost={best['total_cost']:.2f}  "
          f"precision={best['precision']:.3f} recall={best['recall']:.3f}")

    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    booster.save_model("models/artifacts/model_b.json")

    meta = {
        "feature_columns": FEATURE_COLUMNS_B,
        "behavioral_feature_columns": BEHAVIORAL_FEATURE_COLUMNS,
        "relational_feature_columns": RELATIONAL_FEATURE_COLUMNS,
        "train_as_of": train_as_of,
        "val_as_of": val_as_of,
        "threshold_cost_optimal": best["threshold"],
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "seed": cfg["seed"],
        "xgb_params": params,
        "num_boost_round_used": best_iter + 1,
        "val_supplementary_metrics": supplementary_metrics(y_val, val_scores),
        "note": "Same train/val cutoffs, split, and cost budget as Model A. "
                "Only the feature set differs (behavioral+relational vs behavioral-only).",
    }
    with open("models/artifacts/model_b_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nsaved -> models/artifacts/model_b.json, models/artifacts/model_b_meta.json")


if __name__ == "__main__":
    main()
