"""
Phase 1: Model A -- exposed behavioral risk model.

Train:  train-split customers, features as-of end of round 0 (day 89).
Val:    val-split customers,   features as-of end of round-1 first half (day 104).
        Used ONLY to pick the cost-minimizing decision threshold.
Test:   test-split customers,  features as-of end of round 1 (day 119).
        Held-out, in-distribution result -- this is the number that goes in
        the README/pitch as "Model A's real held-out performance".

Round 2-4 (out-of-time, adaptive) evaluation is Phase 2, not here -- this
script only produces the static, in-distribution baseline.

Run:
    python models/model_a.py --data data/ --config configs/config.yaml
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
from data_gen.schema import ABUSIVE_POPULATIONS
from eval.metrics import best_threshold_by_cost, compute_metrics, supplementary_metrics


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


def build_xy(data, split_name, as_of_day):
    split_customers = data["splits"].loc[data["splits"]["split"] == split_name, "customer_id"]
    feats = build_behavioral_features(
        data["orders"], data["returns"], data["products"], data["customers"],
        as_of_day=as_of_day, customer_ids=set(split_customers),
    )
    y = labels_for(data["customers"], feats["customer_id"].values)
    X = feats[BEHAVIORAL_FEATURE_COLUMNS].fillna(0.0)
    return X, y, feats["customer_id"].values


def derive_cost_fn(data, train_customers, as_of_day, cfg):
    if not cfg["costs"]["derive_cost_fn_from_train_refunds"]:
        return cfg["costs"]["cost_fn_fallback"]
    r = data["returns"]
    r = r[(r.customer_id.isin(train_customers)) & (r.return_day <= as_of_day) & (r.true_abuse_label == 1)]
    if len(r) == 0:
        return cfg["costs"]["cost_fn_fallback"]
    return float(r["refund_amount"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = load_data(args.data)
    rounds = cfg["simulation"]["rounds"]

    train_as_of = rounds[0]["end"]                                   # day 89
    val_as_of = rounds[1]["start"] + (rounds[1]["end"] - rounds[1]["start"]) // 2   # day 104
    test_as_of = rounds[1]["end"]                                    # day 119

    X_train, y_train, train_ids = build_xy(data, "train", train_as_of)
    X_val, y_val, val_ids = build_xy(data, "val", val_as_of)
    X_test, y_test, test_ids = build_xy(data, "test", test_as_of)

    print(f"train n={len(X_train)} (positives={y_train.sum()}, "
          f"{y_train.mean():.1%})  [as-of day {train_as_of}]")
    print(f"val   n={len(X_val)} (positives={y_val.sum()}, "
          f"{y_val.mean():.1%})  [as-of day {val_as_of}]")
    print(f"test  n={len(X_test)} (positives={y_test.sum()}, "
          f"{y_test.mean():.1%})  [as-of day {test_as_of}]")

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=cfg["seed"],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    cost_fp = cfg["costs"]["cost_fp"]
    cost_fn = derive_cost_fn(data, data["splits"].loc[data["splits"].split == "train", "customer_id"],
                              train_as_of, cfg)
    print(f"\ncost_fp={cost_fp} cost_fn={cost_fn:.2f} "
          f"(derived from train-split abusive refunds)" if cfg["costs"]["derive_cost_fn_from_train_refunds"]
          else f"\ncost_fp={cost_fp} cost_fn={cost_fn:.2f} (fallback constant)")

    best_val = best_threshold_by_cost(y_val, val_scores, cost_fp, cost_fn)
    threshold = best_val["threshold"]

    fixed_test = compute_metrics(y_test, test_scores, 0.5, cost_fp, cost_fn)
    cost_opt_test = compute_metrics(y_test, test_scores, threshold, cost_fp, cost_fn)
    supp = supplementary_metrics(y_test, test_scores)

    def show(label, m):
        print(f"\n[{label}] threshold={m['threshold']:.3f}")
        print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  fpr={m['fpr']:.3f}")
        print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
        print(f"  fp_cost={m['fp_cost']:.1f}  fn_cost={m['fn_cost']:.1f}  "
              f"total_cost={m['total_cost']:.1f}")

    print("\n=== HELD-OUT TEST RESULTS (round-1, in-distribution, never-seen customers) ===")
    show("threshold=0.5 (naive)", fixed_test)
    show("threshold=cost-optimal (tuned on val)", cost_opt_test)
    print(f"\n[supplementary] AUC={supp['auc']:.3f}  PR-AUC={supp['pr_auc']:.3f}")

    # persist artifacts for Phase 2 (out-of-time scoring) and Phase 4 (ablation)
    art_dir = Path("models/artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(art_dir / "model_a.json"))

    meta = {
        "feature_columns": BEHAVIORAL_FEATURE_COLUMNS,
        "train_as_of": train_as_of, "val_as_of": val_as_of, "test_as_of": test_as_of,
        "threshold_cost_optimal": threshold,
        "cost_fp": cost_fp, "cost_fn": cost_fn,
        "seed": cfg["seed"],
    }
    with open(art_dir / "model_a_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "phase1_model_a_test_metrics.json", "w") as f:
        json.dump({
            "threshold_0.5": fixed_test, "threshold_cost_optimal": cost_opt_test,
            "supplementary": supp,
        }, f, indent=2)

    print(f"\nsaved model -> {art_dir/'model_a.json'}")
    print(f"saved metrics -> {results_dir/'phase1_model_a_test_metrics.json'}")


if __name__ == "__main__":
    main()
