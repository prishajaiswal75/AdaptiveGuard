"""
Phase 2: Frozen Model A under adaptive distribution shift (out-of-time).

Loads the FROZEN Model A artifact (models/artifacts/model_a.json) and the
Phase 1 validation-selected threshold/cost (models/artifacts/model_a_meta.json).
Does NOT retrain. Does NOT retune the threshold on rounds 2-4. Reuses the
exact same as-of feature construction (models/features.py) as Phase 1.

Scores the held-out TEST-split customers (entity-aware split, fixed since
generation) as of the END of round 1 (day 119, re-scored only as a sanity
check), round 2 (day 149), round 3 (day 179), and round 4 (day 209).

The round-1 re-score must reproduce eval/results/phase1_model_a_test_metrics.json
exactly (same customers, same cutoff, same threshold, same costs) -- this
is a leakage/regression sanity check, not a new modeling decision.

Run:
    python eval/phase2_adaptive_eval.py --data data/ --config configs/config.yaml
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
from models.features import build_behavioral_features
from data_gen.schema import ABUSIVE_POPULATIONS
from eval.metrics import compute_metrics

POPULATIONS = ["normal", "hard_negative", "naive_abuse", "adaptive_abuse"]


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


def score_test_split_at(data, feature_columns, booster, as_of_day):
    """Build features for the FIXED test-split customers as of as_of_day and
    score them with the frozen booster. No fitting/tuning happens here."""
    test_customers = data["splits"].loc[data["splits"]["split"] == "test", "customer_id"]
    feats = build_behavioral_features(
        data["orders"], data["returns"], data["products"], data["customers"],
        as_of_day=as_of_day, customer_ids=set(test_customers),
    )
    if len(feats) == 0:
        return None

    cust_ids = feats["customer_id"].values
    X = feats[feature_columns].fillna(0.0)
    y = labels_for(data["customers"], cust_ids)

    dmat = xgb.DMatrix(X, feature_names=feature_columns)
    scores = booster.predict(dmat)

    pop = data["customers"].set_index("customer_id").loc[cust_ids, "population_type"].values
    return {"customer_id": cust_ids, "y_true": y, "y_score": scores, "population_type": pop}


def population_breakdown(y_true, y_pred, population_type):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "population_type": population_type})
    out = {}
    for pop in POPULATIONS:
        sub = df[df.population_type == pop]
        n = len(sub)
        entry = {"n": n}
        if n > 0:
            tp = int(((sub.y_pred == 1) & (sub.y_true == 1)).sum())
            fp = int(((sub.y_pred == 1) & (sub.y_true == 0)).sum())
            fn = int(((sub.y_pred == 0) & (sub.y_true == 1)).sum())
            tn = int(((sub.y_pred == 0) & (sub.y_true == 0)).sum())
            entry.update(tp=tp, fp=fp, fn=fn, tn=tn)
            if tp + fn > 0:
                entry["recall"] = tp / (tp + fn)
            if fp + tn > 0:
                entry["fpr"] = fp / (fp + tn)
        out[pop] = entry
    return out


def evaluate_round(label, as_of_day, data, feature_columns, booster, cost_fp, cost_fn, threshold):
    scored = score_test_split_at(data, feature_columns, booster, as_of_day)
    if scored is None:
        return {"label": label, "as_of_day": as_of_day, "n": 0, "error": "no active test customers"}

    y_true, y_score = scored["y_true"], scored["y_score"]
    overall = compute_metrics(y_true, y_score, threshold, cost_fp, cost_fn)
    y_pred = (y_score >= threshold).astype(int)
    by_pop = population_breakdown(y_true, y_pred, scored["population_type"])

    return {
        "label": label, "as_of_day": as_of_day,
        "n": len(y_true), "n_positive": int(y_true.sum()),
        "overall": overall, "by_population": by_pop,
        "_scored": scored,  # dropped before json dump, used only for prediction CSVs
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = load_data(args.data)
    rounds = cfg["simulation"]["rounds"]

    meta = json.load(open("models/artifacts/model_a_meta.json"))
    threshold = meta["threshold_cost_optimal"]
    cost_fp = meta["cost_fp"]
    cost_fn = meta["cost_fn"]
    feature_columns = meta["feature_columns"]

    booster = xgb.Booster()
    booster.load_model("models/artifacts/model_a.json")

    print("=== FROZEN ARTIFACTS (not retrained / not retuned) ===")
    print("model      = models/artifacts/model_a.json")
    print(f"threshold  = {threshold}")
    print(f"cost_fp    = {cost_fp}   cost_fn = {cost_fn:.4f}")

    round_cutoffs = [
        ("round_1_resc", rounds[1]["end"]),  # sanity check vs Phase 1 saved metrics
        ("round_2", rounds[2]["end"]),
        ("round_3", rounds[3]["end"]),
        ("round_4", rounds[4]["end"]),
    ]

    results = {}
    for label, as_of_day in round_cutoffs:
        res = evaluate_round(label, as_of_day, data, feature_columns, booster, cost_fp, cost_fn, threshold)
        results[label] = res
        scored = res.pop("_scored", None)

        if scored is not None:
            pred_df = pd.DataFrame({
                "customer_id": scored["customer_id"],
                "population_type": scored["population_type"],
                "y_true": scored["y_true"],
                "y_score": scored["y_score"],
                "y_pred": (scored["y_score"] >= threshold).astype(int),
            })
            pred_df.to_csv(f"eval/results/phase2_predictions_{label}_day{as_of_day}.csv", index=False)

        m = res["overall"]
        print(f"\n--- {label} (as-of day {as_of_day}, n={res['n']}, positives={res['n_positive']}) ---")
        print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} fpr={m['fpr']:.3f}")
        print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
        print(f"  fp_cost={m['fp_cost']:.1f} fn_cost={m['fn_cost']:.1f} total_cost={m['total_cost']:.1f}")
        for pop, entry in res["by_population"].items():
            extra = ""
            if "recall" in entry:
                extra += f" recall={entry['recall']:.3f}"
            if "fpr" in entry:
                extra += f" fpr={entry['fpr']:.3f}"
            print(f"    [{pop}] n={entry['n']}{extra}")

    # ---- sanity check: round_1_resc must match the Phase 1 saved metrics ----
    phase1 = json.load(open("eval/results/phase1_model_a_test_metrics.json"))
    p1 = phase1["threshold_cost_optimal"]
    r1 = results["round_1_resc"]["overall"]
    sanity_ok = (p1["tp"] == r1["tp"] and p1["fp"] == r1["fp"]
                 and p1["fn"] == r1["fn"] and p1["tn"] == r1["tn"])
    print("\n=== SANITY CHECK vs Phase 1 saved metrics (round-1, day 119) ===")
    print(f"  phase1 saved : tp={p1['tp']} fp={p1['fp']} fn={p1['fn']} tn={p1['tn']}")
    print(f"  phase2 resc. : tp={r1['tp']} fp={r1['fp']} fn={r1['fn']} tn={r1['tn']}")
    print(f"  MATCH = {sanity_ok}")

    # ---- adaptive_abuse recall degradation ----
    print("\n=== adaptive_abuse recall across rounds ===")
    baseline_recall = results["round_1_resc"]["by_population"]["adaptive_abuse"].get("recall")
    print(f"  round_1 (day 119, baseline) recall = {baseline_recall}")
    for label in ("round_2", "round_3", "round_4"):
        rec = results[label]["by_population"]["adaptive_abuse"].get("recall")
        as_of = results[label]["as_of_day"]
        delta = None if (rec is None or baseline_recall is None) else rec - baseline_recall
        print(f"  {label} (day {as_of}) recall = {rec}  delta_vs_round1 = {delta}")

    # ---- naive_abuse recall across rounds (contrast population) ----
    print("\n=== naive_abuse recall across rounds (should stay high/stable) ===")
    for label, as_of_day in round_cutoffs:
        rec = results[label]["by_population"]["naive_abuse"].get("recall")
        print(f"  {label} (day {as_of_day}) recall = {rec}")

    # ---- hard_negative FPR across rounds (does the model start crying wolf on hard negatives?) ----
    print("\n=== hard_negative FPR across rounds ===")
    for label, as_of_day in round_cutoffs:
        fpr = results[label]["by_population"]["hard_negative"].get("fpr")
        print(f"  {label} (day {as_of_day}) fpr = {fpr}")

    # ---- total_cost comparison ----
    print("\n=== total_cost across rounds ===")
    for label, as_of_day in round_cutoffs:
        print(f"  {label} (day {as_of_day}) total_cost = {results[label]['overall']['total_cost']:.1f}")

    out = {
        "frozen_artifacts": {
            "model_path": "models/artifacts/model_a.json",
            "threshold_cost_optimal": threshold,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "feature_columns": feature_columns,
        },
        "sanity_check_round1_matches_phase1": bool(sanity_ok),
        "phase1_round1_baseline": p1,
        "rounds": results,
    }
    out_path = Path("eval/results/phase2_adaptive_eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
