"""
Phase 3: Frozen Model B (behavioral + relational) under the SAME adaptive
distribution shift used in Phase 2 for Model A. Does NOT retrain. Does NOT
retune the threshold on rounds 2-4. Uses the exact same held-out TEST-split
customers as Phase 1/2.

Mirrors eval/phase2_adaptive_eval.py's structure/output exactly so the two
are directly comparable round-by-round.

Also runs an explicit leakage check at each cutoff: confirms the
order-observed identifier graph used for Model B's relational features
never connects a TEST customer to a customer in a different split.

Run:
    python eval/phase3_model_b_eval.py --data data/ --config configs/config.yaml
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
from models.features_b import build_relational_features, RELATIONAL_FEATURE_COLUMNS
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


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def leakage_check_split_boundary(orders_df, splits_df, as_of_day):
    """For orders up to as_of_day, verifies the order-observed identifier
    graph (device/address/payment_fp) never connects two customers from
    different train/val/test splits. Returns (n_leaky_clusters, n_clusters)."""
    o = orders_df[orders_df.order_day <= as_of_day]
    uf = UnionFind()
    for c in o["customer_id"].unique():
        uf.find(c)
    for col in ["device_id", "address_id", "payment_fp_id"]:
        for _, members in o.groupby(col)["customer_id"]:
            members = members.unique().tolist()
            for i in range(1, len(members)):
                uf.union(members[0], members[i])
    active = list(uf.parent.keys())
    split_map = splits_df.set_index("customer_id")["split"]
    df = pd.DataFrame({"customer_id": active})
    df["cluster"] = df["customer_id"].map(lambda c: uf.find(c))
    df["split"] = df["customer_id"].map(split_map)
    n_per_cluster = df.groupby("cluster")["split"].nunique()
    n_leaky = int((n_per_cluster > 1).sum())
    return n_leaky, len(n_per_cluster)


def score_test_split_at(data, feature_columns, booster, as_of_day):
    test_customers = data["splits"].loc[data["splits"]["split"] == "test", "customer_id"]
    beh = build_behavioral_features(
        data["orders"], data["returns"], data["products"], data["customers"],
        as_of_day=as_of_day, customer_ids=set(test_customers),
    )
    rel = build_relational_features(
        data["orders"], data["returns"], data["products"],
        as_of_day=as_of_day, customer_ids=set(test_customers),
    )
    feats = beh.merge(rel, on="customer_id", how="left")
    for c in RELATIONAL_FEATURE_COLUMNS:
        feats[c] = feats[c].fillna(0.0)
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

    n_leaky, n_clusters = leakage_check_split_boundary(data["orders"], data["splits"], as_of_day)

    return {
        "label": label, "as_of_day": as_of_day,
        "n": len(y_true), "n_positive": int(y_true.sum()),
        "overall": overall, "by_population": by_pop,
        "leakage_check": {"n_clusters_spanning_splits": n_leaky, "n_clusters_total": n_clusters},
        "_scored": scored,
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

    meta = json.load(open("models/artifacts/model_b_meta.json"))
    threshold = meta["threshold_cost_optimal"]
    cost_fp = meta["cost_fp"]
    cost_fn = meta["cost_fn"]
    feature_columns = meta["feature_columns"]

    booster = xgb.Booster()
    booster.load_model("models/artifacts/model_b.json")

    print("=== FROZEN MODEL B (behavioral + relational, not retrained / not retuned) ===")
    print("model      = models/artifacts/model_b.json")
    print(f"n_features = {len(feature_columns)}  threshold = {threshold}")
    print(f"cost_fp    = {cost_fp}   cost_fn = {cost_fn:.4f}  (identical budget to Model A)")

    round_cutoffs = [
        ("round_1", rounds[1]["end"]),
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
            pred_df.to_csv(f"eval/results/phase3_predictions_{label}_day{as_of_day}.csv", index=False)

        m = res["overall"]
        lk = res["leakage_check"]
        print(f"\n--- {label} (as-of day {as_of_day}, n={res['n']}, positives={res['n_positive']}) ---")
        print(f"  leakage check: {lk['n_clusters_spanning_splits']} / {lk['n_clusters_total']} clusters "
              f"span >1 split (must be 0)")
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

    # ---- side-by-side vs Model A (Phase 2) ----
    phase2 = json.load(open("eval/results/phase2_adaptive_eval_metrics.json"))
    a_rounds = {
        "round_1": phase2["rounds"]["round_1_resc"],
        "round_2": phase2["rounds"]["round_2"],
        "round_3": phase2["rounds"]["round_3"],
        "round_4": phase2["rounds"]["round_4"],
    }

    print("\n=== adaptive_abuse recall: Model A (frozen behavioral) vs Model B (frozen behavioral+relational) ===")
    for label, as_of_day in round_cutoffs:
        a_rec = a_rounds[label]["by_population"]["adaptive_abuse"].get("recall")
        b_rec = results[label]["by_population"]["adaptive_abuse"].get("recall")
        delta = None if (a_rec is None or b_rec is None) else b_rec - a_rec
        print(f"  {label} (day {as_of_day}): A={a_rec:.3f}  B={b_rec:.3f}  delta(B-A)={delta:+.3f}")

    print("\n=== hard_negative FPR: Model A vs Model B (should both stay low) ===")
    for label, as_of_day in round_cutoffs:
        a_fpr = a_rounds[label]["by_population"]["hard_negative"].get("fpr", 0.0)
        b_fpr = results[label]["by_population"]["hard_negative"].get("fpr", 0.0)
        print(f"  {label} (day {as_of_day}): A={a_fpr:.4f}  B={b_fpr:.4f}")

    print("\n=== total_cost: Model A vs Model B ===")
    for label, as_of_day in round_cutoffs:
        a_cost = a_rounds[label]["overall"]["total_cost"]
        b_cost = results[label]["overall"]["total_cost"]
        print(f"  {label} (day {as_of_day}): A={a_cost:.1f}  B={b_cost:.1f}  delta(B-A)={b_cost - a_cost:+.1f}")

    out = {
        "frozen_artifacts": {
            "model_path": "models/artifacts/model_b.json",
            "threshold_cost_optimal": threshold,
            "cost_fp": cost_fp, "cost_fn": cost_fn,
            "feature_columns": feature_columns,
        },
        "rounds": results,
        "comparison_vs_model_a": {
            label: {
                "adaptive_abuse_recall_A": a_rounds[label]["by_population"]["adaptive_abuse"].get("recall"),
                "adaptive_abuse_recall_B": results[label]["by_population"]["adaptive_abuse"].get("recall"),
                "hard_negative_fpr_A": a_rounds[label]["by_population"]["hard_negative"].get("fpr", 0.0),
                "hard_negative_fpr_B": results[label]["by_population"]["hard_negative"].get("fpr", 0.0),
                "total_cost_A": a_rounds[label]["overall"]["total_cost"],
                "total_cost_B": results[label]["overall"]["total_cost"],
            }
            for label, _ in round_cutoffs
        },
    }
    out_path = Path("eval/results/phase3_model_b_eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
