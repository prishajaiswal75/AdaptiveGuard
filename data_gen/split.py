"""
Entity-aware split for AdaptiveGuard.

Rule: a connected component of customers who share a device/address/payment
fingerprint (a ring or household) must land ENTIRELY in one of
train/val/test. Splitting individual ring members across sets would leak
graph structure (a held-out ring member's features would depend on
same-ring customers seen during training).

Temporal boundaries (which round's rows count as train/val/test-in-time vs
out-of-time) are applied separately, downstream, by filtering on
orders/returns `round` -- this module only decides customer-level set
membership.

Run:
    python data_gen/split.py --data data/ --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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


def build_clusters(customers_df):
    uf = UnionFind()
    for col in ("device_id", "address_id", "payment_fp_id"):
        for _, grp in customers_df.groupby(col):
            ids = grp["customer_id"].tolist()
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])
    cluster_of = {c: uf.find(c) for c in customers_df["customer_id"]}
    return cluster_of


def assign_splits(customers_df, cluster_of, test_frac, val_frac, seed):
    df = customers_df.copy()
    df["cluster_id"] = df["customer_id"].map(cluster_of)

    # dominant population per cluster, for stratified assignment
    dominant_pop = (
        df.groupby("cluster_id")["population_type"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    cluster_size = df.groupby("cluster_id").size()

    rng = np.random.default_rng(seed)
    split_of_cluster = {}

    for pop, clusters in dominant_pop.groupby(dominant_pop):
        cluster_ids = list(clusters.index)
        rng.shuffle(cluster_ids)
        sizes = cluster_size.loc[cluster_ids].values
        total = sizes.sum()
        cum = np.cumsum(sizes)
        test_cut = total * test_frac
        val_cut = total * (test_frac + val_frac)
        for cid, c in zip(cluster_ids, cum):
            if c <= test_cut:
                split_of_cluster[cid] = "test"
            elif c <= val_cut:
                split_of_cluster[cid] = "val"
            else:
                split_of_cluster[cid] = "train"

    df["split"] = df["cluster_id"].map(split_of_cluster)
    return df[["customer_id", "cluster_id", "split"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data)
    customers_df = pd.read_csv(data_dir / "customers.csv")

    cluster_of = build_clusters(customers_df)
    splits_df = assign_splits(
        customers_df, cluster_of,
        cfg["split"]["test_fraction"], cfg["split"]["val_fraction"], cfg["seed"],
    )
    splits_df.to_csv(data_dir / "splits.csv", index=False)

    # ---- validation ----
    merged = splits_df.merge(customers_df[["customer_id", "population_type"]], on="customer_id")
    print("split sizes (customers):")
    print(merged["split"].value_counts())
    print("\npopulation balance by split (should be roughly proportional):")
    print(pd.crosstab(merged["population_type"], merged["split"], normalize="columns").round(3))

    leak_check = merged.groupby("cluster_id")["split"] if False else None
    per_cluster_splits = splits_df.groupby("cluster_id")["split"].nunique()
    n_leaky = (per_cluster_splits > 1).sum()
    print(f"\nclusters spanning >1 split (must be 0): {n_leaky}")


if __name__ == "__main__":
    main()
