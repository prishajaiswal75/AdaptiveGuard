"""
Model B relational/contextual feature engineering.

Adds identifier-sharing / graph / merchant-deviation / burst signals on top
of (not instead of) Model A's 14 behavioral features. Model B = behavioral +
relational, trained on the SAME train/val cutoffs as Model A, so that any
difference vs. Model A isolates the value of the relational view rather than
extra data or a different training window.

CRITICAL DESIGN CHOICE (documented, not empirical fact):
The identifier graph (device_id / address_id / payment_fp_id sharing) is
built from ORDER-OBSERVED identifiers up to the as-of cutoff, NOT from the
static customers.csv identifier columns. This is deliberate:
  1. It reflects what a real system would actually observe over time.
  2. It captures round-4 "identity churn" (rings dropping shared IDs on some
     orders) as a genuine degradation of the relational signal, rather than
     hiding it behind ground-truth static identifiers.
Verified empirically (see phase3 leakage-check output) that this graph never
connects two customers who fall in different train/val/test splits, even
using the full 0-209 day horizon. So per-cutoff relational features cannot
leak label information across the entity-aware split.

All aggregates (cluster stats, merchant/category baselines) are computed
using ONLY orders/returns with day <= as_of_day.
"""
import numpy as np
import pandas as pd

# ---- design constants (hand-picked, not tuned/fit) ----
SUSPICIOUS_RETURN_RATE_THRESHOLD = 0.30  # upper bound of config.yaml behavior.return_rate.normal
BURST_HALF_LIFE_DAYS = 14.0              # recency half-life for the burst/decay score

RELATIONAL_FEATURE_COLUMNS = [
    "device_sharing_count",
    "address_sharing_count",
    "payment_fp_sharing_count",
    "identifier_cluster_size",
    "cluster_return_rate",
    "suspicious_neighbor_ratio",
    "merchant_relative_deviation",
    "category_relative_deviation",
    "recent_burst_score",
]


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


def _customer_return_rates(o, r):
    """Per-customer return_rate = n_returns / n_orders, using only rows
    already filtered to day <= as_of_day by the caller."""
    n_orders = o.groupby("customer_id").size()
    n_returns = r.groupby("customer_id").size() if len(r) else pd.Series(dtype=float)
    rr = (n_returns.reindex(n_orders.index).fillna(0.0)) / n_orders
    return rr  # indexed by customer_id


def _identifier_graph_stats(o):
    """Builds the order-observed identifier graph (device/address/payment_fp)
    over customers active as of the cutoff and returns per-identifier-type
    sharing counts plus overall connected-component id and size."""
    active = sorted(o["customer_id"].unique())
    uf = UnionFind()
    for c in active:
        uf.find(c)  # ensure isolated customers register as their own component

    sharing_counts = {col: {} for col in ["device_id", "address_id", "payment_fp_id"]}
    for col in ["device_id", "address_id", "payment_fp_id"]:
        groups = o.groupby(col)["customer_id"].unique()
        for id_val, members in groups.items():
            members = list(members)
            if len(members) > 1:
                for i in range(1, len(members)):
                    uf.union(members[0], members[i])
            # sharing count for this identifier type: (#others who ever used this id)
            for m in members:
                others = len(members) - 1
                sharing_counts[col][m] = sharing_counts[col].get(m, 0) + others

    cluster_of = {c: uf.find(c) for c in active}
    cluster_sizes = pd.Series(cluster_of).value_counts()

    out = pd.DataFrame(index=active)
    out["device_sharing_count"] = pd.Series(sharing_counts["device_id"]).reindex(active).fillna(0).astype(int)
    out["address_sharing_count"] = pd.Series(sharing_counts["address_id"]).reindex(active).fillna(0).astype(int)
    out["payment_fp_sharing_count"] = pd.Series(sharing_counts["payment_fp_id"]).reindex(active).fillna(0).astype(int)
    out["_cluster_id"] = pd.Series(cluster_of).reindex(active)
    out["identifier_cluster_size"] = out["_cluster_id"].map(cluster_sizes).astype(int)
    return out


def _cluster_neighbor_stats(graph_df, return_rate):
    """cluster_return_rate and suspicious_neighbor_ratio, both leave-self-out
    means over the customer's connected component (0.0 if no neighbors)."""
    df = graph_df[["_cluster_id"]].copy()
    df["return_rate"] = return_rate.reindex(df.index).fillna(0.0)
    df["is_suspicious"] = (df["return_rate"] > SUSPICIOUS_RETURN_RATE_THRESHOLD).astype(float)

    grp = df.groupby("_cluster_id")
    cluster_sum_rr = grp["return_rate"].transform("sum")
    cluster_sum_sus = grp["is_suspicious"].transform("sum")
    cluster_n = grp["return_rate"].transform("count")

    others_n = (cluster_n - 1).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_return_rate = np.where(others_n > 0, (cluster_sum_rr - df["return_rate"]) / others_n, 0.0)
        suspicious_neighbor_ratio = np.where(others_n > 0, (cluster_sum_sus - df["is_suspicious"]) / others_n, 0.0)

    return (
        pd.Series(cluster_return_rate, index=df.index, name="cluster_return_rate"),
        pd.Series(suspicious_neighbor_ratio, index=df.index, name="suspicious_neighbor_ratio"),
    )


def _relative_deviation(o, r, group_col, customer_return_rate):
    """Leave-one-customer-out baseline return rate for each group value
    (merchant or category), then a per-customer order-count-weighted
    deviation of their own return rate from the groups they shopped in."""
    o = o.copy()
    returned_order_ids = set(r["order_id"]) if len(r) else set()
    o["is_returned"] = o["order_id"].isin(returned_order_ids).astype(float)

    grp_totals = o.groupby(group_col).agg(n_orders_g=("order_id", "size"), n_returns_g=("is_returned", "sum"))
    global_rate = grp_totals["n_returns_g"].sum() / max(grp_totals["n_orders_g"].sum(), 1)

    cust_grp = o.groupby(["customer_id", group_col]).agg(
        n_orders_cg=("order_id", "size"), n_returns_cg=("is_returned", "sum")
    ).reset_index()
    cust_grp = cust_grp.merge(grp_totals, on=group_col, how="left")

    denom = (cust_grp["n_orders_g"] - cust_grp["n_orders_cg"])
    numer = (cust_grp["n_returns_g"] - cust_grp["n_returns_cg"])
    cust_grp["baseline_loo"] = np.where(denom > 0, numer / denom, global_rate)

    cust_grp["weight"] = cust_grp["n_orders_cg"]
    cust_grp["weighted_baseline"] = cust_grp["baseline_loo"] * cust_grp["weight"]
    agg = cust_grp.groupby("customer_id").agg(
        total_weight=("weight", "sum"), total_weighted_baseline=("weighted_baseline", "sum")
    )
    weighted_baseline = (agg["total_weighted_baseline"] / agg["total_weight"]).fillna(global_rate)

    deviation = customer_return_rate.reindex(weighted_baseline.index).fillna(0.0) - weighted_baseline
    return deviation.rename(f"{group_col}_relative_deviation")


def _recent_burst_score(r, as_of_day, active_customers):
    if len(r) == 0:
        return pd.Series(0.0, index=active_customers)
    decay = np.log(2) / BURST_HALF_LIFE_DAYS
    w = np.exp(-decay * (as_of_day - r["return_day"]).clip(lower=0))
    score = pd.Series(w.values, index=r["customer_id"].values).groupby(level=0).sum()
    return score.reindex(active_customers).fillna(0.0)


def build_relational_features(orders_df, returns_df, products_df, as_of_day, customer_ids=None):
    """Builds the 9 relational features as of `as_of_day`, using only
    orders/returns with day <= as_of_day. The identifier graph and all
    aggregate baselines are built from ALL customers active by the cutoff
    (not just `customer_ids`) so that cluster/merchant/category statistics
    are complete; `customer_ids` only filters the FINAL rows returned.
    """
    o = orders_df[orders_df.order_day <= as_of_day].copy()
    r = returns_df[returns_df.return_day <= as_of_day].copy()
    o = o.merge(products_df[["product_id", "category"]], on="product_id", how="left")

    if len(o) == 0:
        return pd.DataFrame(columns=["customer_id"] + RELATIONAL_FEATURE_COLUMNS)

    active = sorted(o["customer_id"].unique())
    return_rate = _customer_return_rates(o, r)

    graph_df = _identifier_graph_stats(o)
    cluster_rr, susp_ratio = _cluster_neighbor_stats(graph_df, return_rate)
    merch_dev = _relative_deviation(o, r, "merchant_id", return_rate)
    cat_dev = _relative_deviation(o, r, "category", return_rate)
    burst = _recent_burst_score(r, as_of_day, active)

    feats = graph_df.drop(columns=["_cluster_id"]).copy()
    feats["cluster_return_rate"] = cluster_rr
    feats["suspicious_neighbor_ratio"] = susp_ratio
    feats["merchant_relative_deviation"] = merch_dev.reindex(feats.index).fillna(0.0)
    feats["category_relative_deviation"] = cat_dev.reindex(feats.index).fillna(0.0)
    feats["recent_burst_score"] = burst.reindex(feats.index).fillna(0.0)

    feats = feats.reset_index().rename(columns={"index": "customer_id"})
    feats = feats[["customer_id"] + RELATIONAL_FEATURE_COLUMNS]

    if customer_ids is not None:
        feats = feats[feats.customer_id.isin(set(customer_ids))]
    return feats
