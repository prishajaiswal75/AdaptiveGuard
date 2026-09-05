"""
Model A (exposed/behavioral) feature engineering.

Deliberately excludes anything relational (device/address/payment sharing,
cluster stats) -- that split is what makes the central ablation meaningful.
Model B (Phase 3) will add the relational view on top of the same customers.

Every feature is computed "as of" a cutoff day using only orders/returns
that occurred on or before that day. This is what lets the exact same
function be reused in Phase 2 to score customers as of round 2/3/4 without
leaking future information into the round-1 baseline.
"""
import numpy as np
import pandas as pd

FAST_RETURN_DAYS = 3  # matches config behavior.fast_return_days_abuse upper bound


def build_behavioral_features(orders_df, returns_df, products_df, customers_df,
                               as_of_day, customer_ids=None):
    o = orders_df[orders_df.order_day <= as_of_day].copy()
    r = returns_df[returns_df.return_day <= as_of_day].copy()

    if customer_ids is not None:
        cid_set = set(customer_ids)
        o = o[o.customer_id.isin(cid_set)]
        r = r[r.customer_id.isin(cid_set)]

    o = o.merge(products_df[["product_id", "category"]], on="product_id", how="left")
    r = r.merge(o[["order_id", "order_day"]], on="order_id", how="left")
    r["days_to_return"] = r["return_day"] - r["order_day"]

    active_customers = sorted(o["customer_id"].unique())
    if len(active_customers) == 0:
        return pd.DataFrame(columns=["customer_id"] + BEHAVIORAL_FEATURE_COLUMNS)
    feats = pd.DataFrame({"customer_id": active_customers}).set_index("customer_id")

    g = o.groupby("customer_id")
    feats["n_orders"] = g.size()
    feats["total_order_value"] = g["order_value"].sum()
    feats["avg_order_value"] = g["order_value"].mean()
    feats["order_value_std"] = g["order_value"].std().fillna(0.0)
    feats["n_distinct_merchants"] = g["merchant_id"].nunique()
    feats["n_distinct_categories"] = g["category"].nunique()
    feats["last_order_day"] = g["order_day"].max()
    feats["first_order_day"] = g["order_day"].min()

    if len(r) > 0:
        gr = r.groupby("customer_id")
        n_returns = gr.size()
        avg_days_to_return = gr["days_to_return"].mean()
        pct_fast_returns = gr["days_to_return"].apply(lambda s: (s <= FAST_RETURN_DAYS).mean())
        pct_reason_changed_mind = gr["reason_code"].apply(lambda s: (s == "changed_mind").mean())
    else:
        n_returns = pd.Series(dtype=float)
        avg_days_to_return = pd.Series(dtype=float)
        pct_fast_returns = pd.Series(dtype=float)
        pct_reason_changed_mind = pd.Series(dtype=float)

    feats["n_returns"] = n_returns.reindex(feats.index).fillna(0.0)
    feats["avg_days_to_return"] = avg_days_to_return.reindex(feats.index).fillna(-1.0)  # -1 = never returned
    feats["pct_fast_returns"] = pct_fast_returns.reindex(feats.index).fillna(0.0)
    feats["pct_reason_changed_mind"] = pct_reason_changed_mind.reindex(feats.index).fillna(0.0)

    feats["return_rate"] = feats["n_returns"] / feats["n_orders"]
    feats["returns_per_week"] = feats["n_returns"] / ((as_of_day - feats["first_order_day"] + 1) / 7.0)

    cust = customers_df.set_index("customer_id")
    feats["account_age_days"] = as_of_day - cust.loc[feats.index, "signup_day"].values
    feats["days_since_last_order"] = as_of_day - feats["last_order_day"]

    feats = feats.drop(columns=["last_order_day", "first_order_day"])
    return feats.reset_index()


BEHAVIORAL_FEATURE_COLUMNS = [
    "n_orders", "total_order_value", "avg_order_value", "order_value_std",
    "n_distinct_merchants", "n_distinct_categories", "n_returns",
    "avg_days_to_return", "pct_fast_returns", "pct_reason_changed_mind",
    "return_rate", "returns_per_week", "account_age_days", "days_since_last_order",
]
