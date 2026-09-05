"""
AdaptiveGuard synthetic environment generator.

Defense-only, closed, offline synthetic evaluation harness. Produces no real
customer/merchant data, connects to no external system, and is used solely
to test detector robustness under controlled distribution shift.

Run:
    python data_gen/generate.py --config configs/config.yaml --out data/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_gen.schema import (
    REASON_CODES, round_for_day,
)


class UnionFind:
    """Groups customers who share an identifier (device/address/payment_fp)
    into connected components, so a split can keep whole rings/households
    together (no graph leakage across train/val/test)."""

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


class IdCounter:
    def __init__(self, prefix):
        self.prefix, self.n = prefix, 0

    def next(self):
        self.n += 1
        return f"{self.prefix}{self.n:06d}"


def gen_merchants(cfg, rng):
    n = cfg["merchants"]["count"]
    lo, hi = cfg["merchants"]["history_depth_days_range"]
    cats = cfg["merchants"]["categories"]
    rows = []
    for i in range(n):
        rows.append({
            "merchant_id": f"M{i:04d}",
            "category": rng.choice(cats),
            "history_depth_days": int(rng.integers(lo, hi)),
            "size_tier": rng.choice(["small", "medium", "large"], p=[0.5, 0.35, 0.15]),
        })
    return pd.DataFrame(rows)


def gen_products(cfg, merchants_df, rng):
    lo, hi = cfg["merchants"]["products_per_merchant"]
    rows, pid = [], 0
    for _, m in merchants_df.iterrows():
        for _ in range(int(rng.integers(lo, hi + 1))):
            rows.append({
                "product_id": f"P{pid:06d}",
                "merchant_id": m["merchant_id"],
                "category": m["category"],
                "price": round(float(rng.lognormal(3.2, 0.6)), 2),
                "popularity_tier": rng.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
                "defect_prone": bool(rng.random() < 0.05),
            })
            pid += 1
    return pd.DataFrame(rows)


_pending_identifier_overrides = {}


def _cluster_group(members, size_range, share_prob, uf, id_counters, rng, ring_prefix, ring_map):
    """Shuffle `members` into groups of `size_range`, give each group shared
    identifiers with partial-sharing probabilities, register in union-find.
    Side effects: mutates `ring_map` and `_pending_identifier_overrides`."""
    members = list(members)
    rng.shuffle(members)
    i, ring_n = 0, 0
    while i < len(members):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        group = members[i:i + size]
        i += size
        if len(group) < 2:
            continue
        ring_n += 1
        rid = f"{ring_prefix}{ring_n:04d}"
        shared = {
            "device_id": id_counters["device"].next(),
            "address_id": id_counters["address"].next(),
            "payment_fp_id": id_counters["payment"].next(),
        }
        for c in group:
            ring_map[c] = rid
            for key, p in share_prob.items():
                col = f"{key}_id" if not key.endswith("_id") else key
                if rng.random() < p:
                    uf.union(c, shared[col])
                    _pending_identifier_overrides.setdefault(c, {})[col] = shared[col]


def gen_customers(cfg, rng):
    n = cfg["customers"]["total"]
    mix = cfg["customers"]["population_mix"]
    pops = rng.choice(list(mix.keys()), size=n, p=list(mix.values()))
    cust_ids = [f"C{i:07d}" for i in range(n)]
    pop_of = dict(zip(cust_ids, pops))

    id_counters = {"device": IdCounter("D"), "address": IdCounter("A"), "payment": IdCounter("PF")}
    uf = UnionFind()
    ring_map = {}
    _pending_identifier_overrides.clear()

    # base (private) identifiers for everyone
    base_ids = {
        c: {
            "device_id": id_counters["device"].next(),
            "address_id": id_counters["address"].next(),
            "payment_fp_id": id_counters["payment"].next(),
        }
        for c in cust_ids
    }

    def run_clustering(pop_name, size_range, share_prob, prefix):
        members = [c for c, p in pop_of.items() if p == pop_name]
        _cluster_group(members, size_range, share_prob, uf, id_counters, rng, prefix, ring_map)

    run_clustering("naive_abuse", cfg["rings"]["naive_abuse_ring_size_range"],
                    cfg["rings"]["share_prob"], "NA_RING")
    run_clustering("adaptive_abuse", cfg["rings"]["adaptive_abuse_ring_size_range"],
                    cfg["rings"]["share_prob"], "AA_RING")

    # hard-negative subtypes (assign first, then cluster shared_household)
    hn_customers = [c for c, p in pop_of.items() if p == "hard_negative"]
    hn_mix = cfg["customers"]["hard_negative_subtype_mix"]
    subtypes = rng.choice(list(hn_mix.keys()), size=len(hn_customers), p=list(hn_mix.values()))
    subtype_of = dict(zip(hn_customers, subtypes))

    household_members = [c for c, s in subtype_of.items() if s == "shared_household"]
    _cluster_group(household_members, cfg["rings"]["shared_household_size_range"],
                   cfg["rings"]["household_share_prob"], uf, id_counters, rng, "HH", ring_map)

    lo, hi = cfg["customers"]["signup_day_range"]
    rows = []
    for c in cust_ids:
        ids = dict(base_ids[c])
        ids.update(_pending_identifier_overrides.get(c, {}))
        rows.append({
            "customer_id": c,
            "signup_day": int(rng.integers(lo, hi)),
            "population_type": pop_of[c],
            "hard_negative_subtype": subtype_of.get(c, None),
            "ring_id": ring_map.get(c, None),
            "device_id": ids["device_id"],
            "address_id": ids["address_id"],
            "payment_fp_id": ids["payment_fp_id"],
        })
    return pd.DataFrame(rows), uf


def _sample_rate(lo_hi, rng):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def gen_orders_returns(cfg, customers_df, merchants_df, products_df, rng):
    total_days = cfg["simulation"]["total_days"]
    rounds_cfg = cfg["simulation"]["rounds"]
    beh = cfg["behavior"]
    promo_lo, promo_hi = beh["promo_window_days"]
    churn_prob = beh["round4_identity_churn_prob"]

    merchants_by_id = merchants_df.set_index("merchant_id")
    products_by_merchant = products_df.groupby("merchant_id")
    thin_merchants = merchants_df.sort_values("history_depth_days").head(
        max(1, len(merchants_df) // 6)
    )["merchant_id"].tolist()
    popular_defect_products = products_df[
        (products_df.popularity_tier == "high") & (products_df.defect_prone)
    ]
    fallback_products = products_df

    order_ctr, return_ctr = IdCounter("O"), IdCounter("R")
    order_rows, return_rows = [], []

    orders_mean = cfg["customers"]["orders_per_customer_mean"]

    for cust in customers_df.itertuples():
        pop = cust.population_type
        n_orders = max(1, int(rng.poisson(orders_mean.get(pop, 8))))
        active_lo = cust.signup_day
        order_days = sorted(int(d) for d in rng.integers(active_lo, total_days, size=n_orders))

        # pick a merchant preference for cold-start / product-volume hard negatives
        forced_merchant = None
        forced_product_pool = None
        if pop == "hard_negative" and cust.hard_negative_subtype == "new_merchant_thin_history":
            forced_merchant = rng.choice(thin_merchants)
        if pop == "hard_negative" and cust.hard_negative_subtype == "popular_product_volume" \
                and len(popular_defect_products) > 0:
            forced_product_pool = popular_defect_products

        for od in order_days:
            if forced_merchant is not None:
                merchant_id = forced_merchant
                prod_pool = products_by_merchant.get_group(merchant_id)
            elif forced_product_pool is not None:
                prow = forced_product_pool.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
                merchant_id = prow["merchant_id"]
                prod_pool = None
            else:
                merchant_id = rng.choice(merchants_df["merchant_id"].values)
                prod_pool = products_by_merchant.get_group(merchant_id)

            if forced_product_pool is not None and prod_pool is None:
                product_id = prow["product_id"]
                price = prow["price"]
            else:
                prow = prod_pool.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
                product_id = prow["product_id"]
                price = prow["price"]

            rnd = round_for_day(od, rounds_cfg)

            device_id, address_id, payment_fp_id = cust.device_id, cust.address_id, cust.payment_fp_id
            if pop in ("naive_abuse", "adaptive_abuse") and rnd == 4 and rng.random() < churn_prob:
                # round-4 stress test: some ring members abandon the shared
                # identifier on THIS order, to test whether Model B's signal
                # degrades under targeted adaptation too.
                device_id, address_id, payment_fp_id = f"D_FRESH_{cust.customer_id}", \
                    f"A_FRESH_{cust.customer_id}", f"PF_FRESH_{cust.customer_id}"

            oid = order_ctr.next()
            order_rows.append({
                "order_id": oid, "customer_id": cust.customer_id, "merchant_id": merchant_id,
                "product_id": product_id, "order_day": od,
                "order_value": round(float(price), 2),
                "device_id": device_id, "address_id": address_id, "payment_fp_id": payment_fp_id,
                "round": rnd,
            })

            # ---- return decision ----
            p_return, fast = _return_probability(pop, cust, od, rnd, beh, promo_lo, promo_hi, rng)
            if rng.random() < p_return:
                delay_range = beh["fast_return_days_abuse"] if fast else beh["fast_return_days_normal"]
                ret_day = min(total_days - 1, od + int(rng.integers(delay_range[0], delay_range[1] + 1)))
                reason = "changed_mind" if fast and rng.random() < 0.6 else rng.choice(REASON_CODES)
                label = 1 if pop in ("naive_abuse", "adaptive_abuse") else 0
                return_rows.append({
                    "return_id": return_ctr.next(), "order_id": oid, "customer_id": cust.customer_id,
                    "return_day": ret_day, "reason_code": reason,
                    "refund_amount": round(float(price), 2), "true_abuse_label": label,
                })

    return pd.DataFrame(order_rows), pd.DataFrame(return_rows)


def _return_probability(pop, cust, order_day, rnd, beh, promo_lo, promo_hi, rng):
    """Returns (p_return, is_fast_turnaround)."""
    if pop == "normal":
        return _sample_rate(beh["return_rate"]["normal"], rng), False

    if pop == "hard_negative":
        sub = cust.hard_negative_subtype
        base = _sample_rate(beh["return_rate"]["hard_negative_other"], rng)
        if sub == "high_return_fashion":
            return _sample_rate(beh["return_rate"]["hard_negative_high_return_fashion"], rng), False
        if sub == "promo_spike" and promo_lo <= order_day <= promo_hi:
            return base + _sample_rate(beh["return_rate"]["hard_negative_promo_spike_boost"], rng), False
        return base, False

    if pop == "naive_abuse":
        return _sample_rate(beh["return_rate"]["naive_abuse"], rng), True

    if pop == "adaptive_abuse":
        if rnd <= 1:
            return _sample_rate(beh["return_rate"]["adaptive_abuse_pre_round2"], rng), True
        # rounds 2+: mimic normal behaviorally (slower, less obvious returns)
        return _sample_rate(beh["return_rate"]["adaptive_abuse_post_round2"], rng), False

    return 0.05, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["seed"])

    merchants_df = gen_merchants(cfg, rng)
    products_df = gen_products(cfg, merchants_df, rng)
    customers_df, uf = gen_customers(cfg, rng)
    orders_df, returns_df = gen_orders_returns(cfg, customers_df, merchants_df, products_df, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merchants_df.to_csv(out / "merchants.csv", index=False)
    products_df.to_csv(out / "products.csv", index=False)
    customers_df.to_csv(out / "customers.csv", index=False)
    orders_df.to_csv(out / "orders.csv", index=False)
    returns_df.to_csv(out / "returns.csv", index=False)

    print(f"merchants={len(merchants_df)} products={len(products_df)} "
          f"customers={len(customers_df)} orders={len(orders_df)} returns={len(returns_df)}")
    print("population mix:")
    print(customers_df.population_type.value_counts())
    print("\nreturn rate by population (sanity check):")
    merged = returns_df.merge(customers_df[["customer_id", "population_type"]], on="customer_id", how="right")
    order_counts = orders_df.groupby("customer_id").size()
    return_counts = returns_df.groupby("customer_id").size()
    rr = (return_counts.reindex(order_counts.index, fill_value=0) / order_counts).groupby(
        customers_df.set_index("customer_id")["population_type"]
    ).mean()
    print(rr)


if __name__ == "__main__":
    main()
