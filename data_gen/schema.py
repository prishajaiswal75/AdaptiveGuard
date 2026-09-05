"""
AdaptiveGuard data schema.

This is documentation-as-code: every downstream module (features, models,
eval) should reference these column lists rather than hardcoding strings,
so a schema change fails loudly instead of silently.
"""

POPULATION_TYPES = ["normal", "hard_negative", "naive_abuse", "adaptive_abuse"]

HARD_NEGATIVE_SUBTYPES = [
    "high_return_fashion",
    "shared_household",
    "promo_spike",
    "popular_product_volume",
    "new_merchant_thin_history",
]

ROUND_NAMES = {
    0: "baseline",
    1: "deployment",
    2: "adaptation_begins",
    3: "model_a_degrades",
    4: "stress_test_relational",
}

MERCHANTS_COLUMNS = [
    "merchant_id", "category", "history_depth_days", "size_tier",
]

PRODUCTS_COLUMNS = [
    "product_id", "merchant_id", "category", "price", "popularity_tier", "defect_prone",
]

CUSTOMERS_COLUMNS = [
    "customer_id", "signup_day", "population_type", "hard_negative_subtype",
    "ring_id", "device_id", "address_id", "payment_fp_id",
]

ORDERS_COLUMNS = [
    "order_id", "customer_id", "merchant_id", "product_id", "order_day",
    "order_value", "device_id", "address_id", "payment_fp_id", "round",
]

RETURNS_COLUMNS = [
    "return_id", "order_id", "customer_id", "return_day", "reason_code",
    "refund_amount", "true_abuse_label",
]

SPLITS_COLUMNS = ["customer_id", "cluster_id", "split"]

REASON_CODES = [
    "size_fit", "changed_mind", "defective", "not_as_described",
    "duplicate_order", "other",
]

# Oracle label semantics: 1 iff the CUSTOMER's underlying population is
# genuinely abusive. Hard negatives are ALWAYS 0 regardless of how suspicious
# their behavior looks -- that's the point of including them.
ABUSIVE_POPULATIONS = {"naive_abuse", "adaptive_abuse"}


def round_for_day(day: int, rounds_cfg: dict) -> int:
    for rnum, bounds in rounds_cfg.items():
        if bounds["start"] <= day <= bounds["end"]:
            return int(rnum)
    # clamp to last defined round if beyond configured horizon
    return max(int(r) for r in rounds_cfg)
