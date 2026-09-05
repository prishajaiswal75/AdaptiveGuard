"""
Phase 6 -- end-to-end demo entry point.

Run:
  python serve/demo_case.py
  python serve/demo_case.py --customer_id C0000009 --as_of_day 209
  python serve/demo_case.py --as_of_day 209 --explain

IMPORTANT (test-isolation note):
The customer used here is selected from the TEST split for READ-ONLY DEMO
INFERENCE ONLY. This customer was never used for model training, never
part of the Phase 5 review-selection pool (that pool was the validation
split only), and never used to fit or tune either frozen threshold
(thresholds were locked from validation-set cost sweeps in Phases 1-3).
Scoring this one case here does not alter that isolation -- it is the same
read-only usage pattern as p4.score_test()/evaluate_model_on_test() in
Phases 4-5's reporting step.

customer_id handling:
customer_id is a STRING throughout (e.g. "C0000009"), matching the IDs
actually used in data/customers.csv and data/splits.csv -- it is never
coerced to int. If --customer_id is supplied, it must be a customer_id
that exists in the TEST split. An ID that is missing, malformed, or from
a different split raises a clear error immediately; this script never
silently substitutes a different customer in that case. The no-argument
default behavior (scan the TEST split for the first customer with activity
by as_of_day) is unchanged and is the only case where multiple candidates
are tried.
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import phase4_equal_budget_ablation as p4
from infer import load_frozen_models, load_meta, score_case


def _test_ids(data):
    """All customer_id values in the TEST split, as strings (never int)."""
    return data["splits"].loc[data["splits"].split == "test", "customer_id"].astype(str).tolist()


def pick_demo_customer(data, as_of_day, frozen_a, frozen_b, meta_a, meta_b, preferred_id=None):
    """Selects one customer from the TEST split for read-only scoring.
    This customer is not used for training, review selection, or
    threshold tuning -- see module docstring.

    If `preferred_id` is given (as a string, e.g. "C0000009"):
      - it MUST be present in the TEST split's customer_id column, or this
        raises ValueError immediately -- no silent fallback to another
        customer_id.
      - if it IS in the TEST split but has no order activity on or before
        `as_of_day` (score_case returns None), this also raises ValueError
        rather than trying a different customer.

    If `preferred_id` is None, this scans the TEST split in order and
    returns the first customer with activity by `as_of_day` -- this is the
    only scenario where multiple candidates are tried.
    """
    test_ids = _test_ids(data)

    if preferred_id is not None:
        preferred_id = str(preferred_id)
        if preferred_id not in test_ids:
            raise ValueError(
                f"customer_id {preferred_id!r} was not found in the TEST split. "
                f"Demo inference only supports TEST-split customer_ids "
                f"(e.g. {test_ids[0]!r}). Refusing to silently substitute a "
                f"different customer."
            )
        result = score_case(data, preferred_id, as_of_day, frozen_a, frozen_b, meta_a, meta_b)
        if result is None:
            raise ValueError(
                f"customer_id {preferred_id!r} is in the TEST split but has no "
                f"order activity on or before as_of_day={as_of_day}. Try a "
                f"different --as_of_day, or omit --customer_id to auto-select "
                f"a test-split customer with activity."
            )
        return result

    for cid in test_ids:
        result = score_case(data, cid, as_of_day, frozen_a, frozen_b, meta_a, meta_b)
        if result is not None:
            return result
    raise RuntimeError("No test-split customer had activity by the given as_of_day.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer_id", type=str, default=None,
                         help="Optional specific customer_id string, e.g. C0000009. "
                              "Must exist in the TEST split; raises a clear error otherwise.")
    parser.add_argument("--as_of_day", type=int, default=209,
                         help="Cutoff day (default 209 = Phase 4/5 round-4 final cutoff).")
    parser.add_argument("--explain", action="store_true",
                         help="Also call Gemini to write a reviewer-facing summary.")
    args = parser.parse_args()

    data = p4.load_data()
    frozen_a, frozen_b = load_frozen_models()
    meta_a, meta_b = load_meta()

    evidence = pick_demo_customer(
        data, args.as_of_day, frozen_a, frozen_b, meta_a, meta_b,
        preferred_id=args.customer_id,
    )

    print("NOTE: test-split customer used for read-only demo inference only "
          "-- not used for training, review selection, or threshold tuning.\n")
    print(json.dumps(evidence, indent=2))

    if args.explain:
        from explain_llm import explain
        print("\n--- Gemini reviewer summary (evidence explanation only) ---\n")
        print(explain(evidence))


if __name__ == "__main__":
    main()