"""
Phase 5 -- review-policy ablation runner.

Target path in the repo: AdaptiveGuard/eval/phase5_review_policy_ablation.py
(same directory as phase4_equal_budget_ablation.py and phase5_review_env.py)

WHAT THIS DOES
Compares the three approved review-routing policies -- max_ab (Phase 4
baseline), disagreement, risk_x_disagreement -- under a more realistic
review process: fixed 30-after-R2 + 30-after-R3 capacity, per-customer
noisy / unresolved / delayed outcomes (phase5_review_env.py), while
retraining only on whatever labels have actually "arrived" by each round.

WHAT THIS DOES NOT DO
  - It does NOT modify phase4_equal_budget_ablation.py, metrics.py,
    features.py, features_b.py, or the frozen model_a.json / model_b.json /
    model_a_meta.json / model_b_meta.json artifacts. Those are imported /
    loaded read-only and reused exactly as Phase 1-4 left them.
  - It does NOT change REVIEW_BATCH, NOISE_RATE, UNRESOLVED_RATE,
    DELAYED_FRACTION, DELAY_EXTRA_ROUNDS, or any routing-signal definition
    -- those live in phase5_review_env.py and are only imported here.
  - It does NOT touch the test split anywhere except the existing
    score_test() / evaluate_model_on_test() calls used purely for
    reporting, exactly as Phase 4 does.

NO-LEAKAGE / PROTOCOL GUARANTEES
  - Routing/selection scores (risk_a, risk_b) are computed on the
    VALIDATION split only, using the FROZEN controls (never retrained,
    never touched here).
  - Review labels are the validation pool's oracle labels, passed through
    phase5_review_env.apply_review_outcomes -- noisy labels are used as
    observed, never silently corrected; unresolved cases carry no label.
  - Each policy spends an INDEPENDENT 30+30 budget (its own review log),
    so "equal budget" stays true across the comparison, matching Phase 4.
  - The SAME outcome seed (phase5_review_env.DEFAULT_SEED) is reused for
    every policy run, so a customer_id selected under two different
    policies gets an identical simulated noise/unresolved/delay outcome --
    cross-policy differences come only from WHICH customers get selected.
  - At each round r, retraining uses ONLY reviewed rows whose
    available_from_round <= r (phase5_review_env.trainable_rows) -- a
    label that "hasn't arrived yet" in-universe is never used early.
  - Reviewed rows are featurized at their OWN review_day (149 or 179),
    never recomputed at a later cutoff -- same snapshot-freezing rule
    Phase 4 uses for its reviewed batches.
  - Final test customers are never reviewed and never appear in any
    selection/outcome/training code path; they are read only inside
    score_test()/evaluate_model_on_test() for reporting.

OUTPUTS
  eval/results/phase5_review_policy_metrics.json  -- full comparison
  eval/results/phase5_review_audit_<policy>.csv   -- one row per reviewed
                                                      case, per policy
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import phase4_equal_budget_ablation as p4   # Phase 1-4 code -- imported only, never modified
import phase5_review_env as review_env      # Phase 5 review-policy environment (fairness + delay fixes applied)

import xgboost as xgb

ROUNDS = {1: 119, 2: 149, 3: 179, 4: 209}   # identical to Phase 4's round -> as_of_day mapping


def load_frozen_models():
    frozen_a = xgb.XGBClassifier()
    frozen_a.load_model(str(p4.MODELS_ARTIFACTS / "model_a.json"))
    frozen_b = xgb.XGBClassifier()
    frozen_b.load_model(str(p4.MODELS_ARTIFACTS / "model_b.json"))
    return frozen_a, frozen_b


def fit_baseline(data, meta_a, meta_b):
    """0-review-label retrain, identical recipe to Phase 4's initial
    retrain step. Shared across all three policies since it has no
    policy-dependent inputs."""
    train_ids = data["splits"].loc[data["splits"].split == "train", "customer_id"].tolist()
    base_a = p4.build_behavioral(data, train_ids, meta_a["train_as_of"])
    base_b = p4.build_combined(data, train_ids, meta_b["train_as_of"])
    y_base = p4.labels_for(data["customers"], base_a["customer_id"])
    seed = meta_a["seed"]
    model_a = p4.fit_model(base_a[p4.BEHAVIORAL_FEATURE_COLUMNS].fillna(0.0), y_base, seed)
    model_b = p4.fit_model(
        base_b[p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS].fillna(0.0), y_base, seed
    )
    return model_a, model_b


def compute_candidates(data, frozen_a, frozen_b, day):
    """Validation-pool risk_a/risk_b at `day`, from the FROZEN controls
    only -- same computation Phase 4's selection_scores() does. Exposed
    here so it is computed ONCE and shared across all policies: the
    scores don't depend on policy, only which customers get picked does."""
    val_ids = data["splits"].loc[data["splits"].split == "val", "customer_id"].tolist()
    beh = p4.build_behavioral(data, val_ids, day)
    comb = p4.build_combined(data, val_ids, day)
    risk_a = p4.predict(frozen_a, beh, p4.BEHAVIORAL_FEATURE_COLUMNS)
    risk_b = p4.predict(frozen_b, comb, p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS)
    return pd.DataFrame({"customer_id": beh.customer_id, "risk_a": risk_a, "risk_b": risk_b})


def build_reviewed_snapshot_with_labels(data, batches, kind, trainable):
    """Same snapshot-freezing mechanics as Phase 4's build_reviewed_snapshot:
    each batch of reviewed customer_ids is featurized at its own review_day,
    batches are concatenated (never re-derived from one cutoff). The one
    difference from Phase 4: the label attached to each reviewed row is the
    (possibly noisy) `observed_label` from the review simulation, NOT the
    oracle label -- "noisy labels are observed, never silently corrected."
    `trainable` is phase5_review_env.trainable_rows() output: customer_id,
    observed_label, for exactly the rows usable at the current round."""
    cols = p4.BEHAVIORAL_FEATURE_COLUMNS if kind == "a" else p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS
    label_map = trainable.set_index("customer_id")["observed_label"].to_dict()
    frames, labels = [], []
    for ids, day in batches:
        if not ids:
            continue
        frame = p4.build_behavioral(data, ids, day) if kind == "a" else p4.build_combined(data, ids, day)
        frames.append(frame)
        labels.append(np.array([label_map[c] for c in frame.customer_id], dtype=float))
    if frames:
        return pd.concat(frames, ignore_index=True), np.concatenate(labels)
    return pd.DataFrame(columns=["customer_id"] + cols), np.array([])


def train_augmented_with_labels(data, batches, base_day, kind, seed, trainable):
    """Mirrors phase4.train_augmented(): base training rows keep their
    ORACLE labels (untouched, exactly as Phase 1-4 always used); reviewed
    rows use the noisy `observed_label` from the review simulation."""
    train_ids = data["splits"].loc[data["splits"].split == "train", "customer_id"].tolist()
    if kind == "a":
        base = p4.build_behavioral(data, train_ids, base_day)
        cols = p4.BEHAVIORAL_FEATURE_COLUMNS
    else:
        base = p4.build_combined(data, train_ids, base_day)
        cols = p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS

    extra, extra_y = build_reviewed_snapshot_with_labels(data, batches, kind, trainable)

    frames_X = [base[cols]]
    labels_list = [p4.labels_for(data["customers"], base["customer_id"]).astype(float)]
    if len(extra):
        frames_X.append(extra[cols])
        labels_list.append(extra_y)

    X = pd.concat(frames_X, ignore_index=True).fillna(0.0)
    y = np.concatenate(labels_list)
    model = p4.fit_model(X, y, seed)
    return model, len(X), int(np.nansum(y))


def run_one_policy(data, meta_a, meta_b, frozen_a, frozen_b, baseline_a, baseline_b,
                    candidates_by_round, policy, seed):
    threshold_a = meta_a["threshold_cost_optimal"]
    threshold_b = meta_b["threshold_cost_optimal"]
    cost_fp, cost_fn = meta_a["cost_fp"], meta_a["cost_fn"]
    model_seed = meta_a["seed"]
    oracle_labels = p4.customer_labels_map(data["customers"])
    pop_lookup = data["customers"].set_index("customer_id")["population_type"]

    # ---- Selection + simulated outcomes, VALIDATION POOL ONLY, using the
    # FROZEN controls. Each policy spends its own independent 30+30 budget. ----
    reviewed_ids = set()
    review_log_parts = []
    for review_round, review_day in ((2, 149), (3, 179)):
        candidates = candidates_by_round[review_round]
        batch = review_env.select_batch(candidates, policy, review_env.REVIEW_BATCH, reviewed_ids)
        outcomes = review_env.apply_review_outcomes(batch, oracle_labels, review_round, seed=seed)
        outcomes["review_round"] = review_round
        outcomes["review_day"] = review_day
        outcomes = outcomes.merge(batch[["customer_id", "risk_a", "risk_b"]], on="customer_id", how="left")
        outcomes = outcomes.rename(columns={"risk_a": "frozen_a_risk", "risk_b": "frozen_b_risk"})
        outcomes["population_type_for_audit_only"] = pop_lookup.loc[outcomes.customer_id].values
        reviewed_ids |= set(batch.customer_id)
        review_log_parts.append(outcomes)

    review_log = pd.concat(review_log_parts, ignore_index=True)
    review_day_lookup = review_log[["customer_id", "review_day"]].drop_duplicates()

    # ---- Round-by-round retrain (only what's trainable by then) + eval on
    # the untouched test split. ----
    results = {}
    retrain_a, retrain_b = baseline_a, baseline_b
    for r, day in ROUNDS.items():
        n_trainable = 0
        if r >= 3:
            trainable = review_env.trainable_rows(review_log, current_round=r)
            n_trainable = len(trainable)
            merged = trainable.merge(review_day_lookup, on="customer_id", how="left")
            batches = [(grp.customer_id.tolist(), int(rd)) for rd, grp in merged.groupby("review_day")]
            retrain_a, _, _ = train_augmented_with_labels(data, batches, meta_a["train_as_of"], "a", model_seed, trainable)
            retrain_b, _, _ = train_augmented_with_labels(data, batches, meta_b["train_as_of"], "b", model_seed, trainable)

        frozen = p4.score_test(data, day, frozen_a, frozen_b, threshold_a, threshold_b, cost_fp, cost_fn)
        ma_rt, by_a_rt, _ = p4.evaluate_model_on_test(
            data, day, retrain_a, p4.BEHAVIORAL_FEATURE_COLUMNS, threshold_a, cost_fp, cost_fn
        )
        mb_rt, by_b_rt, _ = p4.evaluate_model_on_test(
            data, day, retrain_b, p4.BEHAVIORAL_FEATURE_COLUMNS + p4.RELATIONAL_FEATURE_COLUMNS, threshold_b, cost_fp, cost_fn
        )

        results[f"round_{r}"] = {
            "as_of_day": day,
            "n_trainable_reviewed_labels": n_trainable,
            "controls": {
                "A_frozen": {
                    "overall": frozen["metrics_a"],
                    "by_population": p4.population_breakdown(
                        frozen["y_true"], (frozen["a"] >= threshold_a).astype(int), frozen["population_type"]
                    ),
                },
                "A+B_frozen": {
                    "overall": frozen["metrics_ab"],
                    "by_population": p4.population_breakdown(
                        frozen["y_true"], (frozen["ab"] >= threshold_b).astype(int), frozen["population_type"]
                    ),
                },
            },
            "retrained": {
                "A_retrain": {"overall": ma_rt, "by_population": by_a_rt},
                "A+B_retrain": {"overall": mb_rt, "by_population": by_b_rt},
            },
        }

    quality = review_env.review_quality_metrics(review_log)
    return results, review_log, quality


def compact(x):
    return {k: x[k] for k in ["precision", "recall", "f1", "fpr", "fp", "fn", "fp_cost", "fn_cost", "total_cost"]}


def main():
    data = p4.load_data()
    meta_a = json.loads((p4.CONFIGS / "model_a_meta.json").read_text())
    meta_b = json.loads((p4.CONFIGS / "model_b_meta.json").read_text())
    seed = review_env.DEFAULT_SEED

    frozen_a, frozen_b = load_frozen_models()
    baseline_a, baseline_b = fit_baseline(data, meta_a, meta_b)

    # Computed ONCE, shared across all policies -- these don't depend on
    # policy, only which customers get selected from them does.
    candidates_by_round = {
        2: compute_candidates(data, frozen_a, frozen_b, 149),
        3: compute_candidates(data, frozen_a, frozen_b, 179),
    }

    policy_results, policy_quality = {}, {}
    for policy in review_env.ROUTING_POLICIES:
        results, review_log, quality = run_one_policy(
            data, meta_a, meta_b, frozen_a, frozen_b, baseline_a, baseline_b,
            candidates_by_round, policy, seed,
        )
        policy_results[policy] = results
        policy_quality[policy] = quality
        review_log.to_csv(p4.RESULTS / f"phase5_review_audit_{policy}.csv", index=False)

    out = {
        "experiment": "Phase 5 review-policy comparison (noisy/delayed/unresolved labels, equal budget)",
        "protocol": {
            "phase_1_4_unchanged": True,
            "frozen_controls_unchanged": True,
            "final_test_touched": False,
            "review_pool": "validation split only",
            "validation_pool_size": int((data["splits"].split == "val").sum()),
            "review_batch_size": review_env.REVIEW_BATCH,
            "total_reviewed_labels_per_policy": review_env.REVIEW_BATCH * 2,
            "routing_policies_compared": list(review_env.ROUTING_POLICIES),
            "each_policy_runs_independent_equal_budget": True,
            "noise_rate": review_env.NOISE_RATE,
            "unresolved_rate": review_env.UNRESOLVED_RATE,
            "delayed_fraction": review_env.DELAYED_FRACTION,
            "delay_extra_rounds": review_env.DELAY_EXTRA_ROUNDS,
            "outcome_seed": seed,
            "outcome_seed_note": (
                "identical across all policies -- a customer selected by two "
                "different policies gets an identical simulated noise/unresolved/"
                "delay outcome; cross-policy differences come only from routing."
            ),
            "thresholds_locked": {"A": meta_a["threshold_cost_optimal"], "A+B": meta_b["threshold_cost_optimal"]},
            "costs": {"cost_fp": meta_a["cost_fp"], "cost_fn": meta_a["cost_fn"]},
            "label_source": "oracle true_abuse_label, passed through the noisy/unresolved/delay review simulation",
        },
        "review_quality_by_policy": policy_quality,
        "rounds_by_policy": policy_results,
        "hero_comparison": {},
    }

    for r in ROUNDS:
        round_key = f"round_{r}"
        out["hero_comparison"][round_key] = {}
        for policy in review_env.ROUTING_POLICIES:
            rr = policy_results[policy][round_key]
            out["hero_comparison"][round_key][policy] = {
                "A_frozen": compact(rr["controls"]["A_frozen"]["overall"]),
                "A_retrain": compact(rr["retrained"]["A_retrain"]["overall"]),
                "A+B_frozen": compact(rr["controls"]["A+B_frozen"]["overall"]),
                "A+B_retrain": compact(rr["retrained"]["A+B_retrain"]["overall"]),
                "n_trainable_reviewed_labels": rr["n_trainable_reviewed_labels"],
                "adaptive_abuse_recall": {
                    "A_frozen": rr["controls"]["A_frozen"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                    "A_retrain": rr["retrained"]["A_retrain"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                    "A+B_frozen": rr["controls"]["A+B_frozen"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                    "A+B_retrain": rr["retrained"]["A+B_retrain"]["by_population"]["adaptive_abuse"].get("recall", 0.0),
                },
                "hard_negative_fpr": {
                    "A_frozen": rr["controls"]["A_frozen"]["by_population"]["hard_negative"].get("fpr", 0.0),
                    "A_retrain": rr["retrained"]["A_retrain"]["by_population"]["hard_negative"].get("fpr", 0.0),
                    "A+B_frozen": rr["controls"]["A+B_frozen"]["by_population"]["hard_negative"].get("fpr", 0.0),
                    "A+B_retrain": rr["retrained"]["A+B_retrain"]["by_population"]["hard_negative"].get("fpr", 0.0),
                },
            }

    (p4.RESULTS / "phase5_review_policy_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(
        {
            "protocol": out["protocol"],
            "review_quality_by_policy": policy_quality,
            "hero_comparison_round_4": out["hero_comparison"]["round_4"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
