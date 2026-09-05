"""
Phase 5 -- review-policy environment.

Target path in the repo: AdaptiveGuard/eval/phase5_review_env.py

Adds capacity, noise, delay, and unresolved-outcome modeling on top of the
Phase 4 common review protocol. This module contains NO xgboost training and
NO test-split access: it only operates on a validation-pool DataFrame of
per-customer risk scores (risk_a, risk_b) and an oracle-label lookup, so it
can be unit-tested with synthetic data before being wired into
phase5_review_policy_ablation.py.

Design constants below are hand-picked defaults for this experiment, not
tuned/fit to any data -- consistent with this repo's existing convention
(see features_b.py BURST_HALF_LIFE_DAYS, phase4 REVIEW_BATCH).

Protocol invariants enforced here (do not relax without updating callers):
  - Total review capacity is fixed at 30 + 30 = 60 across the whole
    experiment, same as Phase 4, regardless of routing policy. Each policy
    is run as an independent 60-label experiment so budgets stay
    comparable across policies.
  - FAIRNESS ACROSS POLICIES: noise / unresolved / delay outcomes for a
    given customer_id are a deterministic function of (customer_id, seed)
    -- not of draw order, batch composition, or which routing policy
    selected the customer. The same seed must be reused across every
    policy run being compared, so differences in downstream metrics come
    only from WHICH customers each policy selects.
  - DELAY HORIZON: delay is applied probabilistically per resolved case
    (DELAYED_FRACTION), not as a fixed extra round applied to everyone.
    The first batch (reviewed after R2) is always available by R4 at the
    latest. The second batch (reviewed after R3) is only partially
    delayed past R4 -- the rest becomes available in time for the R4
    retrain, so the 30+30 budget can actually affect the R4 timeline.
  - Noisy labels are returned as `observed_label`; the true `oracle_label`
    is carried alongside for audit ONLY and must never be substituted
    back in as a training target.
  - A resolved+labeled case is not "trainable" until `available_from_round`
    <= the round being trained for. Callers must filter with
    trainable_rows() rather than using every reviewed row unconditionally.
  - Unresolved cases (`resolved=False`) carry no observed_label and must be
    excluded from training unless a future protocol update defines
    otherwise.
  - This module never reads or references the test split.
"""
import hashlib

import numpy as np
import pandas as pd

REVIEW_BATCH = 30
NOISE_RATE = 0.10           # fraction of RESOLVED reviewed cases whose observed label is flipped from oracle
UNRESOLVED_RATE = 0.10      # fraction of reviewed cases that come back with no usable label at all
DELAYED_FRACTION = 0.50     # fraction of RESOLVED reviewed cases that incur one extra round of delay
DELAY_EXTRA_ROUNDS = 1      # size of that extra delay, in rounds, when it applies
ROUTING_POLICIES = ("max_ab", "disagreement", "risk_x_disagreement")

# Must be identical across every policy run in a comparison set -- see the
# FAIRNESS ACROSS POLICIES note above.
DEFAULT_SEED = 20260905


def routing_signal(risk_a, risk_b, policy):
    """Vectorized routing-priority score for one policy. Higher = reviewed sooner.

    max_ab               -- Phase 4 baseline: max(risk_a, risk_b).
    disagreement          -- |risk_a - risk_b|; untested-until-measured candidate.
    risk_x_disagreement   -- max(risk_a, risk_b) * |risk_a - risk_b|; combined
                              priority candidate, per Phase 5 scope notes.
    """
    risk_a = np.asarray(risk_a, dtype=float)
    risk_b = np.asarray(risk_b, dtype=float)
    if policy == "max_ab":
        return np.maximum(risk_a, risk_b)
    if policy == "disagreement":
        return np.abs(risk_a - risk_b)
    if policy == "risk_x_disagreement":
        return np.maximum(risk_a, risk_b) * np.abs(risk_a - risk_b)
    raise ValueError(f"Unknown routing policy: {policy}")


def select_batch(candidates, policy, batch_size, already_reviewed):
    """candidates: DataFrame with columns customer_id, risk_a, risk_b, drawn
    from the VALIDATION pool only. Returns the next `batch_size` not-yet-
    reviewed rows ranked by routing_signal(policy) descending, customer_id
    ascending as the tie-break -- same deterministic tie-break rule used in
    Phase 4's selection_scores()."""
    df = candidates[~candidates.customer_id.isin(already_reviewed)].copy()
    df["priority_score"] = routing_signal(df.risk_a, df.risk_b, policy)
    df = df.sort_values(["priority_score", "customer_id"], ascending=[False, True])
    return df.head(batch_size).reset_index(drop=True)


def _stable_unit_interval(customer_id, salt, seed):
    """Deterministic pseudo-random value in [0, 1) for (customer_id, salt,
    seed). Independent of call order, batch membership, or which routing
    policy selected the customer -- this is what makes noise/unresolved/
    delay outcomes fair across policies. Python's built-in hash() is NOT
    used because it is randomized per-process (PYTHONHASHSEED); sha256
    keeps this stable across runs, processes, and policies."""
    digest = hashlib.sha256(f"{seed}:{salt}:{customer_id}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16 ** 15)


def _stable_draws(customer_ids, salt, seed):
    return np.array([_stable_unit_interval(cid, salt, seed) for cid in customer_ids])


def apply_review_outcomes(batch, oracle_labels, review_round, seed=DEFAULT_SEED):
    """Simulates what a limited/imperfect review process returns for one
    freshly-selected batch.

    batch: DataFrame with a customer_id column (output of select_batch()).
    oracle_labels: dict customer_id -> true 0/1 abuse label (ground truth;
        used only to populate the audit column here, never written into
        observed_label for noisy/unresolved cases).
    review_round: the round number this batch was reviewed after (2 or 3
        in this experiment). Availability timing is anchored to this.
    seed: fixes the per-customer noise/unresolved/delay draws. MUST be the
        same value across every routing-policy run being compared --
        outcomes are a pure function of (customer_id, seed), never of
        batch composition or draw order.

    Returns a DataFrame, one row per reviewed customer:
        customer_id, oracle_label (audit only), resolved (bool),
        noisy (bool), observed_label (float, NaN if unresolved),
        available_from_round (float, NaN if unresolved).
    """
    out = batch[["customer_id"]].copy()
    out["oracle_label"] = [oracle_labels[c] for c in out.customer_id]

    unresolved_draw = _stable_draws(out.customer_id, "unresolved", seed)
    noise_draw = _stable_draws(out.customer_id, "noise", seed)
    delay_draw = _stable_draws(out.customer_id, "delay", seed)

    unresolved_mask = unresolved_draw < UNRESOLVED_RATE
    noise_mask = (~unresolved_mask) & (noise_draw < NOISE_RATE)
    delayed_mask = (~unresolved_mask) & (delay_draw < DELAYED_FRACTION)

    observed = out["oracle_label"].astype(float).to_numpy()
    observed[noise_mask] = 1.0 - observed[noise_mask]   # flip; never "corrected" back later
    observed[unresolved_mask] = np.nan

    out["resolved"] = ~unresolved_mask
    out["noisy"] = noise_mask
    out["observed_label"] = observed

    extra_delay = np.where(delayed_mask, DELAY_EXTRA_ROUNDS, 0)
    # Phase 4 baseline timing = available at review_round + 1 (immediate use
    # at the very next round). The delayed fraction adds DELAY_EXTRA_ROUNDS
    # on top of that baseline for THIS batch's resolved cases only.
    out["available_from_round"] = np.where(
        out["resolved"], review_round + 1 + extra_delay, np.nan
    )
    return out


def trainable_rows(review_log, current_round):
    """review_log: concatenation of apply_review_outcomes() outputs across
    all batches reviewed so far. Returns only rows that are resolved AND
    whose delay has elapsed by current_round -- i.e. safe to append to the
    training set for that round's retrain. Unresolved and still-delayed
    rows are silently excluded, never imputed as a label."""
    ready = review_log["resolved"] & (review_log["available_from_round"] <= current_round)
    return review_log.loc[ready, ["customer_id", "observed_label"]].copy()


def review_quality_metrics(review_log):
    """Review precision / abuse yield / delay impact, computed against the
    oracle_label audit column. Reporting only -- never feeds training."""
    n = len(review_log)
    if n == 0:
        return {"n_reviewed": 0}
    resolved = review_log[review_log.resolved]
    n_resolved = len(resolved)
    n_unresolved = n - n_resolved
    n_noisy = int(resolved.noisy.sum()) if n_resolved else 0
    label_agreement = (
        float((resolved.observed_label == resolved.oracle_label).mean())
        if n_resolved else float("nan")
    )
    return {
        "n_reviewed": n,
        "abuse_yield": float(review_log.oracle_label.mean()),  # fraction of reviewed cases truly abusive
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "unresolved_rate_observed": n_unresolved / n,
        "n_noisy_labels": n_noisy,
        "noisy_rate_observed": (n_noisy / n_resolved) if n_resolved else float("nan"),
        "observed_vs_oracle_label_agreement": label_agreement,
    }


if __name__ == "__main__":
    # Self-test with synthetic data only -- touches no repo data files.
    # Run with: python eval/phase5_review_env.py
    rng = np.random.default_rng(42)
    n = 200
    synth = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "risk_a": rng.random(n),
        "risk_b": rng.random(n),
    })
    oracle = {cid: int(rng.random() < 0.2) for cid in synth.customer_id}

    # Fix #1 check: same customer_id -> same outcome, regardless of round or row order.
    probe_batch = pd.DataFrame({"customer_id": ["c17", "c42", "c103"]})
    out_r2 = apply_review_outcomes(probe_batch, oracle, review_round=2)
    out_r3_shuffled = apply_review_outcomes(probe_batch.iloc[::-1].reset_index(drop=True), oracle, review_round=3)
    cols = ["resolved", "noisy", "observed_label"]
    assert out_r2.set_index("customer_id")[cols].sort_index().equals(
        out_r3_shuffled.set_index("customer_id")[cols].sort_index()
    ), "noise/unresolved outcome depends on round or row order"
    print("fairness check passed\n")

    for policy in ROUTING_POLICIES:
        reviewed = set()
        b1 = select_batch(synth, policy, REVIEW_BATCH, reviewed)
        log1 = apply_review_outcomes(b1, oracle, review_round=2)
        reviewed |= set(b1.customer_id)

        b2 = select_batch(synth, policy, REVIEW_BATCH, reviewed)
        log2 = apply_review_outcomes(b2, oracle, review_round=3)
        reviewed |= set(b2.customer_id)

        full_log = pd.concat([log1, log2], ignore_index=True)
        assert len(reviewed) == 2 * REVIEW_BATCH
        assert full_log.customer_id.is_unique

        n_batch2_by_r4 = trainable_rows(log2, current_round=4).shape[0]
        print(f"[{policy}]")
        print(" quality:", review_quality_metrics(full_log))
        print(" trainable @round3:", len(trainable_rows(full_log, current_round=3)))
        print(" trainable @round4:", len(trainable_rows(full_log, current_round=4)))
        print(f" second batch (30) reaching R4: {n_batch2_by_r4}/30")
        assert n_batch2_by_r4 > 0, "second batch should be able to reach R4, not all pushed to R5"
        print()
