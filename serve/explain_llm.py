"""
Phase 6 -- LLM evidence explainer (Google Gemini).

STRICT CONTRACT :
  - Input: the structured evidence dict from evidence.py (already final).
  - Output: a short natural-language summary FOR A HUMAN REVIEWER.
  - The LLM never sees raw training data, never re-scores, never proposes
    a different action/threshold, and its text output is never fed back
    into any decision path. This module has no ability to alter
    `decision` -- that field is passed straight through, unread by the LLM.
  - Gemini receives only the final structured evidence JSON built by
    evidence.py; it is not given access to models, features pipelines,
    or any other repo code/data.

Requires: `pip install google-genai` and env var GEMINI_API_KEY.
This step is optional -- infer.py/demo_case.py work without it.

FIX NOTES (see explain() docstring for detail):
  gemini-3.6-flash is a Gemini 3-series model. Gemini 3 models control
  reasoning effort with `thinking_level` (MINIMAL/LOW/MEDIUM/HIGH), not the
  legacy `thinking_budget` token count -- that parameter is for Gemini 2.5
  models. Passing thinking_budget to a Gemini 3 model doesn't reliably
  throttle it: the model keeps its HIGH-effort default, which combined with
  a tight max_output_tokens silently truncated the visible note mid-sentence
  (that's the garbled, cut-off output previously seen). This version uses
  thinking_level, raises the token budget, and explicitly checks
  finish_reason so a truncated response is retried with more room instead
  of being returned as if it were complete.
"""
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.auth",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.oauth2",
)
import json
import os

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
# Gemini 3.x models spend part of this budget on internal reasoning (thought
# signatures) even at LOW/MINIMAL thinking_level before writing the visible
# answer, so this needs headroom above what the visible note alone needs.
# 2048 was too tight for gemini-3.6-flash and produced truncated notes.
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """You are writing a short note for a human fraud-review analyst.
You will be given structured model evidence (JSON) for one customer case.

Rules you must follow:
- You are explaining evidence only. You do NOT make or change the decision field.
- Do not recommend an action different from evidence["decision"]; just describe why the models produced their scores.
- Treat feature "contributions" as local statistical associations from the model, not proof of intent or causal facts. Use language like "associated with" / "consistent with", never "because the customer did X, the model concluded Y" as a certainty.
- Never infer or state intent, motive, or causality for the customer's behavior, even implicitly.
- Report the top feature contributions faithfully using the feature names, signed values, and directions in evidence["top_contributors"]. You may round contribution values to 4 decimal places for readability, but do not change their sign or direction, omit them, or reorder them.
- Do not cross-reference or pair a top_contributors feature with a raw value from evidence["features"] unless that exact pairing is what top_contributors itself reports; keep the two evidence sections separate in your sentences.
- Do not add subjective qualifiers anywhere in the note -- no words like "minor", "strong", "significant", "notably", "concerning", "high", "low", "small", "large", or similar judgment-laden descriptors. State the numeric values and let them speak for themselves.
- All monetary values must be reported in INR using the ₹ symbol (e.g. ₹236.86), never in USD, unlabeled numbers, or any other currency or unit than what is present in the evidence.
- Do not invent units, facts, feature names, or numbers that are not present in the evidence JSON.
- Keep it to exactly 4 sentences, plain language, no markdown headers, no numbered lists. Be concise.
- End with one sentence reminding the reader this is a model output requiring human judgement."""


def _extract_text(response):
    """Robustly assembles the full visible answer text from a google-genai
    response, and returns the candidate's finish_reason alongside it so the
    caller can detect truncation.

    Walks response.candidates[0].content.parts directly, concatenating
    every part with non-empty `.text` (skipping any part flagged as
    `.thought`, which is internal reasoning trace rather than the visible
    answer). Falls back to the SDK's `response.text` convenience accessor
    only if that manual walk finds nothing, and raises a clear error if the
    model returned no usable text at all (e.g. the response was blocked)
    instead of silently printing an empty or partial string.

    Returns:
        (text, finish_reason) tuple. finish_reason is whatever the SDK
        reports (e.g. "STOP", "MAX_TOKENS") or None if unavailable.
    """
    candidates = getattr(response, "candidates", None) or []
    finish_reason = None
    collected = []

    if candidates:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            if getattr(part, "thought", False):
                continue  # internal reasoning trace, not the visible answer
            text = getattr(part, "text", None)
            if text:
                collected.append(text)

    if collected:
        full_text = "".join(collected).strip()
        if full_text:
            return full_text, finish_reason

    fallback = getattr(response, "text", None)
    if fallback and fallback.strip():
        return fallback.strip(), finish_reason

    raise RuntimeError(
        "Gemini returned no usable text for the reviewer summary "
        f"(finish_reason={finish_reason!r}). The evidence/decision are "
        "unaffected; only the optional explanation text failed to generate."
    )


def _build_config(types, max_output_tokens, thinking_level=None, thinking_budget=None):
    """Builds a GenerateContentConfig. Pass at most one of thinking_level
    (Gemini 3.x: MINIMAL/LOW/MEDIUM/HIGH) or thinking_budget (legacy,
    Gemini 2.5 token count) -- passing both in the same request is rejected
    by Gemini 3 models. Pass neither for a plain config with no thinking
    control at all (oldest SDKs, or as a last-resort fallback).

    Returns None if the requested thinking field isn't available on this
    SDK version, so the caller can skip straight to the next attempt.
    """
    kwargs = dict(system_instruction=SYSTEM_PROMPT, max_output_tokens=max_output_tokens)
    try:
        if thinking_level is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        elif thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        return types.GenerateContentConfig(**kwargs)
    except AttributeError:
        return None


def explain(evidence: dict) -> str:
    """Calls Gemini to turn the evidence dict into a short reviewer note.

    Tries configs in order of preference, retrying with a larger token
    budget whenever a response comes back truncated (finish_reason ==
    "MAX_TOKENS") instead of returning partial text:

      1. thinking_level=LOW (correct control for Gemini 3.x models such as
         the default gemini-3.6-flash) at the base token budget.
      2. Same, at double the token budget, in case LOW thinking still ate
         into the visible answer's share.
      3. thinking_budget=0 (legacy control, for Gemini 2.5-era models if
         GEMINI_MODEL is overridden to one) at double the token budget.
      4. No thinking_config at all, at double the token budget -- last
         resort for SDK versions that expose neither field.

    Raises RuntimeError if every attempt errors out or still comes back
    truncated, rather than ever returning a partial note.
    """
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. Set it to use "
            "the optional --explain step (e.g. export GEMINI_API_KEY=... )."
        )

    client = genai.Client(api_key=api_key)
    user_content = (
        "Evidence JSON for one case:\n\n"
        f"{json.dumps(evidence, indent=2)}\n\n"
        "Write the reviewer note now."
    )

    thinking_level_low = getattr(getattr(types, "ThinkingLevel", None), "LOW", None)

    attempts = [
        dict(max_output_tokens=MAX_OUTPUT_TOKENS, thinking_level=thinking_level_low),
        dict(max_output_tokens=MAX_OUTPUT_TOKENS * 2, thinking_level=thinking_level_low),
        dict(max_output_tokens=MAX_OUTPUT_TOKENS * 2, thinking_budget=0),
        dict(max_output_tokens=MAX_OUTPUT_TOKENS * 2),
    ]

    last_error = None
    for attempt_kwargs in attempts:
        config = _build_config(types, **attempt_kwargs)
        if config is None:
            continue  # this SDK version doesn't support this attempt's fields

        try:
            response = client.models.generate_content(
                model=MODEL, contents=user_content, config=config,
            )
        except genai_errors.ClientError as exc:
            # This exact request (e.g. an unsupported thinking field for this
            # model/API version) was rejected as invalid -- try the next
            # attempt rather than crashing the demo.
            last_error = exc
            continue

        text, finish_reason = _extract_text(response)
        if finish_reason == "MAX_TOKENS":
            # Truncated -- don't return partial text, try the next attempt.
            last_error = RuntimeError(
                f"response truncated (finish_reason=MAX_TOKENS) at "
                f"max_output_tokens={attempt_kwargs['max_output_tokens']}"
            )
            continue

        return text

    raise RuntimeError(
        "Gemini could not produce a complete reviewer summary after "
        f"retrying with larger token budgets (last issue: {last_error!r}). "
        "The evidence/decision are unaffected; only the optional "
        "explanation text failed to generate."
    )