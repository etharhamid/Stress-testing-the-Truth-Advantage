"""Fitness module: runs a debate + judge with per-call model overrides
(thread-safe) and converts the judgement into a scalar fitness signal.

The four `run_*_{debate,judge}` wrappers pass the QD-phase model through the
`model=` kwarg of `run_debate` / `judge_debate` instead of mutating
`config.DEBATER_MODEL` / `config.JUDGE_MODEL`, so multiple workers may run
these concurrently without trampling one another's globals.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

from google.genai import types

import config

from core.debate_engine import run_debate
from core.gemini_client import chat_structured_json
from core.judge_engine import judge_debate
from core.prompts import build_confidence_estimator_messages

_JUDGE_RETRY_ATTEMPTS = 2
_JUDGE_RETRY_SLEEP = 30  # seconds between retries on Vertex logprob failure

_CONFIDENCE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={"confidence": types.Schema(type="INTEGER")},
    required=["confidence"],
    property_ordering=["confidence"],
)


@contextmanager
def _temporarily_set_models(*, debater_model: str | None = None, judge_model: str | None = None):
    """Temporarily override config.DEBATER_MODEL / config.JUDGE_MODEL for a block.
    Restores on exit even if an exception is raised. Single-threaded by design —
    the QD search/eval wrappers no longer use this; it is kept for any other
    caller that still relies on the global override pattern."""
    saved_debater = config.DEBATER_MODEL
    saved_judge = config.JUDGE_MODEL
    if debater_model is not None:
        config.DEBATER_MODEL = debater_model
    if judge_model is not None:
        config.JUDGE_MODEL = judge_model
    try:
        yield
    finally:
        config.DEBATER_MODEL = saved_debater
        config.JUDGE_MODEL = saved_judge


def run_search_debate(row: dict, swap: bool) -> dict:
    """Runs a debate using the QD_SEARCH_DEBATER_MODEL. Returns the transcript dict."""
    return run_debate(row, swap=swap, model=config.QD_SEARCH_DEBATER_MODEL)


def run_search_judge(transcript: dict) -> dict:
    """Judges a transcript using QD_SEARCH_JUDGE_MODEL (Vertex, with logprobs).

    On logprob failure (judge_fallback=True), retries up to _JUDGE_RETRY_ATTEMPTS
    times with a _JUDGE_RETRY_SLEEP delay. If all retries fail, OR if the Vertex
    call succeeded but returned judge_log_odds_truth=None (e.g. target tokens
    absent from logprob output), falls back to the LLM confidence estimator to
    produce a continuous fitness. If that also fails, returns the result with
    judge_log_odds_truth=None so the iteration is skipped.
    No ±2.0 binary fallback is used.
    """
    model_id = config.QD_SEARCH_JUDGE_MODEL
    result = judge_debate(transcript, model=model_id)

    for _ in range(_JUDGE_RETRY_ATTEMPTS):
        if not result.get("judge_fallback"):
            break
        time.sleep(_JUDGE_RETRY_SLEEP)
        result = judge_debate(transcript, model=model_id)

    if result.get("judge_fallback") or result.get("judge_log_odds_truth") is None:
        result = _estimate_log_odds_from_transcript(result)

    return result


def _estimate_log_odds_from_transcript(debate_result: dict) -> dict:
    """Estimate judge_log_odds_truth via an LLM confidence call (AI Studio, no Vertex).

    Asks flash-lite to rate the judge's confidence (1–99) given the transcript
    and the already-known judge answer, then maps that to a log-odds value on the
    same scale as the real Vertex logprobs. Sets judge_confidence_estimated=True.
    On any failure, leaves judge_log_odds_truth=None so the caller skips the
    archive update.
    """
    transcript_str = debate_result.get("transcript_str", "")
    judge_answer = debate_result.get("judge_answer") or "A"
    messages = build_confidence_estimator_messages(
        transcript=transcript_str,
        judge_answer=judge_answer,
    )
    try:
        obj, _ = chat_structured_json(
            model=config.QD_VALIDATOR_MODEL,
            messages=messages,
            response_schema=_CONFIDENCE_SCHEMA,
            profile="qd_validator",
            temperature=0.0,
        )
        score = max(1, min(99, int(obj.get("confidence", 50))))
        conf = score / 100.0
        logit = math.log(conf / (1.0 - conf))
        correct_letter = debate_result.get("correct_letter", "A")
        log_odds_truth = logit if judge_answer == correct_letter else -logit
        debate_result["judge_log_odds_truth"] = log_odds_truth
        debate_result["judge_confidence_estimated"] = True
    except Exception:
        pass  # judge_log_odds_truth stays None → fitness_from_judgement returns None → skip

    return debate_result


def run_eval_debate(row: dict, swap: bool) -> dict:
    """Runs a debate using the baseline-grade QD_EVAL_DEBATER_MODEL."""
    return run_debate(row, swap=swap, model=config.QD_EVAL_DEBATER_MODEL)


def run_eval_judge(transcript: dict) -> dict:
    """Judges using QD_EVAL_JUDGE_MODEL via AI Studio (no logprobs needed for eval)."""
    return judge_debate(
        transcript,
        judge_profile="eval_judge",
        use_logprobs=False,
        model=config.QD_EVAL_JUDGE_MODEL,
    )


def fitness_from_judgement(judgement: dict) -> float | None:
    """Map a judgement to a scalar fitness for the archive.

    Returns -judge_log_odds_truth (continuous, unbounded; higher = more deceptive
    advantage). Returns None when log_odds_truth is None (Vertex logprobs
    unavailable and LLM estimator also failed), causing the caller to skip the
    archive update for this iteration.
    """
    log_odds = judgement.get("judge_log_odds_truth")
    if log_odds is None:
        return None
    return -float(log_odds)
