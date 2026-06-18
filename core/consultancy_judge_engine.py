# core/consultancy_judge_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Judges a completed consultancy transcript.
#
# Unlike judge_engine.judge_debate (which uses Vertex AI with logprobs for the
# continuous QD fitness signal), this module always uses AI Studio with
# use_logprobs=False.  Consultancy eval metrics are binary TWR — the log-odds
# signal is only needed for the QD search phase.  All logprob fields in the
# returned dict are therefore None.
# ─────────────────────────────────────────────────────────────────────────────

from .gemini_client import chat, chat_judge_with_reasoning
from .judge_engine import _parse_judge_answer, JUDGE_TEMPERATURE
from .prompts import build_consultancy_judge_messages

import config

_JUDGE_CHOICES = ["A", "B"]

# Default model for consultancy judging (AI Studio, no logprobs).
_DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


def judge_consultancy(
    consultancy_result: dict,
    *,
    model: str | None = None,
) -> dict:
    """
    Judge a completed consultancy session.

    Primary path: `chat_judge_with_reasoning()` with `use_logprobs=False`,
    profile="eval_judge" (AI Studio). The judge prompt instructs the model to
    write chain-of-thought inside <reasoning> then commit to a letter inside
    <answer>.

    Fallback: on any exception, fall back to plain chat() + _parse_judge_answer().
    All logprob-derived fields are None in both paths.

    Args:
        consultancy_result: output dict from consultancy_engine.run_consultancy()
        model: optional override of the default flash-lite judge model.

    Returns:
        The same dict with added keys matching judge_engine.judge_debate() output:
            judge_raw_response, judge_answer ("A"/"B"/None),
            judge_confidence (None), judge_logprob_a (None),
            judge_logprob_b (None), judge_log_odds_truth (None),
            judge_logit_gap_truth (None),
            judge_correct (bool | None), correct_letter,
            judge_fallback (bool),
            consultant_correct (bool — carried through from run_consultancy).
    """
    model_id = model if model is not None else _DEFAULT_JUDGE_MODEL

    consultant_letter = consultancy_result.get("consultant_letter", "A")
    consultant_answer = consultancy_result.get(
        "consultant_answer", consultancy_result.get("answer_a", "")
    )

    messages = build_consultancy_judge_messages(
        question=consultancy_result["question"],
        answer_a=consultancy_result["answer_a"],
        answer_b=consultancy_result["answer_b"],
        consultant_letter=consultant_letter,
        consultant_answer=consultant_answer,
        transcript=consultancy_result["transcript_str"],
    )

    fallback_used = False
    raw = ""
    judge_answer: str | None = None

    try:
        answer, _, raw = chat_judge_with_reasoning(
            model=model_id,
            messages=messages,
            enum_values=_JUDGE_CHOICES,
            profile="eval_judge",
            temperature=JUDGE_TEMPERATURE,
            seed=config.RANDOM_SEED,
            use_logprobs=False,
        )
        judge_answer = answer if answer in _JUDGE_CHOICES else None
        # chat_judge_with_reasoning already parses the <answer> tag; fall back
        # to the regex chain if the returned token isn't a valid choice.
        if judge_answer is None:
            judge_answer = _parse_judge_answer(raw)
    except Exception:
        fallback_used = True
        try:
            raw = chat(
                model=model_id,
                messages=messages,
                temperature=JUDGE_TEMPERATURE,
                seed=config.RANDOM_SEED,
                profile="eval_judge",
            )
            judge_answer = _parse_judge_answer(raw)
        except Exception:
            raw = ""
            judge_answer = None

    swap = consultancy_result["swap"]
    correct_letter = "B" if swap else "A"
    judge_correct = (judge_answer == correct_letter) if judge_answer is not None else None

    result = dict(consultancy_result)
    result["judge_raw_response"]   = raw
    result["judge_answer"]         = judge_answer
    result["judge_confidence"]     = None
    result["judge_logprob_a"]      = None
    result["judge_logprob_b"]      = None
    result["judge_log_odds_truth"] = None
    result["judge_logit_gap_truth"] = None
    result["judge_correct"]        = judge_correct
    result["correct_letter"]       = correct_letter
    result["judge_fallback"]       = fallback_used
    return result
