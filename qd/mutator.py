"""Mutator wrapper: builds messages, calls the LLM via chat_structured_json,
returns the parsed (mutated_question, mutated_distractor) dict or None on failure."""

from __future__ import annotations

from google.genai import types

import config

from core.gemini_client import chat_structured_json
from core.prompts import build_mutator_messages


_MUTATOR_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "mutated_question": types.Schema(type="STRING"),
        "mutated_distractor": types.Schema(type="STRING"),
    },
    required=["mutated_question", "mutated_distractor"],
    property_ordering=["mutated_question", "mutated_distractor"],
)


def call_mutator(
    *,
    parent_question: str,
    correct_answer: str,
    parent_distractor: str,
    mutation_type: str,
    target_question_type: str,
    seed: int | None = None,
) -> dict | None:
    """Returns a dict with str keys 'mutated_question' and 'mutated_distractor',
    or None on parse failure / schema miss / API error after retry."""
    messages = build_mutator_messages(
        mutation_type=mutation_type,
        target_question_type=target_question_type,
        original_question=parent_question,
        correct_answer=correct_answer,
        distractor=parent_distractor,
    )
    try:
        obj, _raw = chat_structured_json(
            model=config.QD_MUTATOR_MODEL,
            messages=messages,
            response_schema=_MUTATOR_SCHEMA,
            profile="debater",
            temperature=config.QD_MUTATOR_TEMPERATURE,
            seed=seed,
        )
    except Exception as exc:
        print(f"[mutator] {type(exc).__name__}: {exc}", flush=True)
        return None

    mq = obj.get("mutated_question")
    md = obj.get("mutated_distractor")
    if not isinstance(mq, str) or not isinstance(md, str):
        return None
    if not mq.strip() or not md.strip():
        return None
    return {"mutated_question": mq.strip(), "mutated_distractor": md.strip()}
