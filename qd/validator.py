"""Validator wrapper: returns True/False for whether a mutated question still
has the same correct answer. Conservative: any error → False."""

from __future__ import annotations

from google.genai import types

import config

from core.gemini_client import chat_structured_json
from core.prompts import build_classifier_messages, build_validator_messages
from .archive import QUESTION_TYPES


_CLASSIFIER_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "question_type": types.Schema(type="STRING"),
    },
    required=["question_type"],
    property_ordering=["question_type"],
)

_VALIDATOR_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "valid": types.Schema(type="BOOLEAN"),
    },
    required=["valid"],
    property_ordering=["valid"],
)


def call_validator(
    *,
    story: str,
    original_question: str,
    correct_answer: str,
    distractor: str,
    mutated_question: str,
    mutated_distractor: str,
    mutation_type: str,
    seed: int | None = None,
) -> bool:
    """Returns True iff the mutated pair is judged valid. Any error returns False."""
    messages = build_validator_messages(
        story=story,
        original_question=original_question,
        correct_answer=correct_answer,
        distractor=distractor,
        mutated_question=mutated_question,
        mutated_distractor=mutated_distractor,
        mutation_type=mutation_type,
    )
    try:
        obj, _raw = chat_structured_json(
            model=config.QD_VALIDATOR_MODEL,
            messages=messages,
            response_schema=_VALIDATOR_SCHEMA,
            profile="qd_validator",
            temperature=config.QD_VALIDATOR_TEMPERATURE,
            seed=seed,
        )
    except Exception:
        return False

    valid = obj.get("valid")
    return bool(valid) if isinstance(valid, bool) else False


def classify_question_type(
    mutated_question: str,
    seed: int | None = None,
    target_type: str | None = None,
) -> str | None:
    """Return the QUESTION_TYPES label for mutated_question, or None on any failure.

    When target_type is given the classifier is told the intended type, making
    the check a validation rather than a from-scratch classification.
    """
    messages = build_classifier_messages(question=mutated_question, target_type=target_type)
    try:
        obj, _raw = chat_structured_json(
            model=config.QD_VALIDATOR_MODEL,
            messages=messages,
            response_schema=_CLASSIFIER_SCHEMA,
            profile="qd_validator",
            temperature=config.QD_VALIDATOR_TEMPERATURE,
            seed=seed,
        )
    except Exception:
        return None

    qt = obj.get("question_type")
    if not isinstance(qt, str):
        return None
    qt = qt.strip()
    return qt if qt in QUESTION_TYPES else None
