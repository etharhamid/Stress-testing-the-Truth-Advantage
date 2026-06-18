# core/consultancy_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Consultancy protocol (assigned, interactive) implementing Khan et al. /
# Kenton et al.'s single-consultant design:
#
#   Round 1 — Consultant argues blind (no prior transcript, no judge question).
#   Round N — Judge asks a probing question; consultant responds.
#
# The consultant always argues for answer_a.  The swap flag (passed from the
# entry point's balanced schedule) controls whether answer_a is the correct
# answer (swap=False) or the wrong one (swap=True), giving a 50/50
# correct / incorrect split analogous to the debate baseline.
#
# Sequential within each round (not parallel): judge_question_turn runs first,
# then consultant_turn — opposite of debate's simultaneous design.
# ─────────────────────────────────────────────────────────────────────────────

import re

from .gemini_client import chat
from .prompts import build_consultant_messages, build_judge_question_messages
from .debate_engine import extract_argument, truncate, verify_quotes

import config


# ── Transcript builder ────────────────────────────────────────────────────────
def build_consultancy_transcript(rounds: list[dict]) -> str:
    """
    Format completed consultancy rounds into the transcript string shown to
    the consultant (for context) and the final judge.

    Each round entry has:
      judge_question     : str | None  (None for round 0)
      consultant_verified: str         (quote-verified argument)
      consultant_label   : str         (e.g. 'Consultant (arguing for "...")')
    """
    lines = []
    for i, r in enumerate(rounds):
        lines.append(f"Round {i + 1}:\n\n")
        q = r.get("judge_question")
        if q:
            lines.append(f'Judge Question: """{q}"""\n\n')
        arg = r.get("consultant_verified", "")
        if arg:
            label = r.get("consultant_label", "Consultant")
            lines.append(f'{label}: """{arg}"""\n\n')
        if i + 1 < len(rounds):
            lines.append("---------------------------------------------\n\n")
    return "".join(lines).strip()


# ── Single consultant turn ────────────────────────────────────────────────────
def consultant_turn(
    *,
    question: str,
    answer_defending: str,
    answer_opposing: str,
    answer_a: str,
    answer_b: str,
    story: str,
    transcript_so_far: str,
    judge_question: str | None,
    round_num: int,
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """
    Call the consultant model for one turn and return the truncated,
    extracted argument.

    Uses the same extract_argument / truncate / verify_quotes utilities as
    debate_engine so formatting is consistent across protocols.

    `model` overrides `config.DEBATER_MODEL` for this single call without
    touching the module global — required for thread-safe parallel use.
    """
    messages = build_consultant_messages(
        question=question,
        answer_defending=answer_defending,
        answer_opposing=answer_opposing,
        answer_a=answer_a,
        answer_b=answer_b,
        story=story,
        transcript_so_far=transcript_so_far,
        judge_question=judge_question,
        round_num=round_num,
        word_limit=config.WORD_LIMIT,
    )
    raw = chat(
        model=model if model is not None else config.DEBATER_MODEL,
        messages=messages,
        temperature=temperature,
        seed=config.RANDOM_SEED,
        profile="debater",
    )
    argument = extract_argument(raw)
    argument = truncate(argument, config.WORD_LIMIT_HARD)
    return argument


# ── Judge in-round question turn ──────────────────────────────────────────────
def judge_question_turn(
    *,
    question: str,
    answer_a: str,
    answer_b: str,
    consultant_answer: str,
    transcript_so_far: str,
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """
    Call the judge model to generate an in-round probing question.

    Returns the text inside <question>…</question>, falling back to text
    after </reasoning> or the raw response when the tag is absent.

    Uses profile="eval_judge" (AI Studio, no logprobs — question generation
    needs no probability signal).
    """
    messages = build_judge_question_messages(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        consultant_answer=consultant_answer,
        transcript_so_far=transcript_so_far,
        word_limit=config.CONSULTANCY_JUDGE_QUESTION_WORD_LIMIT,
    )
    raw = chat(
        model=model if model is not None else "gemini-2.5-flash-lite",
        messages=messages,
        temperature=temperature,
        seed=config.RANDOM_SEED,
        profile="eval_judge",
    )
    m = re.search(r"<question>(.*?)</question>", raw, re.DOTALL)
    if m:
        question_text = m.group(1).strip()
    elif "</reasoning>" in raw:
        question_text = raw.split("</reasoning>", 1)[1].strip()
    else:
        question_text = raw.strip()

    # Hard-truncate to keep consultant context bounded (same truncation
    # utility used for debater arguments).
    return truncate(question_text, config.CONSULTANCY_JUDGE_QUESTION_WORD_LIMIT)


# ── Full consultancy session ──────────────────────────────────────────────────
def run_consultancy(
    row: dict,
    swap: bool = False,
    *,
    model: str | None = None,
    judge_question_model: str | None = None,
) -> dict:
    """
    Run a complete assigned consultancy session for a single question row.

    Protocol (3 rounds, sequential):
      Round 1 — Consultant argues with no prior transcript (blind opening).
      Round 2 — Judge asks probing question; consultant responds.
      Round 3 — Judge asks probing question; consultant responds.
      Final   — Judge decides (handled separately by consultancy_judge_engine).

    Args:
        row:                 A question dict loaded from questions.csv.
        swap:                If True, answer_a = negative_answer (deceptive
                             consultant); if False, answer_a = correct_answer
                             (honest consultant). Same 50/50 balance mechanism
                             as debate_engine.run_debate().
        model:               Optional override of config.DEBATER_MODEL for
                             the consultant's calls.
        judge_question_model: Optional override of the in-round judge model
                             (default: gemini-2.5-flash-lite).

    Returns:
        A dict with debate-compatible fields plus:
          consultant_answer  : str  — text the consultant argued for (= answer_a)
          consultant_correct : bool — True when consultant argues correctly
          consultant_letter  : "A"  — always A (consultant position is fixed)
    """
    question    = row["question"]
    correct_ans = row["correct_answer"]
    negative_ans = row["negative_answer"]
    story       = row["story"]

    if not swap:
        answer_a = correct_ans
        answer_b = negative_ans
    else:
        answer_a = negative_ans
        answer_b = correct_ans

    answer_defending = answer_a
    answer_opposing  = answer_b
    consultant_label = f'Consultant (arguing for "{answer_a}")'

    rounds: list[dict] = []

    for round_num in range(config.NUM_ROUNDS):
        transcript_so_far = build_consultancy_transcript(rounds)

        # Judge asks a probing question in all rounds after the first
        judge_question: str | None = None
        if round_num > 0:
            judge_question = judge_question_turn(
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                consultant_answer=answer_a,
                transcript_so_far=transcript_so_far,
                model=judge_question_model,
            )

        # Consultant responds (sees story + all prior rounds + current judge question)
        arg_raw = consultant_turn(
            question=question,
            answer_defending=answer_defending,
            answer_opposing=answer_opposing,
            answer_a=answer_a,
            answer_b=answer_b,
            story=story,
            transcript_so_far=transcript_so_far,
            judge_question=judge_question,
            round_num=round_num,
            model=model,
        )
        arg_verified = verify_quotes(arg_raw, story)

        rounds.append({
            "judge_question":      judge_question,
            "consultant_raw":      arg_raw,
            "consultant_verified": arg_verified,
            "consultant_label":    consultant_label,
        })

    transcript_str = build_consultancy_transcript(rounds)

    return {
        "id":               row["id"],
        "question":         question,
        "correct_answer":   correct_ans,
        "negative_answer":  negative_ans,
        "story_title":      row["story_title"],
        "question_set_id":  row.get("question_set_id", ""),
        "swap":             swap,
        "answer_a":         answer_a,
        "answer_b":         answer_b,
        "consultant_answer":  answer_a,
        "consultant_correct": not swap,
        "consultant_letter":  "A",
        "rounds":           rounds,
        "transcript_str":   transcript_str,
    }
