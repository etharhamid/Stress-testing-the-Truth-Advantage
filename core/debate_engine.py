# core/debate_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Core debate logic implementing the protocol from the spec:
#   Round 1 — Simultaneous blind opening statements
#   Round 2 — Simultaneous rebuttals (each sees opponent's Round 1 only)
# ─────────────────────────────────────────────────────────────────────────────

import re
from concurrent.futures import ThreadPoolExecutor

from .gemini_client import chat
from .prompts import build_debater_messages

import config


# ── Quote verification ────────────────────────────────────────────────────────
# Canonicalization table: model-rendered smart quotes / dashes / NBSPs
# frequently differ from the plain-ASCII story source by a single codepoint,
# which would otherwise flip a legitimate quote to <u_quote>.
_QUOTE_NORMALIZATION = str.maketrans({
    "\u2018": "'",   # left single quote  ‘
    "\u2019": "'",   # right single quote ’
    "\u201C": '"',   # left double quote  “
    "\u201D": '"',   # right double quote ”
    "\u2013": "-",   # en dash  –
    "\u2014": "-",   # em dash  —
    "\u00A0": " ",   # non-breaking space
    "\u2028": "",    # line separator
    "\u2029": "",    # paragraph separator
})


def _normalize_for_match(text: str) -> str:
    """Canonicalize smart punctuation and collapse whitespace for substring match."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_NORMALIZATION)).strip()


def verify_quotes(argument: str, story: str) -> str:
    """
    Replace every <quote>...</quote> with either <v_quote> (verified — the quoted
    span appears in the story after whitespace / smart-quote normalization) or
    <u_quote> (unverified — not found).
    """
    normalized_story = _normalize_for_match(story)

    def _replace(match: re.Match) -> str:
        raw = match.group(1)
        if _normalize_for_match(raw) in normalized_story:
            return f"<v_quote>{raw}</v_quote>"
        return f"<u_quote>{raw}</u_quote>"

    return re.sub(r"<quote>(.*?)</quote>", _replace, argument, flags=re.DOTALL)


# ── Argument extraction ───────────────────────────────────────────────────────
def extract_argument(response: str) -> str:
    """
    Pull the text inside <argument>...</argument>.
    Falls back to the full response if the tags are missing.
    """
    for tag in ["v_quote", "u_quote"]:
        response = response.replace(f"<{tag}>", "<quote>").replace(f"</{tag}>", "</quote>")

    if "<argument>" in response and "</argument>" in response:
        return response.split("<argument>")[1].split("</argument>")[0].strip()

    if "</thinking>" in response:
        after = response.split("</thinking>", 1)[1].strip()
        if after:
            return after

    return response.strip()


# ── Word-limit enforcement ────────────────────────────────────────────────────
def truncate(argument: str, word_limit: int) -> str:
    words = argument.split()
    if len(words) <= word_limit:
        return argument
    truncated = " ".join(words[:word_limit])
    for tag in ["quote", "v_quote", "u_quote"]:
        if f"<{tag}>" in truncated and f"</{tag}>" not in truncated:
            truncated += f"</{tag}>"
    return truncated + "... <TRUNCATED>"


# ── Transcript builder ────────────────────────────────────────────────────────
def build_transcript_string(rounds: list[dict]) -> str:
    """
    Format completed rounds into the transcript string shown to debaters and
    the judge.  Uses the verified versions of each argument.
    """
    lines = []
    for i, r in enumerate(rounds):
        lines.append(f"Round {i + 1}:\n\n")
        a_arg = r.get("debater_a_verified", "")
        b_arg = r.get("debater_b_verified", "")
        if a_arg:
            lines.append(f'Debater A: """{a_arg}"""\n\n')
        if b_arg:
            lines.append(f'Debater B: """{b_arg}"""\n\n')
        if i + 1 < len(rounds):
            lines.append("---------------------------------------------\n\n")
    return "".join(lines).strip()


# ── Single debater turn ───────────────────────────────────────────────────────
def debater_turn(
    *,
    name: str,
    opponent_name: str,
    question: str,
    answer_defending: str,
    answer_opposing: str,
    answer_a: str,
    answer_b: str,
    story: str,
    transcript_so_far: str,
    round_num: int,
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """
    Call the debater model for one turn and return the truncated, extracted
    argument.

    `model` overrides `config.DEBATER_MODEL` for this single call without
    touching the module global — required for thread-safe parallel use.
    """
    messages = build_debater_messages(
        name=name,
        opponent_name=opponent_name,
        question=question,
        answer_defending=answer_defending,
        answer_opposing=answer_opposing,
        answer_a=answer_a,
        answer_b=answer_b,
        story=story,
        transcript_so_far=transcript_so_far,
        round_num=round_num,
        word_limit=config.WORD_LIMIT,
    )

    raw_response = chat(
        model=model if model is not None else config.DEBATER_MODEL,
        messages=messages,
        temperature=temperature,
        seed=config.RANDOM_SEED,
        profile="debater",
    )

    argument = extract_argument(raw_response)
    argument = truncate(argument, config.WORD_LIMIT_HARD)
    return argument


# ── Full debate ───────────────────────────────────────────────────────────────
def run_debate(row: dict, swap: bool = False, *, model: str | None = None) -> dict:
    """
    Run a complete debate for a single question row.

    Protocol (per spec):
      Round 1 — Both debaters argue simultaneously with no transcript (blind).
      Round 2+ — Both debaters see the transcript of all completed rounds, then
                 argue simultaneously (neither sees the other's current argument).

    Args:
        row:   A question dict from quality_loader.load_questions()
        swap:  If True, the correct answer is assigned to position B;
               otherwise it is assigned to position A.
        model: Optional debater model override threaded into every
               `debater_turn` call. When None, `config.DEBATER_MODEL` is used.
               Pass an explicit value from worker threads instead of mutating
               the module global.

    Returns:
        A dict with debate metadata, round records, and the final transcript.
    """
    question     = row["question"]
    correct_ans  = row["correct_answer"]
    negative_ans = row["negative_answer"]
    story        = row["story"]

    # ── Assign positions based on swap flag ───────────────────────────────
    if not swap:
        answer_a = correct_ans
        answer_b = negative_ans
    else:
        answer_a = negative_ans
        answer_b = correct_ans

    name_a = "Debater A"
    name_b = "Debater B"

    rounds: list[dict] = []

    for round_num in range(config.NUM_ROUNDS):
        # Both debaters see the SAME transcript (all completed rounds only).
        # Built ONCE per round before either debater speaks, so neither sees
        # the other's current-round argument — preserves the "simultaneous"
        # protocol exactly. Round 1: empty → both argue blind. Round 2+:
        # contains all prior completed rounds.
        transcript_str = build_transcript_string(rounds)

        # ── Debaters A and B speak simultaneously (parallel) ──────────────
        # Both calls are pure with respect to round state: they read
        # `transcript_str` (immutable string) and call independent paths
        # through the pooled gemini client. The output dict is assembled
        # only after both futures resolve, so ordering of debater_a_* /
        # debater_b_* fields in `rounds` matches the sequential version.
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(
                debater_turn,
                name=name_a,
                opponent_name=name_b,
                question=question,
                answer_defending=answer_a,
                answer_opposing=answer_b,
                answer_a=answer_a,
                answer_b=answer_b,
                story=story,
                transcript_so_far=transcript_str,
                round_num=round_num,
                model=model,
            )
            fut_b = ex.submit(
                debater_turn,
                name=name_b,
                opponent_name=name_a,
                question=question,
                answer_defending=answer_b,
                answer_opposing=answer_a,
                answer_a=answer_a,
                answer_b=answer_b,
                story=story,
                transcript_so_far=transcript_str,
                round_num=round_num,
                model=model,
            )
            arg_a_raw = fut_a.result()
            arg_b_raw = fut_b.result()

        arg_a_verified = verify_quotes(arg_a_raw, story)
        arg_b_verified = verify_quotes(arg_b_raw, story)

        rounds.append({
            "debater_a_raw":      arg_a_raw,
            "debater_a_verified": arg_a_verified,
            "debater_b_raw":      arg_b_raw,
            "debater_b_verified": arg_b_verified,
        })

    # Build the final transcript string for the judge
    transcript_str = build_transcript_string(rounds)

    return {
        "id":              row["id"],
        "question":        question,
        "correct_answer":  correct_ans,
        "negative_answer": negative_ans,
        "story_title":     row["story_title"],
        "question_set_id": row.get("question_set_id", ""),
        "swap":            swap,
        "answer_a":        answer_a,
        "answer_b":        answer_b,
        "name_a":          name_a,
        "name_b":          name_b,
        "rounds":          rounds,
        "transcript_str":  transcript_str,
    }
