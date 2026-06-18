# core/gemini_client.py
# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini via the google-genai SDK, with a backend split by role:
#   - Debater  → AI Studio (API key)       — broader preview access
#   - Judge    → Vertex AI (ADC)           — response_logprobs required
# Auth (AI Studio):  GOOGLE_API_KEY  or comma-sep GOOGLE_API_KEYS
# Auth (Vertex AI):  Application Default Credentials + GOOGLE_CLOUD_PROJECT
#                    (or comma-sep GOOGLE_CLOUD_PROJECTS) + GOOGLE_CLOUD_LOCATION
#                    (defaults to us-central1)
#
# Pooling and failover
# --------------------
# Each profile holds a `_Pool` of `_PoolEntry` objects — one per Vertex
# project (for `judge`) or one per AI Studio key (for the other profiles).
# `_acquire(profile)` returns the next non-cooling entry via round-robin
# (per-profile pointer, protected by `_Pool.lock`). When all entries are
# cooling, the caller sleeps until the soonest cooldown expires and retries.
#
# Failover is per-attempt: each entry point's tenacity-retried `_call()`
# closure acquires a fresh entry on every attempt. On `RESOURCE_EXHAUSTED` /
# `QUOTA` / 429 the offending entry is marked cooling for 60 s before the
# exception is re-raised, so tenacity's next attempt routes around it. Other
# transient errors (503/504/UNAVAILABLE/DEADLINE_EXCEEDED) do NOT mark
# cooldown — they're handled by tenacity's exponential backoff alone.
#
# When no env-var list is set, the pool falls back to a 1-entry list built
# from the legacy singleton (`GOOGLE_API_KEY` / `GOOGLE_CLOUD_PROJECT`), so
# existing single-credential setups behave exactly as before.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config

Profile = Literal["debater", "judge", "eval_judge", "qd_validator"]

# How long an entry stays cooling after a quota / 429 / RESOURCE_EXHAUSTED.
# Long enough to amortise the per-minute window most quotas use; short enough
# that a brief spike doesn't shelve the entry for a noticeable wall-clock gap.
_COOLDOWN_SECONDS = 60.0


@dataclass
class _PoolEntry:
    client: genai.Client
    label: str              # short identifier for logs (last 4 of key/proj)
    cooldown_until: float = 0.0
    disabled_reason: str | None = None


@dataclass
class _Pool:
    entries: list[_PoolEntry]
    lock: threading.Lock = field(default_factory=threading.Lock)
    rr_index: int = 0


_pools: dict[str, _Pool] = {}
_pools_lock = threading.Lock()


def _tail_label(s: str, n: int = 4) -> str:
    """Last `n` chars, prefixed with `...`. Empty/short strings round-trip."""
    if not s:
        return "(empty)"
    return f"...{s[-n:]}" if len(s) > n else s


def _parse_json_object_text(raw_text: str) -> dict:
    """Parse a JSON object, tolerating common markdown fence wrappers.

    Some schema-constrained models still return otherwise-valid JSON with a
    trailing ``` fence or a short preamble. Keep recovery conservative: accept
    only an object root and still raise when no complete object can be decoded.
    """
    text = raw_text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(obj, dict):
            return obj
        raise RuntimeError(
            f"Structured JSON call returned non-object root: {raw_text[:200]!r}"
        )

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj
            raise RuntimeError(
                f"Structured JSON call returned non-object root: {raw_text[:200]!r}"
            )

    start = text.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj
            raise RuntimeError(
                f"Structured JSON call returned non-object root: {raw_text[:200]!r}"
            )

    raise RuntimeError(f"Structured JSON call returned non-JSON: {raw_text[:200]!r}")


def _build_pool(profile: Profile) -> _Pool:
    """Construct the per-profile pool from current config. Called once per
    profile, then cached in `_pools`."""
    # eval_judge and judge carry full story passages (~30K chars) + debate
    # transcripts in each request; flash-lite can take >120s under load.
    timeout_ms = 300_000 if profile in ("eval_judge", "judge") else 120_000
    http_opts = types.HttpOptions(timeout=timeout_ms)

    entries: list[_PoolEntry] = []
    if profile == "judge":
        if not config.GCP_PROJECTS:
            raise RuntimeError(
                "core.gemini_client._build_pool(judge): no Vertex projects "
                "configured. Set GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECTS."
            )
        for project_id in config.GCP_PROJECTS:
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=config.GCP_LOCATION,
                http_options=http_opts,
            )
            entries.append(_PoolEntry(client=client, label=_tail_label(project_id)))
    else:  # "debater", "eval_judge", or "qd_validator" → AI Studio
        if not config.GOOGLE_API_KEYS:
            raise RuntimeError(
                f"core.gemini_client._build_pool({profile}): no AI Studio keys "
                "configured. Set GOOGLE_API_KEY or GOOGLE_API_KEYS."
            )
        for api_key in config.GOOGLE_API_KEYS:
            client = genai.Client(
                api_key=api_key,
                http_options=http_opts,
            )
            entries.append(_PoolEntry(client=client, label=_tail_label(api_key)))

    return _Pool(entries=entries)


def _get_pool(profile: Profile) -> _Pool:
    """Return the per-profile pool, building it on first access."""
    if profile in _pools:
        return _pools[profile]
    with _pools_lock:
        if profile not in _pools:
            _pools[profile] = _build_pool(profile)
        return _pools[profile]


def _acquire(profile: Profile) -> _PoolEntry:
    """Return the next non-cooling, non-disabled pool entry via round-robin.

    If every live entry is cooling, sleep until the earliest cooldown expires
    and retry. Permanently invalid entries are skipped; if all entries are
    disabled, fail loudly so the user can renew/remove the credentials.
    """
    pool = _get_pool(profile)
    while True:
        with pool.lock:
            now = time.time()
            n = len(pool.entries)
            live_entries = [e for e in pool.entries if e.disabled_reason is None]
            if not live_entries:
                disabled = ", ".join(
                    f"{e.label} ({e.disabled_reason})" for e in pool.entries
                )
                raise RuntimeError(
                    f"No active credentials left for profile={profile!r}; "
                    f"disabled entries: {disabled}"
                )
            for i in range(n):
                idx = (pool.rr_index + i) % n
                e = pool.entries[idx]
                if e.disabled_reason is None and e.cooldown_until <= now:
                    pool.rr_index = (idx + 1) % n
                    return e
            soonest = min(e.cooldown_until for e in live_entries)
        # All entries cooling — wait until the soonest expires. Clamp to a
        # sensible range so we don't spin if clocks drift or sleep forever
        # if a cooldown is stuck far in the future.
        wait = max(0.1, min(60.0, soonest - time.time()))
        time.sleep(wait)


def _mark_cooling(entry: _PoolEntry, seconds: float = _COOLDOWN_SECONDS) -> None:
    """Set this entry's cooldown timestamp. Single float write — atomic in
    CPython, so no lock required; at worst two threads race and one's mark
    wins (both extend cooldown to roughly the same instant)."""
    entry.cooldown_until = time.time() + seconds


def _mark_disabled(entry: _PoolEntry, reason: str) -> None:
    """Permanently remove an invalid credential from rotation for this process."""
    if entry.disabled_reason is None:
        print(f"[pool] disabling credential {entry.label}: {reason}", flush=True)
    entry.disabled_reason = reason


def _is_quota_error(e: Exception) -> bool:
    """True iff the exception looks like a per-key / per-project rate limit.
    Subset of `_is_retryable` — quota errors trigger pool cooldown; other
    transient errors (503/504/UNAVAILABLE/DEADLINE_EXCEEDED) do not, since
    they are typically backend hiccups rather than client-side rate limits."""
    msg = str(e).upper()
    return any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "QUOTA"))


def _is_invalid_api_key_error(e: Exception) -> bool:
    """True for permanent AI Studio API-key failures such as expired keys."""
    msg = str(e).upper()
    return (
        "API_KEY_INVALID" in msg
        or "API KEY EXPIRED" in msg
        or "API_KEY_EXPIRED" in msg
    )


def pool_summary() -> str:
    """One-line summary per profile, e.g.
       "judge: 2 Vertex projects [...4d2a, ...e7c1]; debater: 4 AI Studio keys; ..."
    Builds any uninitialised pools as a side effect. Safe to call from any
    thread — uses the same per-profile locks as `_acquire`."""
    parts: list[str] = []
    for profile in ("judge", "debater", "eval_judge", "qd_validator"):
        try:
            pool = _get_pool(profile)  # type: ignore[arg-type]
        except Exception as e:
            parts.append(f"{profile}: error ({e!r})")
            continue
        labels = ", ".join(
            f"{e.label}{' disabled' if e.disabled_reason else ''}"
            for e in pool.entries
        )
        backend = "Vertex projects" if profile == "judge" else "AI Studio keys"
        parts.append(f"{profile}: {len(pool.entries)} {backend} [{labels}]")
    return "; ".join(parts)


# Per-project per-minute throughput we expect from gemini-2.5-flash-lite on
# Vertex — used only as a rule-of-thumb in the diagnostic banner so the user
# can sanity-check whether a chosen `--workers` is going to saturate quota.
# Not authoritative; real quotas vary by project tier and current spike.
_VERTEX_RPM_RULE_OF_THUMB = 250


def format_pool_diagnostics(workers: int | None = None) -> str:
    """Multi-line startup banner showing how many credentials each profile
    is pooling and (optionally) the effective Vertex RPM headroom for a
    given worker count. Called from each run_* entry point right after
    `config.validate_llm_credentials()` so the log makes pooling state
    obvious at a glance."""
    lines = ["[pool] " + s for s in pool_summary().split("; ")]
    if workers is not None:
        n_proj = len(config.GCP_PROJECTS)
        headroom = n_proj * _VERTEX_RPM_RULE_OF_THUMB
        lines.append(f"[pool] effective workers: {workers}")
        lines.append(
            f"[pool] Vertex RPM headroom (rule of thumb): "
            f"~{n_proj} × {_VERTEX_RPM_RULE_OF_THUMB} = {headroom}"
        )
    return "\n".join(lines)


def _openai_messages_to_turns(
    messages: list[dict],
) -> tuple[str | None, list[tuple[str, str]]]:
    """
    Map prompts.py OpenAI-style messages to (system_instruction, conversation turns).

    Each turn is (role, text) with role 'user' or 'model'. Must end with 'user'.
    """
    system_chunks: list[str] = []
    turns: list[tuple[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        if role == "system":
            system_chunks.append(text)
        elif role == "assistant":
            turns.append(("model", text))
        else:
            turns.append(("user", text))

    system_instruction = "\n\n".join(system_chunks) if system_chunks else None

    if not turns:
        raise RuntimeError("Gemini request has no user/assistant messages")
    if turns[-1][0] != "user":
        raise RuntimeError("Gemini expects the last message to be from the user")

    return system_instruction, turns


def _contents_from_turns(turns: list[tuple[str, str]]) -> list[types.Content]:
    return [
        types.Content(
            role=r,
            parts=[types.Part.from_text(text=t)],
        )
        for r, t in turns
    ]


def _is_retryable(e: Exception) -> bool:
    msg = str(e).upper()
    if _is_invalid_api_key_error(e):
        return True
    return any(
        k in msg
        for k in (
            "503", "429", "504",
            "UNAVAILABLE", "RESOURCE_EXHAUSTED", "QUOTA", "DEADLINE_EXCEEDED",
        )
    )


def _text_from_response(response: types.GenerateContentResponse) -> str:
    try:
        text = response.text
    except ValueError as e:
        raise RuntimeError(
            f"Gemini returned no text (blocked or empty). "
            f"prompt_feedback={getattr(response, 'prompt_feedback', None)!r}"
        ) from e
    if not text:
        raise RuntimeError("Gemini returned empty text")
    return text


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.4,
    seed: int | None = None,
    *,
    profile: Profile = "debater",
    max_output_tokens: int | None = None,
) -> str:
    """
    Run Gemini using the same OpenAI-shaped message lists as prompts.py.
    Retries up to 5 times with exponential backoff on transient 503/429/504
    errors; on quota errors the failed pool entry is also marked cooling so
    the next attempt routes to a different key/project.
    """
    system_instruction, turns = _openai_messages_to_turns(messages)
    contents = _contents_from_turns(turns)

    cfg_kwargs: dict = {"temperature": temperature}
    if system_instruction is not None:
        cfg_kwargs["system_instruction"] = system_instruction
    if seed is not None:
        cfg_kwargs["seed"] = seed
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens
    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_is_retryable),
    )
    def _call() -> str:
        entry = _acquire(profile)
        try:
            response = entry.client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
        except Exception as e:
            if _is_invalid_api_key_error(e):
                _mark_disabled(entry, "invalid/expired API key")
                raise
            if _is_quota_error(e):
                _mark_cooling(entry)
            raise
        return _text_from_response(response)

    try:
        return _call()
    except Exception as e:
        raise RuntimeError(
            f"Gemini API call failed (profile={profile!r}, model={model!r}): {e}"
        ) from e


def _extract_token_logprobs(
    response: types.GenerateContentResponse,
    allowed_tokens: list[str],
) -> dict[str, float] | None:
    """Return a dict mapping every allowed token to its log-probability at the
    first generated position, or None if Vertex omitted logprobs_result.
    Never raises.

    Reads `top_candidates[0].candidates` — because `chat_structured` requests
    `logprobs: len(allowed_tokens)` under constrained enum decoding, all
    allowed tokens should appear in the top-K list. Missing tokens are simply
    absent from the returned dict; the caller decides how to handle partial
    coverage (e.g., `judge_engine` treats a missing non-chosen logprob the
    same as a missing logprob overall when computing log_odds_truth).

    If `top_candidates` is absent but `chosen_candidates[0]` is present, falls
    back to a single-entry dict keyed on the chosen token, so the legacy
    confidence signal still works even when the full top-K is unavailable.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        lr = getattr(candidates[0], "logprobs_result", None)
        if lr is None:
            return None

        out: dict[str, float] = {}

        top = getattr(lr, "top_candidates", None) or []
        if top:
            inner = getattr(top[0], "candidates", None) or []
            allowed_set = set(allowed_tokens)
            for cand in inner:
                tok = getattr(cand, "token", None)
                lp = getattr(cand, "log_probability", None)
                if tok is not None and lp is not None and tok in allowed_set:
                    out[str(tok)] = float(lp)

        # Backstop: if top_candidates didn't cover the chosen token, try to
        # pick it up from chosen_candidates so the legacy confidence signal
        # still works.
        if not out:
            chosen = getattr(lr, "chosen_candidates", None) or []
            if chosen:
                tok = getattr(chosen[0], "token", None)
                lp = getattr(chosen[0], "log_probability", None)
                if tok is not None and lp is not None:
                    out[str(tok)] = float(lp)

        return out or None
    except Exception:
        return None


def _build_enum_type(enum_values: list[str]) -> type[Enum]:
    """Dynamic str-Enum for Gemini's response_schema. Members must be valid
    Python identifiers; A/B already are, so the mapping is identity."""
    members = {v: v for v in enum_values}
    return Enum("JudgeChoice", members, type=str)  # type: ignore[return-value]


def chat_structured(
    model: str,
    messages: list[dict],
    enum_values: list[str],
    *,
    profile: Profile = "judge",
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[str, dict[str, float] | None]:
    """
    Run a Gemini call constrained to one of `enum_values` (e.g. ["A", "B"])
    and return (answer, logprobs_by_token).

    `logprobs_by_token` maps each requested token to its log-probability at
    the first generated position (e.g. `{"A": -0.12, "B": -2.13}`), or None
    if Vertex omitted logprobs_result for this response. Callers derive the
    signals they need:
      - `confidence = exp(logprobs_by_token[answer])` (probability in [0,1])
      - `log_odds = logprobs_by_token[a] - logprobs_by_token[b]`
    and handle a None / partial dict by falling back to a text-mode judge.

    Retries transient 503/429/504/quota errors up to 5 times with exponential
    backoff — identical policy to `chat()`.
    """
    if not enum_values:
        raise ValueError("enum_values must be non-empty")

    system_instruction, turns = _openai_messages_to_turns(messages)
    contents = _contents_from_turns(turns)
    enum_cls = _build_enum_type(enum_values)

    cfg_kwargs: dict = {
        "temperature": temperature,
        "response_schema": enum_cls,
        "response_mime_type": "text/x.enum",
        "response_logprobs": True,
        "logprobs": len(enum_values),
    }
    if system_instruction is not None:
        cfg_kwargs["system_instruction"] = system_instruction
    if seed is not None:
        cfg_kwargs["seed"] = seed
    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_is_retryable),
    )
    def _call() -> tuple[str, dict[str, float] | None]:
        entry = _acquire(profile)
        try:
            response = entry.client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
        except Exception as e:
            if _is_invalid_api_key_error(e):
                _mark_disabled(entry, "invalid/expired API key")
                raise
            if _is_quota_error(e):
                _mark_cooling(entry)
            raise
        answer = _text_from_response(response).strip()
        logprobs_by_token = _extract_token_logprobs(response, enum_values)
        return answer, logprobs_by_token

    try:
        return _call()
    except Exception as e:
        raise RuntimeError(
            f"Gemini structured call failed (profile={profile!r}, model={model!r}): {e}"
        ) from e


def _extract_xml_answer_logprobs(
    response: types.GenerateContentResponse,
    enum_values: list[str],
) -> dict[str, float] | None:
    """Return {token: logprob} at the answer-token position of a free-text
    <answer>X</answer> response, or None if not locatable. Never raises.

    Scans chosen_candidates from the end for the last position whose chosen
    token is in enum_values (the answer letter), then collects top-K logprobs
    at that same position from top_candidates.
    """
    try:
        candidates_list = getattr(response, "candidates", None) or []
        if not candidates_list:
            return None
        lr = getattr(candidates_list[0], "logprobs_result", None)
        if lr is None:
            return None

        allowed = set(enum_values)
        chosen = getattr(lr, "chosen_candidates", None) or []
        top = getattr(lr, "top_candidates", None) or []

        # Find the last position (scanning from end) where chosen token is in enum.
        # The <answer>X</answer> is at the end of the response, so the last
        # occurrence of an A/B token is the answer commitment.
        target_idx: int | None = None
        for i in range(len(chosen) - 1, -1, -1):
            tok = getattr(chosen[i], "token", None)
            if tok is not None and tok in allowed:
                target_idx = i
                break

        if target_idx is None:
            return None

        out: dict[str, float] = {}

        # Collect top-K logprobs at target position, filtering to enum_values.
        if target_idx < len(top):
            inner = getattr(top[target_idx], "candidates", None) or []
            for cand in inner:
                tok = getattr(cand, "token", None)
                lp = getattr(cand, "log_probability", None)
                if tok is not None and lp is not None and tok in allowed:
                    out[str(tok)] = float(lp)

        # Backstop: use chosen token's own logprob if top-K didn't cover it.
        if not out:
            tok = getattr(chosen[target_idx], "token", None)
            lp = getattr(chosen[target_idx], "log_probability", None)
            if tok is not None and lp is not None:
                out[str(tok)] = float(lp)

        return out or None
    except Exception:
        return None


def chat_judge_with_reasoning(
    model: str,
    messages: list[dict],
    enum_values: list[str],
    *,
    profile: Profile = "judge",
    temperature: float = 0.1,
    seed: int | None = None,
    use_logprobs: bool = True,
    max_output_tokens: int | None = None,
) -> tuple[str, dict[str, float] | None, str]:
    """
    Judge call with XML-structured reasoning-then-answer output.

    The judge prompt instructs the model to emit:
        <reasoning>...</reasoning>
        <answer>A</answer>

    Uses free-text generation (no response_schema) with response_logprobs=True.
    Vertex AI returns logprobs for plain text but rejects them under
    application/json schema mode. Returns (answer, logprobs_by_token, raw_text):
      - answer           : one of enum_values, extracted from <answer> tag.
      - logprobs_by_token: dict mapping enum values to logprobs at the answer
                           token position, or None if Vertex omitted logprobs_result.
      - raw_text         : full response text (for judge_raw_response).

    Retries transient 503/429/504/quota errors up to 5 times with
    exponential backoff — identical policy to `chat()`.
    """
    if not enum_values:
        raise ValueError("enum_values must be non-empty")

    system_instruction, turns = _openai_messages_to_turns(messages)
    contents = _contents_from_turns(turns)

    cfg_kwargs: dict = {"temperature": temperature}
    if use_logprobs:
        cfg_kwargs["response_logprobs"] = True
        cfg_kwargs["logprobs"] = max(len(enum_values), 5)
    if system_instruction is not None:
        cfg_kwargs["system_instruction"] = system_instruction
    if seed is not None:
        cfg_kwargs["seed"] = seed
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens
    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_is_retryable),
    )
    def _call() -> tuple[str, dict[str, float] | None, str]:
        entry = _acquire(profile)
        try:
            response = entry.client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
        except Exception as e:
            if _is_invalid_api_key_error(e):
                _mark_disabled(entry, "invalid/expired API key")
                raise
            if _is_quota_error(e):
                _mark_cooling(entry)
            raise
        raw_text = _text_from_response(response)
        m = re.search(r"<answer>\s*([A-Z])\s*</answer>", raw_text, re.IGNORECASE)
        if m is None:
            raise RuntimeError(
                f"Judge response missing <answer> tag: {raw_text[:300]!r}"
            )
        answer = m.group(1).upper()
        if answer not in enum_values:
            raise RuntimeError(
                f"Judge <answer> is {answer!r}, not in {enum_values}: {raw_text[:300]!r}"
            )
        logprobs_by_token = _extract_xml_answer_logprobs(response, enum_values)
        return answer, logprobs_by_token, raw_text

    try:
        return _call()
    except Exception as e:
        raise RuntimeError(
            f"Gemini judge-with-reasoning call failed "
            f"(profile={profile!r}, model={model!r}): {e}"
        ) from e


def chat_structured_json(
    model: str,
    messages: list[dict],
    response_schema: types.Schema,
    *,
    profile: Profile = "debater",
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[dict, str]:
    """
    Generic JSON-schema-constrained call. Returns (parsed_dict, raw_json_text).

    Used by the QD mutator and validator. Reuses the same retry policy as the
    other entry points. On JSON-decode failure raises RuntimeError so the
    caller's retry loop can decide whether to discard the candidate.
    """
    system_instruction, turns = _openai_messages_to_turns(messages)
    contents = _contents_from_turns(turns)

    cfg_kwargs: dict = {
        "temperature": temperature,
        "response_schema": response_schema,
        "response_mime_type": "application/json",
    }
    if system_instruction is not None:
        if model.startswith("gemma-"):
            # Gemma doesn't support system_instruction; fold it into the first user turn.
            if contents and contents[0].parts:
                merged = system_instruction + "\n\n" + (contents[0].parts[0].text or "")
                contents[0].parts[0] = types.Part.from_text(text=merged)
        else:
            cfg_kwargs["system_instruction"] = system_instruction
    if seed is not None:
        cfg_kwargs["seed"] = seed
    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(_is_retryable),
    )
    def _call() -> tuple[dict, str]:
        entry = _acquire(profile)
        try:
            response = entry.client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
        except Exception as e:
            if _is_invalid_api_key_error(e):
                _mark_disabled(entry, "invalid/expired API key")
                raise
            if _is_quota_error(e):
                _mark_cooling(entry)
            raise
        raw_text = _text_from_response(response)
        obj = _parse_json_object_text(raw_text)
        return obj, raw_text

    try:
        return _call()
    except Exception as e:
        raise RuntimeError(
            f"Gemini structured-json call failed "
            f"(profile={profile!r}, model={model!r}): {e}"
        ) from e


def check_model_available(model: str, *, profile: Profile = "debater") -> bool:
    """Best-effort: list models visible to this Vertex project."""
    # Use the first pool entry — listing models is a cheap probe and doesn't
    # need failover or round-robin.
    entry = _acquire(profile)
    try:
        for m in entry.client.models.list():
            name = m.name or ""
            short = name.split("/")[-1]
            if short == model or name.endswith(model):
                return True
        return False
    except Exception:
        return False
