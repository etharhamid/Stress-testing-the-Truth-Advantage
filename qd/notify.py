"""Slack webhook notifier.

Reads SLACK_WEBHOOK_URL from the environment. If unset, attempts a one-shot
dotenv load from the project root so the helper is usable from scripts that
don't import `config`. Still a no-op if the var stays unset.

All POST exceptions are swallowed by `slack()` / `slack_throttled()` so a
running search or eval never crashes because Slack is unreachable. Use
`notify_diag()` to surface the underlying error during setup.

Usage:
    from qd.notify import slack, slack_throttled, notify_diag

    slack("flat_grid run complete: covered 62/62")
    slack_throttled("vertex_quota", 300.0, "Vertex quota hit on seed=42")
    notify_diag()  # prints what's loaded + actual POST status
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path


_LOCK = threading.Lock()
_LAST_SENT: dict[str, float] = {}
_DOTENV_TRIED = False


def _maybe_load_dotenv() -> None:
    """One-shot best-effort .env load. Idempotent; never raises."""
    global _DOTENV_TRIED
    if _DOTENV_TRIED:
        return
    _DOTENV_TRIED = True
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except Exception:
        return
    # Walk up from this file looking for a .env (project root holds it).
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            try:
                load_dotenv(candidate, override=False)
            except Exception:
                pass
            return


def _webhook_url() -> str:
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        _maybe_load_dotenv()
        url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    return url


def _post(text: str, *, raise_on_error: bool = False) -> tuple[bool, str]:
    """Return (sent, detail). detail is empty on success, else an error string."""
    url = _webhook_url()
    if not url:
        return False, "SLACK_WEBHOOK_URL not set"
    payload = json.dumps({"text": text}).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200 and body.strip() == "ok":
                return True, ""
            return False, f"HTTP {resp.status}: {body!r}"
    except Exception as e:
        if raise_on_error:
            raise
        return False, repr(e)


def slack(text: str) -> None:
    """Fire-and-forget Slack notification. Never raises."""
    _post(text)


def slack_throttled(key: str, cooldown_sec: float, text: str) -> bool:
    """Send only if cooldown for `key` has elapsed since the last send.

    Returns True iff the message was actually sent. Thread-safe.
    """
    now = time.time()
    with _LOCK:
        last = _LAST_SENT.get(key, 0.0)
        if now - last < cooldown_sec:
            return False
        _LAST_SENT[key] = now
    _post(text)
    return True


def notify_diag(text: str = "qd_debate notifier diagnostic") -> None:
    """Print a verbose diagnostic and try to send a single test message."""
    url = _webhook_url()
    if not url:
        print("[notify] SLACK_WEBHOOK_URL: <UNSET>", file=sys.stderr)
        print(
            "[notify] tried loading .env from project root; "
            "either python-dotenv is missing or .env lacks SLACK_WEBHOOK_URL",
            file=sys.stderr,
        )
        return
    masked = url[:32] + "..." + url[-6:] if len(url) > 44 else url
    print(f"[notify] SLACK_WEBHOOK_URL = {masked}", file=sys.stderr)
    ok, detail = _post(text)
    if ok:
        print("[notify] POST ok — message delivered to Slack", file=sys.stderr)
    else:
        print(f"[notify] POST FAILED: {detail}", file=sys.stderr)
