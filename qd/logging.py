"""Append-only JSONL writer that flushes after every record.

Thread-safe: a per-instance `threading.Lock` guards each `write_record` call
so concurrent workers in the parallel MAP-Elites loop never interleave
partial JSON lines. Per-record `flush()` is preserved — no buffering.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write_record(self, obj: dict) -> None:
        line = json.dumps(obj, default=_json_default) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _json_default(o: Any) -> Any:
    # Tuples (e.g. cell keys) and other non-JSON-native objects fall here.
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def make_failure_record(
    *,
    iter_no: int,
    qid: str | None,
    target_cell: tuple[str, str] | None,
    stage: str,
    detail: str = "",
    mutated_question: str | None = None,
    mutated_distractor: str | None = None,
    round: int | None = None,
) -> dict:
    rec: dict = {
        "iter": iter_no,
        "qid": qid,
        "target_cell": list(target_cell) if target_cell is not None else None,
        "stage": stage,
        "detail": detail,
    }
    if round is not None:
        rec["round"] = round
    if mutated_question is not None:
        rec["mutated_question"] = mutated_question
    if mutated_distractor is not None:
        rec["mutated_distractor"] = mutated_distractor
    return rec
