"""Archive: 5x4 MAP-Elites grid, persistence, update + neighbour logic.

Thread-safety: `Archive` exposes a public `lock` attribute (a
`threading.Lock`). Methods do NOT acquire the lock internally — callers
in multi-threaded code must wrap compound operations (`try_update` +
`iter_count` update + `save`) in `with archive.lock:` so the
read-compare-write stays atomic across workers. Single-threaded callers
may ignore the lock entirely.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


QUESTION_TYPES: list[str] = [
    "Identity",
    "Process",
    "Cause",
    "Interpretive",
    "Negation",
]

MUTATION_TYPES: list[str] = [
    "plain_reword",
    "abstraction_lift",
    "prime_inversion",
    "distractor_boost",
]


Cell = tuple[str, str]  # (question_type, mutation_type)


@dataclass
class Elite:
    qid: str
    parent_qid: str
    mutated_question: str
    mutated_distractor: str
    correct_answer: str
    fitness: float
    iteration: int
    bleu_to_parent: float
    swap: bool
    judge_log_odds_truth: float | None
    judge_correct: bool | None
    # Pre-confirmation probe: result of running the eval judges on the
    # flash-lite SEARCH transcript that produced this elite. None = not
    # probed yet. False = judge picked the deceptive side. True = judge
    # picked the truthful side. A cell is "promotable" iff both probe
    # fields are False (both eval judges already fooled on the cheap
    # search debate).
    probe_flash_judge_correct: bool | None = None
    probe_lite_judge_correct:  bool | None = None
    probe_iter: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Elite":
        """Construct an Elite from a JSON dict, tolerating schema drift.

        Drops keys unknown to the current dataclass and supplies defaults
        for missing optional fields, so an `archive.json` written by an
        earlier version of the code keeps loading after `Elite` evolves.
        Required identity fields (qid, mutated_question, ...) still raise
        if absent. Corruption there should fail loudly rather than silently."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        filtered.setdefault("parent_qid", "")
        filtered.setdefault("bleu_to_parent", 0.0)
        filtered.setdefault("swap", False)
        filtered.setdefault("judge_log_odds_truth", None)
        filtered.setdefault("judge_correct", None)
        filtered.setdefault("probe_flash_judge_correct", None)
        filtered.setdefault("probe_lite_judge_correct", None)
        filtered.setdefault("probe_iter", None)
        return cls(**filtered)


@dataclass
class Archive:
    cells: dict[Cell, Elite | None] = field(default_factory=dict)
    iter_count: int = 0

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = {(q, m): None for q in QUESTION_TYPES for m in MUTATION_TYPES}
        # Instance attribute (not a dataclass field), so it doesn't appear in
        # asdict() / to_jsonable() and the on-disk schema is unaffected.
        self.lock = threading.Lock()

    # ── Cell lookups ─────────────────────────────────────────────────────
    def get(self, cell: Cell) -> Elite | None:
        return self.cells.get(cell)

    def occupied_cells(self) -> list[Cell]:
        return [c for c, e in self.cells.items() if e is not None]

    def empty_cells(self) -> list[Cell]:
        return [c for c, e in self.cells.items() if e is None]

    def occupied_elites(self) -> list[tuple[Cell, Elite]]:
        return [(c, e) for c, e in self.cells.items() if e is not None]

    # ── Updates ──────────────────────────────────────────────────────────
    def try_update(self, cell: Cell, candidate: Elite) -> bool:
        """Replace the cell elite if empty or if the candidate is strictly fitter.
        Returns True iff the cell was updated.

        Not internally locked — multi-threaded callers must hold `self.lock`.
        """
        existing = self.cells.get(cell)
        if existing is None or candidate.fitness > existing.fitness:
            self.cells[cell] = candidate
            return True
        return False

    # ── Neighbours (Manhattan-1, 4-connected) ────────────────────────────
    @staticmethod
    def neighbors(cell: Cell) -> list[Cell]:
        q, m = cell
        try:
            qi = QUESTION_TYPES.index(q)
            mi = MUTATION_TYPES.index(m)
        except ValueError:
            return []
        out: list[Cell] = []
        for dqi, dmi in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = qi + dqi, mi + dmi
            if 0 <= ni < len(QUESTION_TYPES) and 0 <= nj < len(MUTATION_TYPES):
                out.append((QUESTION_TYPES[ni], MUTATION_TYPES[nj]))
        return out

    # ── Coverage report ──────────────────────────────────────────────────
    def coverage(self) -> dict:
        elites = [e for _, e in self.occupied_elites()]
        if not elites:
            return {
                "occupied": 0,
                "total": len(self.cells),
                "min_fitness": None,
                "mean_fitness": None,
                "max_fitness": None,
            }
        fits = [e.fitness for e in elites]
        return {
            "occupied": len(elites),
            "total": len(self.cells),
            "min_fitness": min(fits),
            "mean_fitness": sum(fits) / len(fits),
            "max_fitness": max(fits),
        }

    # ── Persistence ──────────────────────────────────────────────────────
    def to_jsonable(self) -> dict:
        return {
            "iter_count": self.iter_count,
            "question_types": QUESTION_TYPES,
            "mutation_types": MUTATION_TYPES,
            "cells": [
                {
                    "cell": [q, m],
                    "elite": asdict(e) if e is not None else None,
                }
                for (q, m), e in self.cells.items()
            ],
        }

    def save(self, path: str | os.PathLike) -> None:
        """Atomic write to `path` (tmp + rename).

        Not internally locked — multi-threaded callers must hold `self.lock`
        when paired with `try_update` / `iter_count` mutation so the file
        snapshot stays consistent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8"
        ) as f:
            json.dump(self.to_jsonable(), f, indent=2)
            tmp_name = f.name
        os.replace(tmp_name, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Archive":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        archive = cls()
        archive.iter_count = int(data.get("iter_count", 0))
        for entry in data.get("cells", []):
            cell = tuple(entry["cell"])  # type: ignore[assignment]
            elite_d = entry.get("elite")
            if elite_d is None:
                archive.cells[cell] = None
            else:
                archive.cells[cell] = Elite.from_dict(elite_d)
        return archive
