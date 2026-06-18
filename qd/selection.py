"""Cell + parent selection for the MAP-Elites loop.

- Cell selection: probabilistic empty-cell preference (P=0.3); else softmax
  over occupied cells with weight ∝ exp((max_fit − fit) / T), preferring weak cells.
- Parent selection: 30% chance — sample uniformly from the 29-question pool
  (diversity injection); 70% chance — sample from occupied archive elites
  weighted by 1/count(qid) so over-represented questions are drawn less often.
  Falls back to the pool when the archive is empty, soft-preferring rows whose
  question_type matches the target cell's row label.
"""

from __future__ import annotations

import math
import random

from .archive import Archive, Cell


# Fraction of iterations that target an empty cell when occupied cells also exist.
# Prevents hard lock-in on the last empty cell while still seeding it steadily.
_P_EMPTY_PREF = 0.3

# Softmax temperature for weak-cell preference (lower = more concentrated on weakest).
_SOFTMAX_T = 0.1

# Probability of sampling a fresh question from the pool instead of the archive.
# Prevents any single question from dominating via stepping-stone inheritance.
_P_POOL_INJECT = 0.3


def select_target_cell(archive: Archive, rng: random.Random) -> Cell:
    empties = archive.empty_cells()
    occupied = archive.occupied_elites()

    # Always target an empty cell when the archive is completely empty.
    # When both empty and occupied cells exist, prefer empty with probability _P_EMPTY_PREF.
    if empties and (not occupied or rng.random() < _P_EMPTY_PREF):
        return rng.choice(empties)

    # Softmax over occupied cells, preferring WEAK cells (low fitness).
    # softmax(-fitness / T) ∝ exp((max_fit − fit) / T) — numerically stable form.
    fits = [e.fitness for _, e in occupied]
    max_fit = max(fits)
    weights = [math.exp((max_fit - f) / _SOFTMAX_T) for f in fits]
    cells = [c for c, _ in occupied]
    return rng.choices(cells, weights=weights, k=1)[0]


def _row_for_qid(pool: list[dict], qid: str) -> dict | None:
    for r in pool:
        if str(r["id"]) == str(qid):
            return r
    return None


def select_parent(
    archive: Archive,
    target_cell: Cell,
    pool: list[dict],
    rng: random.Random,
) -> tuple[dict, str | None]:
    """Returns (parent_row_dict, parent_qid).

    The returned row has the same schema as a pool row, but `question` and
    `negative_answer` may be inherited from an archive elite. The
    `correct_answer`, `story`, `story_title`, `id`, etc. always come from the
    pool row identified by `parent_qid`.

    Parent selection policy:
    - Archive empty → pool fallback, soft-preferring matching question_type.
    - _P_POOL_INJECT (30%) → sample uniformly from all 29 pool questions.
    - Otherwise (70%) → archive elites, weighted by 1/count(qid) so
      over-represented questions are drawn less often.
    """
    target_qtype, _ = target_cell
    occupied = archive.occupied_elites()

    # Archive empty: fall back to pool with type-preference (unchanged behaviour).
    if not occupied:
        typed = [r for r in pool if r.get("question_type") == target_qtype]
        chosen = rng.choice(typed if typed else pool)
        return dict(chosen), str(chosen["id"])

    # 30% pool injection: uniform over all 29 questions.
    if rng.random() < _P_POOL_INJECT:
        chosen = rng.choice(pool)
        return dict(chosen), str(chosen["id"])

    # 70% archive path: inverse-count weighted sample.
    qid_counts: dict[str, int] = {}
    for _, e in occupied:
        qid_counts[e.qid] = qid_counts.get(e.qid, 0) + 1

    weights = [1.0 / qid_counts[e.qid] for _, e in occupied]
    _, chosen_elite = rng.choices(occupied, weights=weights, k=1)[0]

    base_row = _row_for_qid(pool, chosen_elite.qid)
    if base_row is not None:
        inherited = dict(base_row)
        inherited["question"] = chosen_elite.mutated_question
        inherited["negative_answer"] = chosen_elite.mutated_distractor
        return inherited, chosen_elite.qid

    # qid not found in pool (should not happen) — safe fallback.
    chosen = rng.choice(pool)
    return dict(chosen), str(chosen["id"])
