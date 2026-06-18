"""Sentence-level BLEU filter (Rainbow-Teaming style).

We use sacrebleu's `sentence_bleu` which returns a percentage score [0, 100].
We normalise to [0, 1] for direct comparison with the configured threshold.
"""

from __future__ import annotations

import sacrebleu


def bleu_score(parent: str, mutated: str) -> float:
    """Sentence-level BLEU between mutated and parent, normalised to [0, 1]."""
    if not parent.strip() or not mutated.strip():
        return 0.0
    return sacrebleu.sentence_bleu(mutated, [parent]).score / 100.0


def passes_filter(parent: str, mutated: str, threshold: float) -> tuple[float, bool]:
    """Return (score, accept). Accept means score <= threshold."""
    score = bleu_score(parent, mutated)
    return score, score <= threshold


def self_bleu_archive(questions: list[str]) -> float:
    """Average pairwise sentence-BLEU across all questions (lower = more diverse)."""
    if len(questions) < 2:
        return 0.0
    total, count = 0.0, 0
    for i, qi in enumerate(questions):
        for j, qj in enumerate(questions):
            if i != j:
                total += bleu_score(qj, qi)
                count += 1
    return total / count
