"""Grid state report: dump a human-readable markdown summary of the current archive.

Run standalone:

    python -m qd.grid_state
    python -m qd.grid_state --archive qd_results/archive.json --out qd_results/grid_state.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qd.archive import Archive, MUTATION_TYPES, QUESTION_TYPES
from qd.bleu import self_bleu_archive


def _trunc(text: str, max_len: int = 100) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _judge_call_breakdown(judgements_path: str | Path) -> list[str]:
    """Read search_judgements.jsonl and return a markdown breakdown table."""
    path = Path(judgements_path)
    if not path.exists():
        return []

    vertex_ok = 0
    estimator = 0
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("judge_confidence_estimated"):
            estimator += 1
        elif rec.get("judge_fallback"):
            skipped += 1
        else:
            vertex_ok += 1

    total = vertex_ok + estimator + skipped
    if total == 0:
        return []

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%"

    lines = [
        f"## Judge call breakdown — {total} total",
        "",
        "| Path | Count | % |",
        "|------|------:|--:|",
        f"| Vertex AI (logprobs) | {vertex_ok} | {pct(vertex_ok)} |",
        f"| LLM confidence estimator | {estimator} | {pct(estimator)} |",
        f"| Skipped (estimator also failed) | {skipped} | {pct(skipped)} |",
        "",
    ]
    return lines


def dump_grid_state(
    archive_path: str | Path,
    output_path: str | Path | None = None,
    judgements_path: str | Path | None = None,
) -> str:
    """Build a markdown grid-state report and optionally write it to output_path.

    When judgements_path is provided, includes a judge-call breakdown section
    showing how many calls used Vertex logprobs vs the LLM confidence estimator.

    Returns the full report as a string so callers can print or log it."""
    archive = Archive.load(archive_path)
    occupied = len(archive.occupied_cells())
    total = len(archive.cells)

    lines: list[str] = []
    lines.append(f"# Archive grid state — iter {archive.iter_count}")
    lines.append(f"Occupied: **{occupied}/{total}** cells\n")

    if judgements_path is not None:
        lines.extend(_judge_call_breakdown(judgements_path))

    # ── Per-row detail tables ─────────────────────────────────────────────────
    for qt in QUESTION_TYPES:
        lines.append(f"## {qt}")
        lines.append("")
        lines.append(
            "| Mutation | qid | log_odds_truth | fitness | iter | correct | Mutated question |"
        )
        lines.append("|---|---|---:|---:|---:|:---:|---|")
        for mt in MUTATION_TYPES:
            elite = archive.get((qt, mt))
            if elite is None:
                lines.append(f"| `{mt}` | — | — | — | — | — | *empty* |")
            else:
                lo = (
                    f"{elite.judge_log_odds_truth:+.3f}"
                    if elite.judge_log_odds_truth is not None
                    else "fallback"
                )
                correct_sym = (
                    "✓" if elite.judge_correct is True
                    else "✗" if elite.judge_correct is False
                    else "?"
                )
                q_trunc = _trunc(elite.mutated_question)
                lines.append(
                    f"| `{mt}` | {elite.qid} | {lo} | {elite.fitness:+.3f}"
                    f" | {elite.iteration} | {correct_sym} | {q_trunc} |"
                )
        lines.append("")

    # ── Compact summary grid ──────────────────────────────────────────────────
    lines.append("## Compact summary (fitness | log_odds_truth)")
    lines.append("")
    header = "| | " + " | ".join(f"`{mt}`" for mt in MUTATION_TYPES) + " |"
    sep = "|---|" + "---|" * len(MUTATION_TYPES)
    lines.append(header)
    lines.append(sep)
    for qt in QUESTION_TYPES:
        row_parts = [f"**{qt}**"]
        for mt in MUTATION_TYPES:
            elite = archive.get((qt, mt))
            if elite is None:
                row_parts.append("—")
            else:
                lo = (
                    f"{elite.judge_log_odds_truth:+.2f}"
                    if elite.judge_log_odds_truth is not None
                    else "fb"
                )
                row_parts.append(f"{elite.fitness:+.2f} / {lo}")
        lines.append("| " + " | ".join(row_parts) + " |")
    lines.append("")

    # ── Self-BLEU diversity metric ────────────────────────────────────────────
    elite_questions = [e.mutated_question for e in archive.cells.values() if e is not None]
    sb = self_bleu_archive(elite_questions)
    lines.append(f"---\n**Self-BLEU** (avg pairwise, lower = more diverse): "
                 f"`{sb:.3f}` over {len(elite_questions)} elites\n")

    report = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump archive grid state to a markdown file."
    )
    parser.add_argument("--archive", default="qd_results/archive.json")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <archive_dir>/grid_state.md).",
    )
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(args.archive).parent / "grid_state.md"
    )

    report = dump_grid_state(args.archive, out_path)
    print(report)
    print(f"\n[grid_state] written to {out_path}")


if __name__ == "__main__":
    main()
