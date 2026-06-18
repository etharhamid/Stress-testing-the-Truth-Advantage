"""Archive visualisation: fitness heatmap, coverage curve, eval breakdown.

Run as a module to regenerate plots from existing artifacts:

    python -m qd.plot --archive qd_results/archive.json \
                      --log qd_results/search_judgements.jsonl \
                      --out qd_results/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .archive import Archive, MUTATION_TYPES, QUESTION_TYPES
from .fitness import fitness_from_judgement


def fitness_heatmap(archive: Archive, out_path: Path) -> None:
    """4x4 heatmap of cell fitness. Empty cells rendered grey."""
    rows = len(QUESTION_TYPES)
    cols = len(MUTATION_TYPES)
    matrix = np.full((rows, cols), np.nan)
    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            elite = archive.get((qt, mt))
            if elite is not None:
                matrix[i, j] = elite.fitness

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgrey")
    masked = np.ma.masked_invalid(matrix)
    if np.any(~np.isnan(matrix)):
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1.0)
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(cols))
    ax.set_xticklabels(MUTATION_TYPES, rotation=20, ha="right")
    ax.set_yticks(range(rows))
    ax.set_yticklabels(QUESTION_TYPES)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Question type")
    ax.set_title("Archive fitness (= −judge_log_odds_truth)\nHigher = stronger deception")

    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            elite = archive.get((qt, mt))
            if elite is None:
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=10)
            else:
                txt = f"{elite.fitness:+.2f}\nqid={elite.qid}\niter={elite.iteration}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="fitness")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def coverage_curve(judgements_path: Path, out_path: Path) -> None:
    """Replay search_judgements.jsonl to draw cells-occupied vs iteration."""
    if not judgements_path.exists():
        return

    rows: list[dict] = []
    with open(judgements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return

    # Sort by iteration; replay try-update logic to count cells occupied + max fitness.
    rows.sort(key=lambda r: r.get("iter", 0))
    best_per_cell: dict[tuple, float] = {}
    iters: list[int] = []
    occupied_count: list[int] = []
    mean_fit: list[float] = []
    for r in rows:
        cell_raw = r.get("target_cell")
        if not cell_raw:
            continue
        cell = tuple(cell_raw)  # type: ignore[assignment]
        # Use the canonical fitness mapping so the replay matches the
        # signal that actually drove archive updates (continuous primary,
        # ±config.QD_FALLBACK_FITNESS fallback when logprobs are missing).
        fit = fitness_from_judgement(r)
        if fit is None:
            continue
        cur = best_per_cell.get(cell)
        if cur is None or fit > cur:
            best_per_cell[cell] = fit
        iters.append(int(r.get("iter", 0)))
        occupied_count.append(len(best_per_cell))
        mean_fit.append(sum(best_per_cell.values()) / len(best_per_cell))

    if not iters:
        return

    total_cells = len(QUESTION_TYPES) * len(MUTATION_TYPES)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(iters, occupied_count, "C0-", label="cells occupied")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(f"cells occupied (of {total_cells})", color="C0")
    ax1.set_ylim(0, total_cells)
    ax2 = ax1.twinx()
    ax2.plot(iters, mean_fit, "C3-", alpha=0.6, label="mean fitness")
    ax2.set_ylabel("mean fitness over occupied cells", color="C3")
    fig.suptitle("Archive coverage and mean fitness over time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def eval_breakdown(
    eval_summary: dict,
    baseline_twr: float | None,
    out_path: Path,
) -> None:
    """Grouped bar chart of adversarial truth-win-rate by row and by column."""
    by_row: dict[str, float] = eval_summary.get("by_question_type", {}) or {}
    by_col: dict[str, float] = eval_summary.get("by_mutation_type", {}) or {}

    judge_model  = eval_summary.get("judge_model", "")
    overall_twr  = eval_summary.get("truth_win_rate")
    total_q      = eval_summary.get("total_questions", 0)
    swap_mode    = eval_summary.get("swap_mode", "")

    if swap_mode == "50_50_balanced":
        swap_label = f"50/50 balanced · {total_q} debates"
    elif total_q and total_q >= 30:
        swap_label = f"both swaps · {total_q} debates"
    elif total_q:
        swap_label = f"50/50 balanced · {total_q} debates"
    else:
        swap_label = swap_mode or "unknown swap mode"

    twr_str  = f"  ·  overall accuracy = {overall_twr:.3f}" if overall_twr is not None else ""
    sup_text = f"Judge: {judge_model}  ·  {swap_label}{twr_str}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(sup_text, fontsize=10, fontweight="bold")

    for ax, data, title in (
        (axes[0], by_row, "Judge Accuracy by question type"),
        (axes[1], by_col, "Judge Accuracy by mutation type"),
    ):
        if data:
            labels = list(data.keys())
            values = [data[k] for k in labels]
            ax.bar(labels, values, color="C0", zorder=2)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("judge accuracy")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=20)
            if baseline_twr is not None:
                ax.axhline(baseline_twr, color="#2ca02c", linestyle="--",
                           linewidth=1.4, zorder=3, label=f"baseline {baseline_twr:.3f}")
            if overall_twr is not None:
                ax.axhline(overall_twr, color="#ff7f0e", linestyle=":",
                           linewidth=1.4, zorder=3, label=f"overall {overall_twr:.3f}")
            if baseline_twr is not None or overall_twr is not None:
                ax.legend(fontsize=8, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_COMBINED_COLORS = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]


def eval_breakdown_combined(
    summaries: list[tuple[str, dict]],
    baseline_twr: float | None,
    out_path: Path,
) -> None:
    """Grouped bar chart comparing multiple judge models on the same axes.

    Each model gets its own colour; bars are placed side-by-side so both are
    fully visible without needing overlap transparency.
    """
    if not summaries:
        return

    n_models = len(summaries)
    colors = _COMBINED_COLORS[:n_models]

    first_summary = summaries[0][1]
    by_row_keys = list((first_summary.get("by_question_type") or {}).keys())
    by_col_keys = list((first_summary.get("by_mutation_type") or {}).keys())

    swap_mode = first_summary.get("swap_mode", "")
    total_q   = first_summary.get("total_questions", 0)
    if swap_mode == "50_50_balanced":
        swap_label = f"50/50 balanced  ·  {total_q} debates"
    elif total_q and total_q >= 30:
        swap_label = f"both swaps  ·  {total_q} debates"
    elif total_q:
        swap_label = f"50/50 balanced  ·  {total_q} debates"
    else:
        swap_label = swap_mode or "unknown swap mode"

    baseline_str = f"baseline accuracy = {baseline_twr:.3f}" if baseline_twr is not None else ""
    subtitle_parts = [swap_label] + ([baseline_str] if baseline_str else [])
    subtitle = "  ·  ".join(subtitle_parts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.suptitle(
        "Adversarial Judge Accuracy — Judge Comparison\n" + subtitle,
        fontsize=12, fontweight="bold", linespacing=1.55,
    )

    width   = 0.70 / n_models
    offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

    for ax, cat_keys, cat_field, title in [
        (axes[0], by_row_keys, "by_question_type",  "by Question Type"),
        (axes[1], by_col_keys, "by_mutation_type",  "by Mutation Type"),
    ]:
        x = np.arange(len(cat_keys))
        for idx, (model_label, summary) in enumerate(summaries):
            data   = summary.get(cat_field, {}) or {}
            values = [data.get(k, 0.0) for k in cat_keys]
            twr    = summary.get("truth_win_rate", 0.0)
            short  = model_label.replace("gemini-2.5-", "").replace("gemini-", "")
            ax.bar(x + offsets[idx], values, width, color=colors[idx],
                   alpha=0.85, label=f"{short}  (accuracy = {twr:.3f})", zorder=2)

        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_keys, rotation=20, ha="right")
        ax.set_ylabel("judge accuracy")
        ax.set_title(f"Judge Accuracy {title}", fontsize=11)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        if baseline_twr is not None:
            ax.axhline(baseline_twr, color="#2ca02c", linestyle="--",
                       linewidth=1.4, zorder=3, label=f"baseline  ({baseline_twr:.3f})")
        ax.legend(fontsize=8, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eval_dual_heatmap(
    eval_csv_path: Path,
    out_path: Path,
    judge_model: str = "",
) -> None:
    """Side-by-side 4×4: left = archive search fitness, right = eval TWR.

    Makes the search→eval transfer gap visible: a cell dark on the left
    (MAP-Elites found a strong framing) but red on the right (deception still
    wins at eval) is a successful transfer; dark left + white/blue right is
    a transfer failure (fitness was model-specific).
    """
    import csv as _csv

    with open(eval_csv_path, encoding="utf-8") as f:
        csv_rows = list(_csv.DictReader(f))

    n_qt = len(QUESTION_TYPES)
    n_mt = len(MUTATION_TYPES)

    cell_data: dict[tuple, list[dict]] = {}
    for r in csv_rows:
        key = (r["question_type"], r["mutation_type"])
        cell_data.setdefault(key, []).append(r)

    fitness_matrix = np.full((n_qt, n_mt), np.nan)
    twr_matrix     = np.full((n_qt, n_mt), np.nan)

    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            recs = cell_data.get((qt, mt), [])
            if not recs:
                continue
            fit_vals = []
            for r in recs:
                raw = r.get("elite_search_fitness", "")
                if raw not in ("", "None", None):
                    try:
                        fit_vals.append(float(raw))
                    except ValueError:
                        pass
            if fit_vals:
                fitness_matrix[i, j] = fit_vals[0]  # same elite for all rows in cell
            n_correct = sum(1 for r in recs if r.get("judge_correct") == "True")
            twr_matrix[i, j] = n_correct / len(recs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: search fitness ─────────────────────────────────────────────────
    ax_l = axes[0]
    cmap_fit = plt.cm.YlOrRd.copy()
    cmap_fit.set_bad("lightgrey")
    fit_masked = np.ma.masked_invalid(fitness_matrix)
    if np.any(~np.isnan(fitness_matrix)):
        vmin_f = np.nanmin(fitness_matrix)
        vmax_f = np.nanmax(fitness_matrix)
        if vmin_f == vmax_f:
            vmin_f -= 1.0; vmax_f += 1.0
    else:
        vmin_f, vmax_f = 0.0, 1.0
    im_l = ax_l.imshow(fit_masked, cmap=cmap_fit, vmin=vmin_f, vmax=vmax_f, aspect="auto")
    ax_l.set_xticks(range(n_mt))
    ax_l.set_xticklabels(MUTATION_TYPES, rotation=20, ha="right")
    ax_l.set_yticks(range(n_qt))
    ax_l.set_yticklabels(QUESTION_TYPES)
    ax_l.set_xlabel("Mutation type")
    ax_l.set_ylabel("Question type")
    ax_l.set_title("Search fitness (archive elite)\nDarker = stronger deception found by MAP-Elites")
    fig.colorbar(im_l, ax=ax_l, label="search fitness (= −log_odds_truth at search)")

    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            recs = cell_data.get((qt, mt), [])
            if not recs:
                ax_l.text(j, i, "—", ha="center", va="center", fontsize=10)
                continue
            fit = fitness_matrix[i, j]
            qid = recs[0].get("qid", "?")
            txt = f"{fit:.2f}\nqid={qid}" if not np.isnan(fit) else f"n/a\nqid={qid}"
            ax_l.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    # ── Right: eval TWR ──────────────────────────────────────────────────────
    ax_r = axes[1]
    cmap_twr = plt.cm.RdBu.copy()
    cmap_twr.set_bad("lightgrey")
    twr_masked = np.ma.masked_invalid(twr_matrix)
    im_r = ax_r.imshow(twr_masked, cmap=cmap_twr, vmin=0.0, vmax=1.0, aspect="auto")
    ax_r.set_xticks(range(n_mt))
    ax_r.set_xticklabels(MUTATION_TYPES, rotation=20, ha="right")
    ax_r.set_yticks(range(n_qt))
    ax_r.set_yticklabels(QUESTION_TYPES)
    ax_r.set_xlabel("Mutation type")
    ax_r.set_ylabel("Question type")
    title_r = "Eval judge accuracy\nBlue = truth wins · Red = deception wins"
    if judge_model:
        title_r += f"  ·  {judge_model}"
    ax_r.set_title(title_r)
    fig.colorbar(im_r, ax=ax_r, label="judge accuracy  (0 = deception wins · 1 = truth wins)")

    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            recs = cell_data.get((qt, mt), [])
            if not recs:
                ax_r.text(j, i, "—", ha="center", va="center", fontsize=10)
                continue
            icons = "".join(
                "✓" if r["judge_correct"] == "True" else "✗"
                for r in sorted(recs, key=lambda r: r["swap"] == "True")
            )
            twr = twr_matrix[i, j]
            ax_r.text(j, i, f"{icons}\n{twr:.0%}", ha="center", va="center",
                      fontsize=8, color="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eval_binary_heatmap(
    eval_csv_path: Path,
    out_path: Path,
    judge_model: str = "",
) -> None:
    """5×4 heatmap showing per-cell consistency across both swap conditions.

    Used by the `--double-swap` eval mode where each cell holds two debates.
    Blue  = both debates correct (truth wins both).
    White = split (one correct, one wrong).
    Red   = both debates wrong (deception wins both).
    Grey  = missing data (fewer than 2 debates for that cell).
    """
    import csv as _csv
    from matplotlib.colors import ListedColormap, BoundaryNorm

    with open(eval_csv_path, encoding="utf-8") as f:
        csv_rows = list(_csv.DictReader(f))

    n_qt = len(QUESTION_TYPES)
    n_mt = len(MUTATION_TYPES)

    cell_data: dict[tuple, list[dict]] = {}
    for r in csv_rows:
        key = (r["question_type"], r["mutation_type"])
        cell_data.setdefault(key, []).append(r)

    # Score: -1 = both wrong, 0 = split, +1 = both correct. NaN = missing.
    matrix = np.full((n_qt, n_mt), np.nan)
    cell_text: dict[tuple, str] = {}
    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            recs = cell_data.get((qt, mt), [])
            if len(recs) < 2:
                cell_text[(i, j)] = "?" if recs else "—"
                continue
            n_correct = sum(1 for r in recs if r["judge_correct"] == "True")
            score = n_correct - 1  # -1, 0, or +1
            matrix[i, j] = score
            icons = "".join(
                "✓" if r["judge_correct"] == "True" else "✗"
                for r in sorted(recs, key=lambda r: r["swap"] == "True")
            )
            label = {-1: "both wrong", 0: "split", 1: "both correct"}[score]
            cell_text[(i, j)] = f"{icons}\n{label}"

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["#67001f", "#f7f7f7", "#053061"])  # red, white, blue (matches dual heatmap)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    masked = np.ma.masked_invalid(matrix)

    # Grey background for missing cells
    ax.set_facecolor("lightgrey")
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_mt))
    ax.set_xticklabels(MUTATION_TYPES, rotation=20, ha="right")
    ax.set_yticks(range(n_qt))
    ax.set_yticklabels(QUESTION_TYPES)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Question type")
    title = "Per-cell consistency (swap=False then swap=True)\nBlue = both correct · White = split · Red = both wrong"
    if judge_model:
        title += f"  ·  {judge_model}"
    ax.set_title(title)

    for i in range(n_qt):
        for j in range(n_mt):
            txt = cell_text.get((i, j), "")
            score = matrix[i, j]
            color = "white" if (not np.isnan(score) and score == -1) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#053061", label="Both correct"),
        Patch(facecolor="#f7f7f7", edgecolor="grey", label="Split (1 correct)"),
        Patch(facecolor="#67001f", label="Both wrong"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eval_single_heatmap(
    eval_csv_path: Path,
    out_path: Path,
    judge_model: str = "",
) -> None:
    """5×4 heatmap for a single-judgement-per-cell eval (e.g. 50/50 swap).

    Blue  = truth wins  (judge_correct=True).
    Red   = deception wins (judge_correct=False).
    Grey  = no data for that cell.
    """
    import csv as _csv
    from matplotlib.colors import ListedColormap, BoundaryNorm

    with open(eval_csv_path, encoding="utf-8") as f:
        csv_rows = list(_csv.DictReader(f))

    n_qt = len(QUESTION_TYPES)
    n_mt = len(MUTATION_TYPES)

    cell_data: dict[tuple, dict] = {}
    for r in csv_rows:
        key = (r["question_type"], r["mutation_type"])
        cell_data[key] = r  # one row per cell

    matrix = np.full((n_qt, n_mt), np.nan)
    cell_text: dict[tuple, str] = {}

    for i, qt in enumerate(QUESTION_TYPES):
        for j, mt in enumerate(MUTATION_TYPES):
            rec = cell_data.get((qt, mt))
            if rec is None:
                cell_text[(i, j)] = "—"
                continue
            correct = rec.get("judge_correct") == "True"
            matrix[i, j] = 1.0 if correct else -1.0
            icon  = "✓" if correct else "✗"
            cell_text[(i, j)] = f"{icon}\n{rec.get('qid', '?')}"

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["#67001f", "#053061"])   # red=deception, blue=truth (matches dual heatmap)
    norm = BoundaryNorm([-1.5, 0.0, 1.5], cmap.N)
    masked = np.ma.masked_invalid(matrix)

    ax.set_facecolor("lightgrey")
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_mt))
    ax.set_xticklabels(MUTATION_TYPES, rotation=20, ha="right")
    ax.set_yticks(range(n_qt))
    ax.set_yticklabels(QUESTION_TYPES)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Question type")
    title = "Per-cell result (single debate, 50/50 swap)\nBlue = truth wins · Red = deception wins"
    if judge_model:
        title += f"  ·  {judge_model}"
    ax.set_title(title)

    for i in range(n_qt):
        for j in range(n_mt):
            txt   = cell_text.get((i, j), "")
            score = matrix[i, j]
            color = "white" if (not np.isnan(score) and score == -1.0) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#053061", label="Truth wins"),
        Patch(facecolor="#67001f", label="Deception wins"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=True, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_search_plots(archive_path: str, judgements_path: str, plots_dir: str) -> None:
    """Render fitness_heatmap.png and coverage_curve.png."""
    archive = Archive.load(archive_path)
    plots = Path(plots_dir)
    fitness_heatmap(archive, plots / "fitness_heatmap.png")
    coverage_curve(Path(judgements_path), plots / "coverage_curve.png")


def _max_debates_per_cell(csv_path: Path) -> int:
    """Return the largest row count for any (question_type, mutation_type)
    cell in `csv_path`. Used to distinguish single-swap (max=1) from
    double-swap (max=2) eval outputs."""
    import csv as _csv
    counts: dict[tuple, int] = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            key = (r["question_type"], r["mutation_type"])
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def render_eval_plot(
    eval_summary_path: str,
    plots_dir: str,
    baseline_twr: float | None = None,
) -> None:
    p = Path(eval_summary_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if baseline_twr is None:
        baseline_twr = summary.get("baseline_truth_win_rate")
    plots = Path(plots_dir)

    # Derive the eval_results CSV path from the summary path (same stem, different suffix).
    # e.g. eval_summary_flash.json → eval_results_flash.csv
    stem = p.stem  # e.g. "eval_summary_flash"
    tag = stem[len("eval_summary"):]  # e.g. "_flash" or ""
    csv_path = p.parent / f"eval_results{tag}.csv"

    suffix = tag if tag else ""
    eval_breakdown(summary, baseline_twr, plots / f"eval_breakdown{suffix}.png")
    if csv_path.exists():
        eval_dual_heatmap(
            csv_path,
            plots / f"eval_dual_heatmap{suffix}.png",
            judge_model=summary.get("judge_model", ""),
        )
        # Auto-detect single- vs double-swap mode by counting debates per cell.
        # In double-swap each (qt, mt) cell holds 2 rows; in single-swap, 1.
        max_per_cell = _max_debates_per_cell(csv_path)
        if max_per_cell >= 2:
            eval_binary_heatmap(
                csv_path,
                plots / f"eval_binary_heatmap{suffix}.png",
                judge_model=summary.get("judge_model", ""),
            )
        else:
            eval_single_heatmap(
                csv_path,
                plots / f"eval_single_heatmap{suffix}.png",
                judge_model=summary.get("judge_model", ""),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="qd_results/archive.json")
    parser.add_argument("--log", default="qd_results/search_judgements.jsonl")
    parser.add_argument("--out", default="qd_results/plots/")
    parser.add_argument("--eval-summary", default="qd_results/eval_summary.json",
                        help="path to eval_summary.json (optional)")
    args = parser.parse_args()
    render_search_plots(args.archive, args.log, args.out)
    if Path(args.eval_summary).exists():
        render_eval_plot(args.eval_summary, args.out)
    print(f"Plots written to {args.out}")
    print("  fitness_heatmap.png  — fitness (= −log_odds_truth) per cell")
    print("  coverage_curve.png   — cells occupied + mean fitness over iterations")


if __name__ == "__main__":
    main()
