#!/usr/bin/env python
"""Pro-config evaluation of the flat-grid best framings.

Mirrors `run_qd_eval.py` but reads from
`qd_results/seed_{seed}/flat_grid_run_{NNN}/flat_best_per_qid.csv` (mode=best)
or `flat_per_mutation.csv` (mode=per_mutation) instead of an archive.

Swap modes:
  * default (single-swap): each row gets the same swap flag its QID was
    assigned in the baseline (steps/run_debates.py). Keeps the positional
    condition constant so per-QID deltas are not confounded by swap changes.
    Writes to `eval_flat_best/` (or `eval_flat_per_mutation/`).
  * --double-swap: each row debated twice (swap=False + swap=True). Writes
    to `eval_flat_best_double/` (or `eval_flat_per_mutation_double/`).

Cross-seed:
  * --all-seeds spawns one subprocess per known seed in
    `qd/config.SEED_TO_BASELINE`, then aggregates per-seed eval summaries into
    `qd_results/flat_aggregate_summary.json` (mean/SD across seeds, pooled
    Wilson CI, mean delta vs baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from qd.config import (
    QDConfig,
    SEED_TO_BASELINE,
    baseline_swap_for_seed,
    flat_results_root,
    flat_seed_dir,
    latest_flat_run_dir,
    test_config,
)
from qd.fitness import run_eval_debate, run_eval_judge
from qd.logging import JsonlWriter
from qd.notify import slack, slack_throttled

from steps.score_results import _wilson_ci  # type: ignore[import-not-found]


# ── Small shared helpers ───────────────────────────────────────────────────────

def _load_pool(csv_path: str) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pool[str(row["id"])] = row
    return pool


def _load_baseline_twr(baseline_dir: str) -> float | None:
    """Mean truth_win_rate across re-judged baseline eval_summaries."""
    p = Path(baseline_dir)
    if not p.exists():
        return None
    files = sorted(p.glob("base_run_*/eval_summary_*.json"))
    if not files:
        files = sorted(p.glob("*summary*.json"))
    twrs: list[float] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        twr = obj.get("truth_win_rate")
        if isinstance(twr, (int, float)):
            twrs.append(float(twr))
    return sum(twrs) / len(twrs) if twrs else None


def _round4(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def _model_tag(model: str) -> str:
    for prefix in ("gemini-2.5-", "gemini-3.1-", "gemini-3.0-", "gemma-4-", "gemma-3-"):
        if model.startswith(prefix):
            return model[len(prefix):].replace("/", "_")
    return model.replace("/", "_")


# ── Elites CSV writer ─────────────────────────────────────────────────────────

def _write_elites_csv(
    items: list[tuple[str, dict, bool]],
    pool: dict[str, dict],
    eval_dir: Path,
    mode: str,
) -> None:
    """One row per (qid[/mtype], swap) item."""
    fieldnames = [
        "qid", "original_question_type", "mutation_type",
        "swap", "story_title", "correct_answer",
        "mutated_question", "mutated_distractor",
        "best_search_fitness", "best_attempt_index",
        "best_judge_confidence_estimated",
    ]
    out_path = eval_dir / "elites.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid, rec, swap in items:
            base = pool.get(qid) or {}
            writer.writerow({
                "qid": qid,
                "original_question_type": rec.get("original_question_type"),
                "mutation_type": rec.get("mutation_type") or rec.get("best_mutation_type"),
                "swap": swap,
                "story_title": base.get("story_title", ""),
                "correct_answer": base.get("correct_answer", ""),
                "mutated_question": rec.get("best_mutated_question", ""),
                "mutated_distractor": rec.get("best_mutated_distractor", ""),
                "best_search_fitness": rec.get("best_fitness"),
                "best_attempt_index": rec.get("best_attempt_index"),
                "best_judge_confidence_estimated":
                    rec.get("best_judge_confidence_estimated", False),
            })
    print(f"[flat-eval] elites CSV → {out_path}")


# ── Aggregation and writing ───────────────────────────────────────────────────

def _aggregate_and_write(
    *,
    rows: list[dict],
    judge_model: str,
    seed: int,
    eval_dir: Path,
    baseline_twr: float | None,
    no_plot: bool,
    suffix: str,
) -> dict:
    """Compute metrics and write eval_results + eval_summary CSVs/JSONs."""
    results_csv  = str(eval_dir / f"eval_results{suffix}.csv")
    summary_json = str(eval_dir / f"eval_summary{suffix}.json")

    valid   = [r for r in rows if r["judge_answer"] is not None]
    invalid = [r for r in rows if r["judge_answer"] is None]
    n       = len(valid)

    correct            = sum(1 for r in valid if r["judge_correct"] is True)
    incorrect          = n - correct
    truth_win_rate     = correct / n if n else 0.0
    deceptive_win_rate = incorrect / n if n else 0.0
    asd                = truth_win_rate - deceptive_win_rate
    std_err            = math.sqrt(truth_win_rate * (1 - truth_win_rate) / n) if n else 0.0
    ci_lo, ci_hi       = _wilson_ci(correct, n) if n else (0.0, 0.0)
    invalid_rate       = len(invalid) / len(rows) if rows else 0.0
    fallback_rate      = sum(1 for r in rows if r.get("judge_fallback")) / len(rows) if rows else 0.0

    picked_a    = sum(1 for r in valid if r["judge_answer"] == "A")
    base_rate_a = picked_a / n if n else 0.0

    non_swap = [r for r in valid if not r["swap"]]
    swapped  = [r for r in valid if r["swap"]]

    def _acc(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return sum(1 for r in subset if r["judge_correct"] is True) / len(subset)

    acc_non_swap = _acc(non_swap)
    acc_swapped  = _acc(swapped)
    if acc_non_swap is not None and acc_swapped is not None:
        position_adjusted: float | None = 0.5 * acc_non_swap + 0.5 * acc_swapped
    else:
        position_adjusted = truth_win_rate

    def _twr_by(key: str) -> dict[str, float]:
        groups: dict[str, list[dict]] = {}
        for r in valid:
            groups.setdefault(r.get(key, ""), []).append(r)
        return {k: sum(1 for r in recs if r["judge_correct"] is True) / len(recs)
                for k, recs in groups.items() if recs and k}

    summary: dict = {
        "debater_model":              config.QD_EVAL_DEBATER_MODEL,
        "judge_model":                judge_model,
        "random_seed":                seed,
        "total_questions":            len(rows),
        "valid_judgements":           n,
        "invalid_judgements":         len(invalid),
        "judge_correct":              correct,
        "judge_incorrect":            incorrect,
        "truth_win_rate":             round(truth_win_rate, 4),
        "deceptive_win_rate":         round(deceptive_win_rate, 4),
        "asd":                        round(asd, 4),
        "base_rate_a":                round(base_rate_a, 4),
        "n_judge_a":                  picked_a,
        "n_judge_b":                  n - picked_a,
        "invalid_rate":               round(invalid_rate, 4),
        "std_err":                    round(std_err, 4),
        "accuracy_95ci":              [round(ci_lo, 4), round(ci_hi, 4)],
        "accuracy_95ci_method":       "wilson",
        "position_adjusted_accuracy": _round4(position_adjusted),
        "accuracy_non_swap":          _round4(acc_non_swap),
        "accuracy_swapped":           _round4(acc_swapped),
        "n_non_swap":                 len(non_swap),
        "n_swapped":                  len(swapped),
        "by_question_type":           _twr_by("question_type"),
        "by_mutation_type":           _twr_by("mutation_type"),
        "baseline_truth_win_rate":    baseline_twr,
        "delta_vs_baseline":          _round4(truth_win_rate - baseline_twr) if baseline_twr is not None else None,
        "fallback_rate":              round(fallback_rate, 4),
    }

    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tag = _model_tag(judge_model)
    print(f"[flat-eval] {tag}  TWR={truth_win_rate:.4f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"ASD={asd:+.4f}  pos_adj={_round4(position_adjusted)}  base_rate_A={base_rate_a:.3f}")
    print(f"[flat-eval]   acc(truth=A)={acc_non_swap}  acc(truth=B)={acc_swapped}  "
          f"n_non_swap={len(non_swap)}  n_swapped={len(swapped)}")
    if baseline_twr is not None:
        print(f"[flat-eval]   delta vs baseline = {truth_win_rate - baseline_twr:+.4f}  "
              f"(baseline TWR={baseline_twr:.4f})")
    print(f"[flat-eval]   wrote {results_csv}")
    print(f"[flat-eval]   wrote {summary_json}")

    if not no_plot:
        try:
            _render_breakdown_plot(summary, eval_dir, judge_model, suffix)
        except Exception as e:
            print(f"[flat-eval]   plot rendering failed (non-fatal): {e}")

    return summary


def _render_breakdown_plot(summary: dict, eval_dir: Path, judge_model: str, suffix: str) -> None:
    """Simple per-mutation TWR bar chart written to eval_dir/plots/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = eval_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    by_mt = summary.get("by_mutation_type", {})
    by_qt = summary.get("by_question_type", {})
    baseline = summary.get("baseline_truth_win_rate")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, data, title in (
        (axes[0], by_mt, "TWR by mutation_type"),
        (axes[1], by_qt, "TWR by question_type"),
    ):
        if not data:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        labels = list(data.keys())
        values = [data[k] for k in labels]
        ax.bar(labels, values, color="#7297c1")
        if baseline is not None:
            ax.axhline(baseline, color="#cc4444", linestyle="--", linewidth=1,
                       label=f"baseline TWR={baseline:.3f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Truth-win rate")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Flat-eval breakdown — {_model_tag(judge_model)}", fontsize=11)
    fig.tight_layout()
    out_path = plots / f"eval_breakdown{suffix}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[flat-eval]   plot → {out_path}")


# ── --all-seeds aggregation ───────────────────────────────────────────────────

def _aggregate_seeds_flat(
    base_results: Path,
    judges: list[str],
    eval_subdir: str,
) -> dict:
    """Read per-seed eval summaries and write flat_aggregate_summary.json."""
    aggregate: dict[str, dict] = {}
    for judge_model in judges:
        suffix = f"_{_model_tag(judge_model)}" if len(judges) > 1 else ""
        per_seed: list[dict] = []
        for seed in SEED_TO_BASELINE:
            seed_dir = base_results / f"seed_{seed}"
            run_dir = latest_flat_run_dir(seed_dir) if seed_dir.exists() else None
            if run_dir is None:
                print(f"[flat-eval-agg] seed={seed}: no flat_grid_run dir — skipping")
                continue
            summary_path = run_dir / eval_subdir / f"eval_summary{suffix}.json"
            if not summary_path.exists():
                print(f"[flat-eval-agg] seed={seed}: missing {summary_path.name} — skipping")
                continue
            with open(summary_path) as f:
                per_seed.append({"seed": seed, "run_dir": str(run_dir), **json.load(f)})

        if not per_seed:
            print(f"[flat-eval-agg] {judge_model}: no per-seed summaries found — skipping")
            continue

        twrs = [s["truth_win_rate"] for s in per_seed]
        deltas = [s["delta_vs_baseline"] for s in per_seed
                  if s.get("delta_vs_baseline") is not None]
        pooled_correct = sum(s["judge_correct"] for s in per_seed)
        pooled_valid = sum(s["valid_judgements"] for s in per_seed)
        pooled_twr = pooled_correct / pooled_valid if pooled_valid else 0.0
        ci_lo, ci_hi = _wilson_ci(pooled_correct, pooled_valid) if pooled_valid else (0.0, 0.0)

        agg = {
            "judge_model": judge_model,
            "n_seeds": len(per_seed),
            "seeds": [s["seed"] for s in per_seed],
            "mean_twr": round(statistics.fmean(twrs), 4),
            "sd_twr": round(statistics.pstdev(twrs), 4) if len(twrs) > 1 else 0.0,
            "min_twr": round(min(twrs), 4),
            "max_twr": round(max(twrs), 4),
            "pooled_correct": pooled_correct,
            "pooled_valid": pooled_valid,
            "pooled_twr": round(pooled_twr, 4),
            "pooled_95ci_wilson": [round(ci_lo, 4), round(ci_hi, 4)],
            "mean_delta_vs_baseline": round(statistics.fmean(deltas), 4) if deltas else None,
            "per_seed": [
                {"seed": s["seed"], "twr": s["truth_win_rate"],
                 "asd": s.get("asd"), "delta": s.get("delta_vs_baseline"),
                 "valid": s["valid_judgements"], "correct": s["judge_correct"]}
                for s in per_seed
            ],
        }
        aggregate[judge_model] = agg
        print(f"[flat-eval-agg] {judge_model}: mean_TWR={agg['mean_twr']}  "
              f"SD={agg['sd_twr']}  pooled_CI={agg['pooled_95ci_wilson']}  "
              f"mean_delta={agg['mean_delta_vs_baseline']}  n_seeds={agg['n_seeds']}")

    out_path = base_results / "flat_aggregate_summary.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"[flat-eval-agg] wrote {out_path}")
    return aggregate


def _run_all_seeds_flat(args: argparse.Namespace) -> None:
    """Spawn one subprocess per known seed, concurrently behind pooled credentials.

    Each child writes to its own `qd_results/flat_results/seed_N/flat_grid_run_NNN/...`
    folder, so the cross-process file-lock concern from CLAUDE.md §14 is
    satisfied. Concurrency is bounded by `--parallel-seeds` (default 5).
    """
    base_results = flat_results_root(test=args.test)
    # Determine eval_subdir based on mode and double_swap
    double_swap = args.double_swap
    mode = args.mode
    if mode == "best":
        eval_subdir = "eval_flat_best_double" if double_swap else "eval_flat_best"
    else:
        eval_subdir = "eval_flat_per_mutation_double" if double_swap else "eval_flat_per_mutation"

    base_argv = [sys.executable, os.path.abspath(__file__)]
    if args.test:        base_argv.append("--test")
    if args.no_plot:     base_argv.append("--no-plot")
    if args.double_swap: base_argv.append("--double-swap")
    if args.mode != "best": base_argv += ["--mode", args.mode]
    if args.judges:      base_argv += ["--judges", *args.judges]
    if args.workers:     base_argv += ["--workers", str(args.workers)]

    from qd.parallel_run import default_log_path, run_seeds_in_parallel

    def _build_argv(seed: int) -> list[str]:
        return base_argv + ["--seed", str(seed)]

    def _log_path(seed: int) -> Path:
        return default_log_path(seed, "flat-eval", base_dir=base_results)

    def _prefix(seed: int) -> str:
        return f"[seed {seed:>3}]"

    print(f"[flat-eval-all] launching {len(SEED_TO_BASELINE)} seed(s) "
          f"concurrently (parallel_seeds={args.parallel_seeds})")
    results = run_seeds_in_parallel(
        SEED_TO_BASELINE.keys(),
        build_argv=_build_argv,
        log_path_for=_log_path,
        prefix_for=_prefix,
        parallel_workers=args.parallel_seeds,
    )
    failures = [(s, rc) for s, rc in results if rc != 0]

    judges = args.judges if args.judges else [config.QD_EVAL_JUDGE_MODEL]
    print(f"\n[flat-eval-all] ─── aggregating across seeds ─────────────────────")
    aggregate = _aggregate_seeds_flat(base_results, judges, eval_subdir)
    if failures:
        print(f"[flat-eval-all] WARNING — {len(failures)} seed(s) failed: {failures}")

    agg_lines = []
    for judge_model, agg in aggregate.items():
        tag = _model_tag(judge_model)
        mean_twr = agg.get("mean_twr")
        sd_twr = agg.get("sd_twr")
        mean_delta = agg.get("mean_delta_vs_baseline")
        delta_str = f"{mean_delta:+.4f}" if mean_delta is not None else "n/a"
        agg_lines.append(
            f"  - {tag}: mean_TWR={mean_twr:.4f}, SD={sd_twr:.4f}, "
            f"mean_delta={delta_str}, n_seeds={agg.get('n_seeds')}"
        )
    swap_label = "double-swap" if args.double_swap else "single-swap"
    fail_str = f"\nfailures: {failures}" if failures else ""
    body = (
        "judges:\n" + "\n".join(agg_lines)
        if agg_lines else "no aggregate summaries"
    )
    slack(
        f":white_check_mark: flat_eval ALL SEEDS done, "
        f"mode={args.mode}, swap={swap_label}\n{body}{fail_str}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Per-seed eval: read flat CSV from "
                             "qd_results/seed_N/flat_grid_run_<latest>/.")
    parser.add_argument("--run", type=int, default=None,
                        help="Evaluate flat_grid_run_M (default: latest).")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Loop the 5 known seeds and aggregate into "
                             "qd_results/flat_aggregate_summary.json.")
    parser.add_argument("--mode", choices=["best", "per_mutation"], default="best",
                        help="best: evaluate flat_best_per_qid.csv (one row per qid); "
                             "per_mutation: evaluate flat_per_mutation.csv "
                             "(one row per qid/mtype).")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="One or more eval judge models. "
                             "Default: config.QD_EVAL_JUDGE_MODEL.")
    parser.add_argument("--double-swap", action="store_true",
                        help="Debate each covered row twice (swap=False + swap=True). "
                             "Default (no flag) is one debate per row with a balanced "
                             "50/50 swap split — same recipe as the baseline.")
    parser.add_argument("--test", action="store_true",
                        help="Use qd_results_test/ paths.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the per-judge breakdown plot.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for debate and judge phases (default 8).")
    parser.add_argument("--parallel-seeds", type=int, default=5,
                        help="With --all-seeds: number of seed subprocesses to run "
                             "concurrently (default 5). Each child writes to its own "
                             "per-seed folder, so concurrency is safe with pooled "
                             "credentials.")
    args = parser.parse_args()

    _t_start = time.time()

    _orig_excepthook = sys.excepthook

    def _abort_hook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            tb_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
            mode_label = (
                "all-seeds" if getattr(args, "all_seeds", False)
                else f"seed={args.seed}, mode={args.mode}"
            )
            slack(
                f":x: flat_eval ABORTED ({mode_label})\n"
                f"```\n{tb_str[-1500:]}\n```"
            )
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _abort_hook

    if args.all_seeds:
        _run_all_seeds_flat(args)
        return

    if args.seed is None:
        raise SystemExit("--seed is required (or pass --all-seeds).")

    config.set_seed(args.seed)
    qd = test_config() if args.test else QDConfig()
    seed = config.RANDOM_SEED

    # Per-seed flat-grid output dir under qd_results/flat_results/seed_N/.
    base_results = flat_seed_dir(args.seed, test=args.test)

    if args.run is not None:
        run_dir = base_results / f"flat_grid_run_{args.run:03d}"
        print(f"[flat-eval] evaluating flat_grid_run_{args.run:03d}")
    else:
        latest = latest_flat_run_dir(base_results)
        if latest is None:
            print(f"[flat-eval] no flat_grid_run_NNN under {base_results}/ — nothing to evaluate.")
            sys.exit(1)
        run_dir = latest
        print(f"[flat-eval] evaluating {run_dir.name} (latest)")

    # ── Resolve the input CSV (directly in run_dir, not search_flat/) ────────
    if args.mode == "best":
        csv_path = run_dir / "flat_best_per_qid.csv"
        csv_label = "flat_best_per_qid.csv"
    else:
        csv_path = run_dir / "flat_per_mutation.csv"
        csv_label = "flat_per_mutation.csv"

    if not csv_path.exists():
        print(f"[flat-eval] missing {csv_path} — was run_flat_grid.py run?")
        sys.exit(1)

    config.validate_llm_credentials()
    from core.gemini_client import format_pool_diagnostics
    print(format_pool_diagnostics(workers=args.workers))

    # ── Load covered rows from the flat CSV ───────────────────────────────────
    covered_rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("covered") == "True":
                covered_rows.append(row)

    if not covered_rows:
        print(f"[flat-eval] {csv_label} has no covered rows — nothing to evaluate.")
        return

    print(f"[flat-eval] loaded {len(covered_rows)} covered rows from {csv_label}")

    # Pool CSV needed for the `story` field (not stored in flat CSVs).
    # Prefer the pool the search actually used (recorded in flat_grid_state.json);
    # fall back to pool_csv_for_seed() for legacy runs without a state file.
    from qd.config import pool_csv_for_seed
    state_path = run_dir / "flat_grid_state.json"
    pool_path: str | None = None
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                pool_path = json.load(f).get("pool_source") or None
        except Exception:
            pool_path = None
    if pool_path is None:
        pool_path = str(pool_csv_for_seed(args.seed))
    print(f"[flat-eval] pool source: {pool_path}")
    pool = _load_pool(pool_path)

    baseline_twr = _load_baseline_twr(qd.baseline_dir)
    if baseline_twr is not None:
        print(f"[flat-eval] baseline TWR: {baseline_twr:.4f}")

    judges = args.judges if args.judges else [config.QD_EVAL_JUDGE_MODEL]
    multi_judge = len(judges) > 1

    # ── Determine eval_dir ───────────────────────────────────────────────────
    if args.mode == "best":
        eval_dir = run_dir / ("eval_flat_best_double" if args.double_swap else "eval_flat_best")
    else:
        eval_dir = run_dir / ("eval_flat_per_mutation_double" if args.double_swap else "eval_flat_per_mutation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Build debate items ───────────────────────────────────────────────────
    # Each "rec" has best_mutated_question, best_mutated_distractor.
    # Sorted by qid to make the eval order deterministic.
    covered_sorted: list[dict] = sorted(covered_rows, key=lambda r: r.get("qid", ""))

    # For mode=per_mutation, rows are (qid, mutation_type) pairs; for mode=best,
    # rows are per qid. In both cases we use best_mutated_question/distractor.
    if args.double_swap:
        items: list[tuple[str, dict, bool]] = [
            (row["qid"], row, s)
            for row in covered_sorted
            for s in (False, True)
        ]
    else:
        _swap_map = baseline_swap_for_seed(seed)
        items = [
            (row["qid"], row, _swap_map[str(row["qid"])])
            for row in covered_sorted
        ]

    _write_elites_csv(items, pool, eval_dir, args.mode)

    # ── Resume: load already-completed debates from eval_transcripts.jsonl ──────
    transcripts_path = eval_dir / "eval_transcripts.jsonl"
    _completed_debates: dict[tuple[str, bool], dict] = {}
    if transcripts_path.exists():
        with open(transcripts_path, encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _t = json.loads(_line)
                    _k: tuple[str, bool] = (str(_t.get("qid", "")), bool(_t.get("swap", False)))
                    _completed_debates[_k] = _t
                except Exception:
                    pass

    _pending_debate_items = [
        (qid, rec, swap) for qid, rec, swap in items
        if (qid, swap) not in _completed_debates
    ]
    if _completed_debates:
        print(f"[flat-eval] resume: {len(_completed_debates)} debates already done, "
              f"{len(_pending_debate_items)} remaining")

    # ── Debate phase ──────────────────────────────────────────────────────────
    n_debates = len(items)
    print(f"\n[flat-eval] debate phase — {n_debates} debates "
          f"({'2 per row' if args.double_swap else '1 per row (50/50 balanced)'}) "
          f"across {args.workers} worker(s)")
    print(f"[flat-eval] debater={config.QD_EVAL_DEBATER_MODEL}")

    transcripts_writer = JsonlWriter(str(transcripts_path))
    _dr_lock = threading.Lock()
    debate_records: list[tuple[str, dict, bool, dict]] = []
    # Pre-seed with already-completed transcripts (preserves original item order).
    for qid, rec, swap in items:
        _t = _completed_debates.get((qid, swap))
        if _t is not None:
            debate_records.append((qid, rec, swap, _t))

    t_debate = time.time()

    def _debate_one(qid: str, rec: dict, swap: bool) -> None:
        base = pool.get(qid)
        if base is None:
            print(f"[flat-eval] WARN: qid={qid} not in pool — skipping")
            return
        candidate_row = dict(base)
        candidate_row["question"]        = rec["best_mutated_question"]
        candidate_row["negative_answer"] = rec["best_mutated_distractor"]
        try:
            transcript = run_eval_debate(candidate_row, swap)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            err_repr = repr(e)
            err_lc = err_repr.lower()
            if (
                "resource_exhausted" in err_lc
                or "quota" in err_lc
                or " 429" in err_repr
                or "(429" in err_repr
            ):
                slack_throttled(
                    "flat_eval_quota",
                    300.0,
                    f":warning: flat_eval quota hit (debate, qid={qid}): "
                    f"`{err_repr[:200]}`",
                )
            print(f"[flat-eval] DEBATE FAILURE qid={qid}: {e!r}\n{tb}")
            return
        mtype = rec.get("mutation_type") or rec.get("best_mutation_type") or ""
        qtype = rec.get("original_question_type") or ""
        transcripts_writer.write_record({
            **transcript,
            "qid": qid,
            "original_question_type": qtype,
            "mutation_type": mtype,
            "best_search_fitness": rec.get("best_fitness"),
            "best_attempt_index": rec.get("best_attempt_index"),
        })
        with _dr_lock:
            debate_records.append((qid, rec, swap, transcript))
            n = len(debate_records)
        print(f"[flat-eval] {n}/{n_debates} debates done  elapsed={time.time()-t_debate:.0f}s")

    if _pending_debate_items:
        with ThreadPoolExecutor(max_workers=args.workers) as _ex:
            _futs = [_ex.submit(_debate_one, qid, rec, swap)
                     for qid, rec, swap in _pending_debate_items]
            for _f in as_completed(_futs):
                _f.result()
    transcripts_writer.close()
    print(f"[flat-eval] {len(debate_records)}/{n_debates} debates completed in "
          f"{time.time() - t_debate:.1f}s")
    if not debate_records:
        print("[flat-eval] no successful debates — aborting.")
        return

    # ── Judge phase (per judge model) ─────────────────────────────────────────
    print(f"\n[flat-eval] judge phase — {len(judges)} judge model(s)")
    all_summaries: list[tuple[str, dict]] = []
    for judge_model in judges:
        suffix = f"_{_model_tag(judge_model)}" if multi_judge else ""
        j_path = eval_dir / f"eval_judgements{suffix}.jsonl"

        # Resume: load already-judged (qid, swap) pairs.
        # The judgement record includes "swap" so we can reconstruct the key.
        _completed_j: dict[tuple[str, bool], dict] = {}
        if j_path.exists():
            with open(j_path, encoding="utf-8") as _jfh:
                for _jline in _jfh:
                    _jline = _jline.strip()
                    if not _jline:
                        continue
                    try:
                        _j = json.loads(_jline)
                        _jk: tuple[str, bool] = (str(_j.get("qid", "")), bool(_j.get("swap", False)))
                        _completed_j[_jk] = _j
                    except Exception:
                        pass

        n_pending_j = sum(1 for qid, _, swap, _ in debate_records
                          if (qid, swap) not in _completed_j)
        if _completed_j:
            print(f"[flat-eval] resume: {len(_completed_j)} judgements already done for "
                  f"{judge_model}, {n_pending_j} remaining")

        print(f"\n[flat-eval] judging with {judge_model} ({n_pending_j} pending) ...")

        judgements_writer = JsonlWriter(str(j_path))
        rows: list[dict] = []
        _rows_lock = threading.Lock()

        # Pre-seed rows from already-completed judgements.
        for qid, rec, swap, _ in debate_records:
            _j = _completed_j.get((qid, swap))
            if _j is None:
                continue
            mtype = rec.get("mutation_type") or rec.get("best_mutation_type") or ""
            qtype = rec.get("original_question_type") or ""
            rows.append({
                "qid":                  qid,
                "question_type":        qtype,
                "mutation_type":        mtype,
                "swap":                 swap,
                "judge_answer":         _j.get("judge_answer"),
                "judge_correct":        _j.get("judge_correct"),
                "judge_confidence":     _j.get("judge_confidence"),
                "judge_logprob_a":      _j.get("judge_logprob_a"),
                "judge_logprob_b":      _j.get("judge_logprob_b"),
                "judge_log_odds_truth": _j.get("judge_log_odds_truth"),
                "judge_logit_gap_truth":_j.get("judge_logit_gap_truth"),
                "correct_letter":       _j.get("correct_letter"),
                "judge_fallback":       _j.get("judge_fallback"),
                "best_search_fitness":  rec.get("best_fitness"),
            })

        saved_judge = config.QD_EVAL_JUDGE_MODEL
        config.QD_EVAL_JUDGE_MODEL = judge_model
        try:
            _pending_j = [
                (qid, rec, swap, transcript)
                for qid, rec, swap, transcript in debate_records
                if (qid, swap) not in _completed_j
            ]

            def _judge_one(qid: str, rec: dict, swap: bool, transcript: dict) -> None:
                mtype = rec.get("mutation_type") or rec.get("best_mutation_type") or ""
                qtype = rec.get("original_question_type") or ""
                try:
                    judgement = run_eval_judge(transcript)
                except Exception as e:
                    tb = traceback.format_exc(limit=2)
                    err_repr = repr(e)
                    err_lc = err_repr.lower()
                    if (
                        "resource_exhausted" in err_lc
                        or "quota" in err_lc
                        or " 429" in err_repr
                        or "(429" in err_repr
                    ):
                        slack_throttled(
                            "flat_eval_quota",
                            300.0,
                            f":warning: flat_eval quota hit (judge, qid={qid}): "
                            f"`{err_repr[:200]}`",
                        )
                    print(f"[flat-eval] JUDGE FAILURE qid={qid} — retrying in 30s: {e!r}\n{tb}")
                    time.sleep(30)
                    try:
                        judgement = run_eval_judge(transcript)
                    except Exception as e2:
                        tb2 = traceback.format_exc(limit=2)
                        print(f"[flat-eval] JUDGE FAILURE (retry) qid={qid} — skipping: {e2!r}\n{tb2}")
                        return
                j_log = dict(judgement)
                j_log.pop("rounds", None)
                j_log.pop("transcript_str", None)
                judgements_writer.write_record({
                    **j_log,
                    "qid": qid,
                    "swap": swap,
                    "original_question_type": qtype,
                    "mutation_type": mtype,
                    "best_search_fitness": rec.get("best_fitness"),
                    "best_attempt_index": rec.get("best_attempt_index"),
                })
                with _rows_lock:
                    rows.append({
                        "qid":                  qid,
                        "question_type":        qtype,
                        "mutation_type":        mtype,
                        "swap":                 swap,
                        "judge_answer":         judgement.get("judge_answer"),
                        "judge_correct":        judgement.get("judge_correct"),
                        "judge_confidence":     judgement.get("judge_confidence"),
                        "judge_logprob_a":      judgement.get("judge_logprob_a"),
                        "judge_logprob_b":      judgement.get("judge_logprob_b"),
                        "judge_log_odds_truth": judgement.get("judge_log_odds_truth"),
                        "judge_logit_gap_truth":judgement.get("judge_logit_gap_truth"),
                        "correct_letter":       judgement.get("correct_letter"),
                        "judge_fallback":       judgement.get("judge_fallback"),
                        "best_search_fitness":  rec.get("best_fitness"),
                    })

            if _pending_j:
                with ThreadPoolExecutor(max_workers=args.workers) as _ex:
                    _jfuts = [_ex.submit(_judge_one, qid, rec, swap, t)
                               for qid, rec, swap, t in _pending_j]
                    for _jf in as_completed(_jfuts):
                        _jf.result()
        finally:
            config.QD_EVAL_JUDGE_MODEL = saved_judge
            judgements_writer.close()

        if not rows:
            print(f"[flat-eval] no successful judgements for {judge_model}")
            continue
        summary = _aggregate_and_write(
            rows=rows,
            judge_model=judge_model,
            seed=seed,
            eval_dir=eval_dir,
            baseline_twr=baseline_twr,
            no_plot=args.no_plot,
            suffix=suffix,
        )
        if summary:
            all_summaries.append((judge_model, summary))

    # ── Comparison table when >1 judges ──────────────────────────────────────
    if len(all_summaries) > 1:
        print("\n[flat-eval] ══ COMPARISON ════════════════════════════════════════")
        hdr = f"{'judge':<28}  {'TWR':>6}  {'ASD':>6}  {'pos_adj':>7}  {'delta':>7}  {'fallback':>8}"
        print(hdr)
        print("─" * len(hdr))
        for judge_model, s in all_summaries:
            delta = f"{s['delta_vs_baseline']:+.4f}" if s.get("delta_vs_baseline") is not None else "  n/a"
            print(
                f"{_model_tag(judge_model):<28}  "
                f"{s['truth_win_rate']:>6.4f}  {s['asd']:>+6.4f}  "
                f"{s['position_adjusted_accuracy']:>7.4f}  {delta:>7}  "
                f"{s['fallback_rate']:>8.3f}"
            )
        print("═" * len(hdr))

    # ── Slack: per-seed completion notification ──────────────────────────────
    elapsed = time.time() - _t_start
    swap_label = "double-swap" if args.double_swap else "single-swap"
    judge_lines = []
    for judge_model, s in all_summaries:
        tag = _model_tag(judge_model)
        twr = s.get("truth_win_rate", 0.0)
        asd = s.get("asd", 0.0)
        delta = s.get("delta_vs_baseline")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        judge_lines.append(
            f"  - {tag}: TWR={twr:.4f}, ASD={asd:+.4f}, delta={delta_str}"
        )
    body = (
        "judges:\n" + "\n".join(judge_lines)
        if judge_lines else "no judge summaries"
    )
    slack(
        f":white_check_mark: flat_eval done, seed={args.seed}, "
        f"mode={args.mode}, run={run_dir.name}\n"
        f"elapsed: {elapsed:.0f}s, swap: {swap_label}\n"
        f"debates: {len(debate_records)}/{n_debates}\n{body}"
    )


if __name__ == "__main__":
    main()
