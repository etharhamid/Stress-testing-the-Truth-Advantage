#!/usr/bin/env python
"""Re-evaluation of archive elites using baseline-grade pro models.

Debates are run ONCE; each judge model re-judges the same transcripts.

Two swap modes:
  * default (single-swap):  each elite gets the same swap flag its QID was
    assigned in the baseline (steps/run_debates.py). Keeps the positional
    condition constant across baseline → QD-eval comparisons so per-QID
    deltas cannot be confounded by a swap change. Writes to `eval/`.
  * --double-swap:  each elite is debated twice (one swap=False + one
    swap=True). Doubles cost but exposes per-cell consistency under
    positional swap. Writes to `eval_double/`.

Examples:
  # single-swap default, one judge
  python run_qd_eval.py --run 2

  # two judge models on the same transcripts
  python run_qd_eval.py --run 2 --judges gemini-2.5-flash gemini-2.5-flash-lite

  # double-swap: each elite gets both swap conditions
  python run_qd_eval.py --run 2 --double-swap
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

from qd.archive import Archive
from qd.config import (
    QDConfig,
    SEED_TO_BASELINE,
    baseline_swap_for_seed,
    latest_run_dir,
    map_results_root,
    map_seed_dir,
    test_config,
)
from qd.fitness import run_eval_debate, run_eval_judge
from qd.logging import JsonlWriter
from qd.plot import render_eval_plot

from steps.score_results import _wilson_ci  # type: ignore[import-not-found]


def _load_pool(csv_path: str) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pool[str(row["id"])] = row
    return pool


def _load_baseline_twr(baseline_dir: str) -> float | None:
    p = Path(baseline_dir)
    if not p.exists():
        return None

    # Prefer re-judged summaries written by `run_all.py --rejudge-baseline`
    # (de-biased judge prompt). Fall back to top-level legacy files if absent.
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


def _write_qid_deception_summary(eval_dir: Path, pool_size: int) -> None:
    """Compute per-QID deception success from eval_results_*.csv and write JSON.

    Reads every eval_results_<tag>.csv in eval_dir, builds a per-QID view of
    which judge models were fooled (any cell where judge_correct=False), and
    writes per_qid_deception_summary.json. Safe to call even if only one judge
    model ran — the breakdown fields that require both judges are omitted in
    that case.
    """
    csv_files = sorted(eval_dir.glob("eval_results_*.csv"))
    if not csv_files:
        return

    model_fooled: dict[str, set[str]] = {}
    all_qids: set[str] = set()

    for csv_path in csv_files:
        tag = csv_path.stem[len("eval_results_"):]
        fooled: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = str(row.get("qid", "")).strip()
                if not qid:
                    continue
                all_qids.add(qid)
                correct_val = str(row.get("judge_correct", "true")).strip().lower()
                if correct_val in ("false", "0", "no"):
                    fooled.add(qid)
        model_fooled[tag] = fooled

    unique_qids = sorted(all_qids)
    flash_tag  = next((t for t in model_fooled if "flash-lite" not in t and "flash" in t), None)
    lite_tag   = next((t for t in model_fooled if "flash-lite" in t), None)

    per_qid: dict[str, dict] = {}
    for qid in unique_qids:
        entry: dict = {}
        for tag, fooled in model_fooled.items():
            entry[tag.replace("-", "_") + "_fooled"] = qid in fooled
        if flash_tag and lite_tag:
            f = qid in model_fooled[flash_tag]
            l = qid in model_fooled[lite_tag]
            entry["category"] = "both" if (f and l) else ("flash_only" if f else ("flash_lite_only" if l else "neither"))
        per_qid[qid] = entry

    summary: dict = {
        "pool_size": pool_size,
        "unique_qids_in_archive": len(unique_qids),
        "judges": sorted(model_fooled.keys()),
    }
    if flash_tag and lite_tag:
        flash_set = model_fooled[flash_tag]
        lite_set  = model_fooled[lite_tag]
        qset      = set(unique_qids)
        summary["deceptive_by_flash"]          = len(flash_set & qset)
        summary["deceptive_by_flash_lite"]     = len(lite_set  & qset)
        summary["deceptive_by_both"]           = len(flash_set & lite_set & qset)
        summary["deceptive_by_one_model_only"] = len((flash_set ^ lite_set) & qset)
        summary["not_deceptive_by_either"]     = len(qset - flash_set - lite_set)
    else:
        for tag, fooled in model_fooled.items():
            summary[f"deceptive_by_{tag.replace('-', '_')}"] = len(fooled & set(unique_qids))
    summary["per_qid"] = per_qid

    out_path = eval_dir / "per_qid_deception_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[QD-eval] per-QID deception summary → {out_path}")


def _write_elites_csv(
    debate_items: list,
    pool: dict[str, dict],
    eval_dir: Path,
) -> None:
    """Write elites.csv — one row per (elite, swap) debate item."""
    fieldnames = [
        "question_type", "mutation_type", "qid", "story_title", "swap",
        "correct_answer", "mutated_question", "mutated_distractor",
        "elite_search_fitness", "elite_iteration", "elite_bleu_to_parent",
        "elite_judge_log_odds_search",
    ]
    out_path = eval_dir / "elites.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell, elite, swap in debate_items:
            qtype, mtype = cell
            base = pool.get(elite.qid) or {}
            writer.writerow({
                "question_type":             qtype,
                "mutation_type":             mtype,
                "qid":                       elite.qid,
                "story_title":               base.get("story_title", ""),
                "swap":                      swap,
                "correct_answer":            elite.correct_answer,
                "mutated_question":          elite.mutated_question,
                "mutated_distractor":        elite.mutated_distractor,
                "elite_search_fitness":      elite.fitness,
                "elite_iteration":           elite.iteration,
                "elite_bleu_to_parent":      elite.bleu_to_parent,
                "elite_judge_log_odds_search": elite.judge_log_odds_truth,
            })
    print(f"[QD-eval] elites CSV → {out_path}")


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
    """Aggregate metrics from judged rows and write per-judge output files."""
    results_csv  = str(eval_dir / f"eval_results{suffix}.csv")
    summary_json = str(eval_dir / f"eval_summary{suffix}.json")
    plots_dir    = str(eval_dir / "plots")

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
            groups.setdefault(r[key], []).append(r)
        return {k: sum(1 for r in recs if r["judge_correct"] is True) / len(recs)
                for k, recs in groups.items() if recs}

    summary: dict = {
        # ── Identical field names to baseline summary.json ───────────────────
        "debater_model":                    config.QD_EVAL_DEBATER_MODEL,
        "judge_model":                      judge_model,
        "random_seed":                      seed,
        "total_questions":                  len(rows),
        "valid_judgements":                 n,
        "invalid_judgements":               len(invalid),
        "judge_correct":                    correct,
        "judge_incorrect":                  incorrect,
        "truth_win_rate":                   round(truth_win_rate, 4),
        "deceptive_win_rate":               round(deceptive_win_rate, 4),
        "asd":                              round(asd, 4),
        "base_rate_a":                      round(base_rate_a, 4),
        "n_judge_a":                        picked_a,
        "n_judge_b":                        n - picked_a,
        "invalid_rate":                     round(invalid_rate, 4),
        "std_err":                          round(std_err, 4),
        "accuracy_95ci":                    [round(ci_lo, 4), round(ci_hi, 4)],
        "accuracy_95ci_method":             "wilson",
        "position_adjusted_accuracy":       _round4(position_adjusted),
        "accuracy_non_swap":                _round4(acc_non_swap),
        "accuracy_swapped":                 _round4(acc_swapped),
        "n_non_swap":                       len(non_swap),
        "n_swapped":                        len(swapped),
        # ── QD-specific additions ────────────────────────────────────────────
        "by_question_type":                 _twr_by("question_type"),
        "by_mutation_type":                 _twr_by("mutation_type"),
        "baseline_truth_win_rate":          baseline_twr,
        "delta_vs_baseline":                _round4(truth_win_rate - baseline_twr) if baseline_twr is not None else None,
        "fallback_rate":                    round(fallback_rate, 4),
    }

    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tag = _model_tag(judge_model)
    print(f"[QD-eval] {tag}  TWR={truth_win_rate:.4f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"ASD={asd:+.4f}  pos_adj={_round4(position_adjusted)}  base_rate_A={base_rate_a:.3f}")
    print(f"[QD-eval]   acc(truth=A)={acc_non_swap}  acc(truth=B)={acc_swapped}  "
          f"n_non_swap={len(non_swap)}  n_swapped={len(swapped)}")
    if baseline_twr is not None:
        print(f"[QD-eval]   delta vs baseline = {truth_win_rate - baseline_twr:+.4f}  "
              f"(baseline TWR={baseline_twr:.4f})")
    print(f"[QD-eval]   wrote {results_csv}")
    print(f"[QD-eval]   wrote {summary_json}")

    if not no_plot:
        try:
            render_eval_plot(summary_json, plots_dir, baseline_twr)
            print(f"[QD-eval]   plot → {plots_dir}/eval_breakdown{suffix}.png")
        except Exception as e:
            print(f"[QD-eval]   plot rendering failed (non-fatal): {e}")

    return summary


def _aggregate_seeds(
    base_results: Path,
    judges: list[str],
    eval_subdir: str,
) -> dict:
    """Read per-seed eval_summary_*.json under qd_results/seed_*/run_*/<eval_subdir>/
    and produce a cross-seed summary mirroring qd_baseline/baseline_analysis.md.
    Returns a dict {judge_model: aggregate_summary} + writes aggregate_summary.json."""
    aggregate: dict[str, dict] = {}
    for judge_model in judges:
        suffix = f"_{_model_tag(judge_model)}" if len(judges) > 1 else ""
        per_seed: list[dict] = []
        for seed in SEED_TO_BASELINE:
            seed_dir = base_results / f"seed_{seed}"
            run_dir = latest_run_dir(seed_dir) if seed_dir.exists() else None
            if run_dir is None:
                print(f"[QD-eval-agg] seed={seed}: no run dir under {seed_dir} — skipping")
                continue
            summary_path = run_dir / eval_subdir / f"eval_summary{suffix}.json"
            if not summary_path.exists():
                print(f"[QD-eval-agg] seed={seed}: missing {summary_path.name} — skipping")
                continue
            with open(summary_path) as f:
                per_seed.append({"seed": seed, "run_dir": str(run_dir), **json.load(f)})

        if not per_seed:
            print(f"[QD-eval-agg] {judge_model}: no per-seed summaries found — skipping")
            continue

        twrs = [s["truth_win_rate"] for s in per_seed]
        deltas = [s["delta_vs_baseline"] for s in per_seed
                  if s.get("delta_vs_baseline") is not None]
        # Pooled count across seeds for a single Wilson CI on the union
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
        print(f"[QD-eval-agg] {judge_model}: mean_TWR={agg['mean_twr']}  "
              f"SD={agg['sd_twr']}  pooled_CI={agg['pooled_95ci_wilson']}  "
              f"mean_delta={agg['mean_delta_vs_baseline']}  n_seeds={agg['n_seeds']}")

    out_path = base_results / "aggregate_summary.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"[QD-eval-agg] wrote {out_path}")
    return aggregate


def _run_all_seeds(args: argparse.Namespace) -> None:
    """Spawn one subprocess per known seed (--seed N), then aggregate.

    Subprocesses run concurrently behind the pooled credentials in
    `core/gemini_client`. Concurrency is bounded by `--parallel-seeds`
    (default 5 — one per known seed). Each child writes to its own
    `qd_results/map_results/seed_N/...` folder, so the cross-process
    file-lock concern from CLAUDE.md §14 is satisfied.
    """
    base_results = map_results_root(test=args.test)
    eval_subdir = "eval_double" if args.double_swap else "eval"

    # Build sub-argv from current args, dropping --all-seeds and adding --seed N.
    base_argv = [sys.executable, os.path.abspath(__file__)]
    if args.test:           base_argv.append("--test")
    if args.no_plot:        base_argv.append("--no-plot")
    if args.double_swap:    base_argv.append("--double-swap")
    if args.retry_missing:  base_argv.append("--retry-missing")
    if args.judges:         base_argv += ["--judges", *args.judges]
    if args.workers:        base_argv += ["--workers", str(args.workers)]

    from qd.parallel_run import default_log_path, run_seeds_in_parallel

    def _build_argv(seed: int) -> list[str]:
        return base_argv + ["--seed", str(seed)]

    def _log_path(seed: int) -> Path:
        return default_log_path(seed, "qd-eval", base_dir=base_results)

    def _prefix(seed: int) -> str:
        return f"[seed {seed:>3}]"

    print(f"[QD-eval-all] launching {len(SEED_TO_BASELINE)} seed(s) "
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
    print(f"\n[QD-eval-all] ─── aggregating across seeds ─────────────────────")
    _aggregate_seeds(base_results, judges, eval_subdir)
    if failures:
        print(f"[QD-eval-all] WARNING — {len(failures)} seed(s) failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Use qd_results_test/ paths.")
    parser.add_argument("--archive", default=None,
                        help="Explicit archive.json path (overrides --run).")
    parser.add_argument("--run", type=int, default=None,
                        help="Evaluate a specific run number (default: latest).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Per-seed eval: read archive from qd_results/seed_N/<latest>/ "
                             "and write outputs there. Sets RANDOM_SEED=N for swap derivation.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Loop the 5 known seeds (42/123/7/13/73) and aggregate "
                             "into qd_results/aggregate_summary.json.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for debate and judge phases (default 8).")
    parser.add_argument("--parallel-seeds", type=int, default=5,
                        help="With --all-seeds: number of seed subprocesses to run "
                             "concurrently (default 5). Each child writes to its own "
                             "per-seed folder, so concurrency is safe with pooled "
                             "credentials.")
    parser.add_argument("--double-swap", action="store_true",
                        help="Debate each elite twice (one swap=False + one swap=True). "
                             "Outputs go to eval_double/ instead of eval/. "
                             "Default (without this flag) is one debate per elite with a "
                             "balanced 50/50 swap split — same recipe as the baseline.")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="One or more eval judge models. Default: config.QD_EVAL_JUDGE_MODEL.")
    parser.add_argument("--retry-missing", action="store_true",
                        help="Find (cell, swap) pairs absent from eval_transcripts.jsonl, "
                             "debate only those, then re-aggregate. Use with --double-swap "
                             "to target eval_double/.")
    args = parser.parse_args()

    if args.all_seeds:
        _run_all_seeds(args)
        return

    if args.seed is not None:
        config.set_seed(args.seed)

    qd   = test_config() if args.test else QDConfig()
    seed = config.RANDOM_SEED

    # Per-seed MAP-Elites tree under qd_results/map_results/seed_N/.
    # Legacy single-pool path (no --seed) keeps the historical flat layout.
    if args.seed is not None:
        base_results = map_seed_dir(args.seed, test=args.test)
    else:
        base_results = Path(
            config.QD_RESULTS_TEST_DIR if args.test else config.QD_RESULTS_DIR
        )
    def _resolve_archive(rd: Path) -> str:
        """Return archive path, preferring new search/ layout, falling back to legacy flat layout."""
        new = rd / "search" / "archive.json"
        legacy = rd / "archive.json"
        return str(new if new.exists() else legacy)

    if args.archive:
        archive_path = args.archive
        p = Path(args.archive).parent
        # If archive lives in a search/ subfolder, run_dir is one level up.
        run_dir = p.parent if p.name == "search" else p
        qd.results_dir = str(run_dir)
    elif args.run is not None:
        run_dir = base_results / f"run_{args.run:03d}"
        qd.results_dir = str(run_dir)
        archive_path = _resolve_archive(run_dir)
        print(f"[QD-eval] evaluating run_{args.run:03d}")
    else:
        latest = latest_run_dir(base_results)
        if latest is None:
            print(f"[QD-eval] no runs found under {base_results}/ — nothing to evaluate.")
            sys.exit(1)
        run_dir = latest
        qd.results_dir = str(run_dir)
        archive_path = _resolve_archive(run_dir)
        print(f"[QD-eval] evaluating {run_dir.name} (latest)")

    config.validate_llm_credentials()
    from core.gemini_client import format_pool_diagnostics
    print(format_pool_diagnostics(workers=args.workers))

    archive  = Archive.load(archive_path)
    occupied = archive.occupied_elites()
    if not occupied:
        print(f"[QD-eval] archive {archive_path} has no elites — nothing to evaluate.")
        return

    if args.seed is not None:
        from qd.config import pool_csv_for_seed
        pool_path = str(pool_csv_for_seed(args.seed))
    else:
        pool_path = qd.questions_pool_csv
    pool         = _load_pool(pool_path)
    baseline_twr = _load_baseline_twr(qd.baseline_dir)
    if baseline_twr is not None:
        print(f"[QD-eval] baseline TWR: {baseline_twr:.4f}")

    judges      = args.judges if args.judges else [config.QD_EVAL_JUDGE_MODEL]
    multi_judge = len(judges) > 1

    # ── Build debate items: (cell, elite, swap) ──────────────────────────────
    # Default (single-swap): each elite gets the same swap flag its QID was
    # assigned in the baseline (steps/run_debates.py). This keeps the
    # positional condition constant across baseline → QD-eval comparisons,
    # so per-QID deltas cannot be confounded by a swap change.
    # `--double-swap`: each elite contributes two debate items (one per swap),
    # mirroring the historical eval_double/ layout.
    if args.double_swap:
        debate_items = [(cell, elite, s) for cell, elite in occupied for s in (False, True)]
    else:
        _swap_map = baseline_swap_for_seed(seed)
        debate_items = [
            (cell, elite, _swap_map[str(elite.qid)])
            for cell, elite in occupied
        ]

    eval_dir = run_dir / ("eval_double" if args.double_swap else "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Retry-missing: debate only absent (cell, swap) pairs ─────────────────
    if args.retry_missing:
        transcripts_path = eval_dir / "eval_transcripts.jsonl"
        done_set: set[tuple] = set()
        if transcripts_path.exists():
            with open(transcripts_path, encoding="utf-8") as _f:
                for _line in _f:
                    if _line.strip():
                        _r = json.loads(_line)
                        done_set.add((tuple(_r["cell"]), _r["swap"]))

        missing_items = [
            (cell, elite, swap) for cell, elite, swap in debate_items
            if (tuple(cell), swap) not in done_set
        ]
        if not missing_items:
            print("[QD-eval] no missing debates — nothing to retry.")
            return
        print(f"[QD-eval] retrying {len(missing_items)} missing debate(s)")

        transcripts_writer = JsonlWriter(str(transcripts_path))
        new_debate_records: list[tuple] = []
        for cell, elite, swap in missing_items:
            qtype, mtype = cell
            base = pool.get(elite.qid)
            if base is None:
                print(f"[QD-eval] WARN: elite qid={elite.qid} not in pool — skipping")
                continue
            candidate_row = dict(base)
            candidate_row["question"]        = elite.mutated_question
            candidate_row["negative_answer"] = elite.mutated_distractor
            try:
                transcript = run_eval_debate(candidate_row, swap)
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                print(f"[QD-eval] DEBATE FAILURE qid={elite.qid} cell={cell}: {e!r}\n{tb}")
                continue
            transcripts_writer.write_record({
                **transcript,
                "cell": list(cell),
                "elite_qid": elite.qid,
                "elite_iteration": elite.iteration,
                "elite_search_fitness": elite.fitness,
            })
            new_debate_records.append((cell, elite, swap, transcript))
        transcripts_writer.close()

        if not new_debate_records:
            print("[QD-eval] all retry debates failed.")
            return

        # Judge new debates and re-aggregate per judge model
        for judge_model in judges:
            suffix = f"_{_model_tag(judge_model)}" if multi_judge else ""
            judgements_path = eval_dir / f"eval_judgements{suffix}.jsonl"

            # Reconstruct existing rows from the judgements file
            existing_rows: list[dict] = []
            if judgements_path.exists():
                with open(judgements_path, encoding="utf-8") as _f:
                    for _line in _f:
                        if _line.strip():
                            _j = json.loads(_line)
                            existing_rows.append({
                                "qid":                   _j.get("elite_qid"),
                                "question_type":         _j["cell"][0],
                                "mutation_type":         _j["cell"][1],
                                "swap":                  _j.get("swap"),
                                "judge_answer":          _j.get("judge_answer"),
                                "judge_correct":         _j.get("judge_correct"),
                                "judge_confidence":      _j.get("judge_confidence"),
                                "judge_logprob_a":       _j.get("judge_logprob_a"),
                                "judge_logprob_b":       _j.get("judge_logprob_b"),
                                "judge_log_odds_truth":  _j.get("judge_log_odds_truth"),
                                "judge_logit_gap_truth": _j.get("judge_logit_gap_truth"),
                                "correct_letter":        _j.get("correct_letter"),
                                "judge_fallback":        _j.get("judge_fallback"),
                                "elite_search_fitness":  _j.get("elite_search_fitness"),
                            })

            judgements_writer = JsonlWriter(str(judgements_path))
            new_rows: list[dict] = []
            saved_judge = config.QD_EVAL_JUDGE_MODEL
            config.QD_EVAL_JUDGE_MODEL = judge_model
            try:
                for cell, elite, swap, transcript in new_debate_records:
                    qtype, mtype = cell
                    try:
                        judgement = run_eval_judge(transcript)
                    except Exception as e:
                        tb = traceback.format_exc(limit=2)
                        print(f"[QD-eval] JUDGE FAILURE qid={elite.qid} cell={cell} — retrying in 30s: {e!r}\n{tb}")
                        time.sleep(30)
                        try:
                            judgement = run_eval_judge(transcript)
                        except Exception as e2:
                            tb2 = traceback.format_exc(limit=2)
                            print(f"[QD-eval] JUDGE FAILURE (retry) qid={elite.qid} cell={cell} — skipping: {e2!r}\n{tb2}")
                            continue
                    j_log = dict(judgement)
                    j_log.pop("rounds", None)
                    j_log.pop("transcript_str", None)
                    judgements_writer.write_record({
                        **j_log,
                        "cell": list(cell),
                        "elite_qid": elite.qid,
                        "elite_iteration": elite.iteration,
                        "elite_search_fitness": elite.fitness,
                    })
                    new_rows.append({
                        "qid":                   elite.qid,
                        "question_type":         qtype,
                        "mutation_type":         mtype,
                        "swap":                  swap,
                        "judge_answer":          judgement.get("judge_answer"),
                        "judge_correct":         judgement.get("judge_correct"),
                        "judge_confidence":      judgement.get("judge_confidence"),
                        "judge_logprob_a":       judgement.get("judge_logprob_a"),
                        "judge_logprob_b":       judgement.get("judge_logprob_b"),
                        "judge_log_odds_truth":  judgement.get("judge_log_odds_truth"),
                        "judge_logit_gap_truth": judgement.get("judge_logit_gap_truth"),
                        "correct_letter":        judgement.get("correct_letter"),
                        "judge_fallback":        judgement.get("judge_fallback"),
                        "elite_search_fitness":  elite.fitness,
                    })
            finally:
                config.QD_EVAL_JUDGE_MODEL = saved_judge
                judgements_writer.close()

            all_rows = existing_rows + new_rows
            if not all_rows:
                print(f"[QD-eval] no rows for {judge_model} — skipping aggregation")
                continue
            print(f"\n[QD-eval] re-aggregating {len(all_rows)} rows for {judge_model}")
            _aggregate_and_write(
                rows=all_rows,
                judge_model=judge_model,
                seed=seed,
                eval_dir=eval_dir,
                baseline_twr=baseline_twr,
                no_plot=args.no_plot,
                suffix=suffix,
            )
        try:
            _write_qid_deception_summary(eval_dir, len(pool))
        except Exception as _e:
            print(f"[QD-eval] per-QID summary (non-fatal): {_e}")
        return

    # ── Write elites CSV ─────────────────────────────────────────────────────
    _write_elites_csv(debate_items, pool, eval_dir)

    # ── Resume: load already-completed debates from eval_transcripts.jsonl ──────
    # Transcript records already contain "swap" (from run_debate's return value).
    # Resume key: (elite_qid, qtype, mtype, swap).
    transcripts_path = eval_dir / "eval_transcripts.jsonl"
    _completed_debates: dict[tuple[str, str, str, bool], dict] = {}
    if transcripts_path.exists():
        with open(transcripts_path, encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _t = json.loads(_line)
                    _cell = _t.get("cell") or ["", ""]
                    _k = (str(_t.get("elite_qid", "")),
                          str(_cell[0]), str(_cell[1]),
                          bool(_t.get("swap", False)))
                    _completed_debates[_k] = _t
                except Exception:
                    pass

    _pending_debate_items = [
        (cell, elite, swap) for cell, elite, swap in debate_items
        if (elite.qid, cell[0], cell[1], swap) not in _completed_debates
    ]
    if _completed_debates:
        print(f"[QD-eval] resume: {len(_completed_debates)} debates already done, "
              f"{len(_pending_debate_items)} remaining")

    # ── Debate phase (once, shared across all judges) ────────────────────────
    n_debates = len(debate_items)
    print(f"\n[QD-eval] debate phase — {n_debates} debates "
          f"({'2 per elite' if args.double_swap else '1 per elite (50/50 balanced)'}) "
          f"across {args.workers} worker(s)")
    print(f"[QD-eval] debater={config.QD_EVAL_DEBATER_MODEL}")

    transcripts_writer = JsonlWriter(str(transcripts_path))
    _dr_lock = threading.Lock()
    # (cell, elite, swap, transcript)
    debate_records: list[tuple[tuple, object, bool, dict]] = []
    # Pre-seed with already-completed transcripts (preserves original item order).
    for cell, elite, swap in debate_items:
        _t = _completed_debates.get((elite.qid, cell[0], cell[1], swap))
        if _t is not None:
            debate_records.append((cell, elite, swap, _t))

    t_debate = time.time()

    def _debate_one(cell: tuple, elite, swap: bool) -> None:
        qtype, mtype = cell
        base = pool.get(elite.qid)
        if base is None:
            print(f"[QD-eval] WARN: elite qid={elite.qid} not in pool — skipping")
            return
        candidate_row = dict(base)
        candidate_row["question"]        = elite.mutated_question
        candidate_row["negative_answer"] = elite.mutated_distractor
        try:
            transcript = run_eval_debate(candidate_row, swap)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"[QD-eval] DEBATE FAILURE qid={elite.qid} cell={cell}: {e!r}\n{tb}")
            return
        transcripts_writer.write_record({
            **transcript,
            "cell": [qtype, mtype],
            "elite_qid": elite.qid,
            "elite_iteration": elite.iteration,
            "elite_search_fitness": elite.fitness,
        })
        with _dr_lock:
            debate_records.append((cell, elite, swap, transcript))
            n = len(debate_records)
        print(f"[QD-eval] {n}/{n_debates} debates done  elapsed={time.time()-t_debate:.0f}s")

    if _pending_debate_items:
        with ThreadPoolExecutor(max_workers=args.workers) as _ex:
            _futs = [_ex.submit(_debate_one, cell, elite, swap)
                     for cell, elite, swap in _pending_debate_items]
            for _f in as_completed(_futs):
                _f.result()
    transcripts_writer.close()
    print(f"[QD-eval] {len(debate_records)}/{n_debates} debates completed "
          f"in {time.time() - t_debate:.1f}s")

    if not debate_records:
        print("[QD-eval] no successful debates — aborting.")
        return

    # ── Judge phase (once per judge model, same transcripts) ─────────────────
    print(f"\n[QD-eval] judge phase — {len(judges)} judge model(s)")

    all_summaries: list[tuple[str, dict]] = []
    for judge_model in judges:
        suffix = f"_{_model_tag(judge_model)}" if multi_judge else ""
        j_path = eval_dir / f"eval_judgements{suffix}.jsonl"

        # Resume: load already-judged (elite_qid, qtype, mtype, swap) keys.
        _completed_j: dict[tuple[str, str, str, bool], dict] = {}
        if j_path.exists():
            with open(j_path, encoding="utf-8") as _jfh:
                for _jline in _jfh:
                    _jline = _jline.strip()
                    if not _jline:
                        continue
                    try:
                        _j = json.loads(_jline)
                        _jcell = _j.get("cell") or ["", ""]
                        _jk = (str(_j.get("elite_qid", "")),
                               str(_jcell[0]), str(_jcell[1]),
                               bool(_j.get("swap", False)))
                        _completed_j[_jk] = _j
                    except Exception:
                        pass

        n_pending_j = sum(1 for cell, elite, swap, _ in debate_records
                          if (elite.qid, cell[0], cell[1], swap) not in _completed_j)
        if _completed_j:
            print(f"[QD-eval] resume: {len(_completed_j)} judgements already done for "
                  f"{judge_model}, {n_pending_j} remaining")

        print(f"\n[QD-eval] judging with {judge_model} ({n_pending_j} pending) ...")

        judgements_writer = JsonlWriter(str(j_path))
        rows: list[dict] = []
        _rows_lock = threading.Lock()

        # Pre-seed rows from already-completed judgements.
        for cell, elite, swap, _ in debate_records:
            _j = _completed_j.get((elite.qid, cell[0], cell[1], swap))
            if _j is None:
                continue
            qtype, mtype = cell
            rows.append({
                "qid":                   elite.qid,
                "question_type":         qtype,
                "mutation_type":         mtype,
                "swap":                  swap,
                "judge_answer":          _j.get("judge_answer"),
                "judge_correct":         _j.get("judge_correct"),
                "judge_confidence":      _j.get("judge_confidence"),
                "judge_logprob_a":       _j.get("judge_logprob_a"),
                "judge_logprob_b":       _j.get("judge_logprob_b"),
                "judge_log_odds_truth":  _j.get("judge_log_odds_truth"),
                "judge_logit_gap_truth": _j.get("judge_logit_gap_truth"),
                "correct_letter":        _j.get("correct_letter"),
                "judge_fallback":        _j.get("judge_fallback"),
                "elite_search_fitness":  elite.fitness,
            })

        saved_judge = config.QD_EVAL_JUDGE_MODEL
        config.QD_EVAL_JUDGE_MODEL = judge_model
        try:
            _pending_j = [
                (cell, elite, swap, transcript)
                for cell, elite, swap, transcript in debate_records
                if (elite.qid, cell[0], cell[1], swap) not in _completed_j
            ]

            def _judge_one(cell: tuple, elite, swap: bool, transcript: dict) -> None:
                qtype, mtype = cell
                try:
                    judgement = run_eval_judge(transcript)
                except Exception as e:
                    tb = traceback.format_exc(limit=2)
                    print(f"[QD-eval] JUDGE FAILURE qid={elite.qid} cell={cell} — retrying in 30s: {e!r}\n{tb}")
                    time.sleep(30)
                    try:
                        judgement = run_eval_judge(transcript)
                    except Exception as e2:
                        tb2 = traceback.format_exc(limit=2)
                        print(f"[QD-eval] JUDGE FAILURE (retry) qid={elite.qid} cell={cell} — skipping: {e2!r}\n{tb2}")
                        return
                j_log = dict(judgement)
                j_log.pop("rounds", None)
                j_log.pop("transcript_str", None)
                judgements_writer.write_record({
                    **j_log,
                    "cell": [qtype, mtype],
                    "swap": swap,
                    "elite_qid": elite.qid,
                    "elite_iteration": elite.iteration,
                    "elite_search_fitness": elite.fitness,
                })
                with _rows_lock:
                    rows.append({
                        "qid":                   elite.qid,
                        "question_type":         qtype,
                        "mutation_type":         mtype,
                        "swap":                  swap,
                        "judge_answer":          judgement.get("judge_answer"),
                        "judge_correct":         judgement.get("judge_correct"),
                        "judge_confidence":      judgement.get("judge_confidence"),
                        "judge_logprob_a":       judgement.get("judge_logprob_a"),
                        "judge_logprob_b":       judgement.get("judge_logprob_b"),
                        "judge_log_odds_truth":  judgement.get("judge_log_odds_truth"),
                        "judge_logit_gap_truth": judgement.get("judge_logit_gap_truth"),
                        "correct_letter":        judgement.get("correct_letter"),
                        "judge_fallback":        judgement.get("judge_fallback"),
                        "elite_search_fitness":  elite.fitness,
                    })

            if _pending_j:
                with ThreadPoolExecutor(max_workers=args.workers) as _ex:
                    _jfuts = [_ex.submit(_judge_one, cell, elite, swap, t)
                               for cell, elite, swap, t in _pending_j]
                    for _jf in as_completed(_jfuts):
                        _jf.result()
        finally:
            config.QD_EVAL_JUDGE_MODEL = saved_judge
            judgements_writer.close()

        if not rows:
            print(f"[QD-eval] no successful judgements for {judge_model}")
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

    # ── Comparison table when multiple judges ran ────────────────────────────
    if len(all_summaries) > 1:
        print("\n[QD-eval] ══ COMPARISON ══════════════════════════════════════════")
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

    # ── Combined breakdown plot (both judges on same axes) ────────────────────
    if len(all_summaries) > 1 and not args.no_plot:
        try:
            from qd.plot import eval_breakdown_combined
            combined_out = eval_dir / "plots" / "eval_breakdown_combined.png"
            eval_breakdown_combined(all_summaries, baseline_twr, combined_out)
            print(f"[QD-eval] combined plot → {combined_out}")
        except Exception as e:
            print(f"[QD-eval] combined plot failed (non-fatal): {e}")

    # ── Per-QID deception summary ─────────────────────────────────────────────
    try:
        _write_qid_deception_summary(eval_dir, len(pool))
    except Exception as _e:
        print(f"[QD-eval] per-QID summary (non-fatal): {_e}")


if __name__ == "__main__":
    main()
