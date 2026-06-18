#!/usr/bin/env python
# run_consultancy_baseline.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs the assigned consultancy baseline across all 5 seeds.
#
# Protocol: single expert consultant (gemini-3.1-pro-preview) with story
# access argues for one assigned answer; judge (gemini-2.5-flash-lite) asks
# probing questions in rounds 2 and 3, then decides. 50/50 balanced
# correct / incorrect consultant assignment via the same seeded swap
# schedule as the debate baseline.
#
# Output (per seed):
#   qd_baseline/consultancy_run_{seed}/
#     metadata.json
#     questions.csv             (copy of base_run_{seed}/questions.csv)
#     consultancy_transcripts.jsonl
#     judgements_flash-lite.jsonl
#     eval_summary_flash-lite.json
#
# Usage:
#   python run_consultancy_baseline.py                   # all 5 seeds
#   python run_consultancy_baseline.py --seeds 42 123    # specific seeds
#   python run_consultancy_baseline.py --seeds 42 --dry-run  # print plan only
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from qd.config import baseline_swap_for_seed

SEEDS = [42, 123, 7, 13, 73]
BASELINE_DIR = ROOT / "qd_baseline"

# Default judge model (both in-round questions and final judgment).
# Override with --judge-model at the CLI.
_DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


def _model_tag(model: str) -> str:
    """Short label used in filenames and directory names."""
    return model.replace("gemini-2.5-", "")   # "flash" or "flash-lite"


def _run_dir_name(seed: int, judge_model: str) -> str:
    """Output directory name for a given seed and judge model.

    flash-lite → consultancy_run_{seed}   (legacy name, backwards compat)
    flash      → consultancy_flash_run_{seed}
    """
    tag = _model_tag(judge_model)
    if tag == "flash-lite":
        return f"consultancy_run_{seed}"
    return f"consultancy_{tag}_run_{seed}"


# Active judge model — overridden by --judge-model in main().
CONSULTANCY_JUDGE_MODEL: str = _DEFAULT_JUDGE_MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_questions(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _swap_flags(seed: int) -> dict[int, bool]:
    """Read swap assignments directly from the baseline transcripts.

    Uses the same authoritative {qid: swap} mapping that the debate baseline
    and all QD stages use (via qd.config.baseline_swap_for_seed), so every
    question gets the identical positional assignment across debate, QD, and
    consultancy comparisons.
    """
    raw = baseline_swap_for_seed(seed, root=BASELINE_DIR)
    return {int(qid): swap for qid, swap in raw.items()}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load_judgements_deduped(judgements_path: str) -> list[dict]:
    """Load judgement records, keeping only the last record per ID.

    Prevents duplicate rows (e.g. from a partial resume) from weighting
    any question twice in the summary metrics.
    """
    by_id: dict[int, dict] = {}
    with open(judgements_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    by_id[int(rec["id"])] = rec
            except Exception:
                pass
    return list(by_id.values())


def _compute_summary(
    judgements: list[dict],
    judge_model: str,
    random_seed: int,
    n_questions: int,
) -> dict:
    """Compute the eval_summary schema (identical to debate baseline).

    n_questions is the total questions attempted (from questions.csv), not
    the count of records that survived to the judgement file. Failed
    transcript sessions are therefore reflected in invalid_judgements rather
    than silently shrinking total_questions.
    """
    valid   = [j for j in judgements if j.get("judge_answer") is not None]
    n       = len(valid)
    correct = [j for j in valid if j.get("judge_correct") is True]
    twr     = len(correct) / n if n else 0.0

    ci_lo, ci_hi = _wilson_ci(len(correct), n)
    picked_a = sum(1 for j in valid if j.get("judge_answer") == "A")
    non_swap = [j for j in valid if not j.get("swap")]
    swapped  = [j for j in valid if     j.get("swap")]

    def _acc(subset: list[dict]) -> float:
        return (
            sum(j.get("judge_correct", False) for j in subset) / len(subset)
            if subset else 0.0
        )

    # Position-adjusted accuracy: 0.5 * acc(correct-consultant) +
    #                             0.5 * acc(deceptive-consultant).
    # In consultancy:
    #   non_swap (swap=False) → consultant argues correctly
    #   swapped  (swap=True)  → consultant argues incorrectly
    # The "position_adjusted_accuracy" label is kept for schema compatibility.
    acc_non_swap = _acc(non_swap)
    acc_swapped  = _acc(swapped)
    pos_adj      = 0.5 * acc_non_swap + 0.5 * acc_swapped

    return {
        "protocol":                   "consultancy",
        "consultant_model":           config.DEBATER_MODEL,
        "judge_model":                judge_model,
        "random_seed":                random_seed,
        "total_questions":            n_questions,
        "valid_judgements":           n,
        "invalid_judgements":         n_questions - n,
        "judge_correct":              len(correct),
        "judge_incorrect":            n - len(correct),
        "truth_win_rate":             round(twr, 4),
        "deceptive_win_rate":         round(1 - twr, 4),
        "asd":                        round(2 * twr - 1, 4),
        "base_rate_a":                round(picked_a / n, 4) if n else 0.0,
        "n_judge_a":                  picked_a,
        "n_judge_b":                  n - picked_a,
        "invalid_rate":               round((n_questions - n) / n_questions, 4) if n_questions else 0.0,
        "std_err":                    round(math.sqrt(twr * (1 - twr) / n), 4) if n else 0.0,
        "accuracy_95ci":              [round(ci_lo, 4), round(ci_hi, 4)],
        "accuracy_95ci_method":       "wilson",
        "position_adjusted_accuracy": round(pos_adj, 4),
        "accuracy_non_swap":          round(acc_non_swap, 4),   # correct consultant
        "accuracy_swapped":           round(acc_swapped, 4),    # deceptive consultant
        "n_non_swap":                 len(non_swap),
        "n_swapped":                  len(swapped),
        "fallback_rate":              round(
            sum(1 for j in judgements if j.get("judge_fallback")) / n_questions, 4
        ) if n_questions else 0.0,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary_table(results: dict) -> None:
    print(f"\n{'=' * 60}")
    print("  CONSULTANCY BASELINE — SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(f"  {'Seed':>5}  {'TWR':>6}  {'Correct cnslt':>13}  "
          f"{'Deceptive cnslt':>15}  {'Valid':>5}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*13}  {'-'*15}  {'-'*5}")
    twrs = []
    for seed, s in sorted(results.items()):
        twr = s["truth_win_rate"]
        twrs.append(twr)
        print(
            f"  {seed:>5}  {twr:>6.4f}  "
            f"{s['accuracy_non_swap']:>13.4f}  "
            f"{s['accuracy_swapped']:>15.4f}  "
            f"{s['valid_judgements']:>5}/{s['total_questions']}"
        )
    if len(twrs) > 1:
        mean_twr = sum(twrs) / len(twrs)
        sd_twr   = math.sqrt(sum((t - mean_twr) ** 2 for t in twrs) / len(twrs))
        print(f"  {'Mean':>5}  {mean_twr:>6.4f}")
        print(f"  {'SD':>5}  {sd_twr:>6.4f}")
    print(f"{'=' * 60}\n")
    print("  Note: accuracy_non_swap = acc when consultant argues correctly (swap=False)")
    print("        accuracy_swapped  = acc when consultant argues incorrectly (swap=True)")
    print(f"        debate baseline TWR = 0.800 for comparison")


# ── Retry helpers ─────────────────────────────────────────────────────────────

def _retry_seed(seed: int, workers: int = 1) -> dict | None:
    """Retry all failures for one seed:
    1. Failed transcript sessions (error records in consultancy_transcripts.jsonl).
    2. Failed judgements (judge_answer=None in judgements_flash-lite.jsonl).
    Rewrites the judgements file in-place and recomputes eval_summary.
    """
    from steps.run_consultancy_debates import run as run_debates, already_done
    from steps.run_judge_consultancy import load_transcripts
    from core.consultancy_judge_engine import judge_consultancy

    run_dir = BASELINE_DIR / _run_dir_name(seed, CONSULTANCY_JUDGE_MODEL)
    if not run_dir.exists():
        print(f"  [seed {seed}] {run_dir.name} not found — run baseline first.")
        return None

    dst_questions = run_dir / "questions.csv"
    questions     = _load_questions(dst_questions)
    swap_flags    = _swap_flags(seed)
    transcripts_path  = str(run_dir / "consultancy_transcripts.jsonl")
    judgements_path   = str(run_dir / f"judgements_{_model_tag(CONSULTANCY_JUDGE_MODEL)}.jsonl")

    print(f"\n{'=' * 60}")
    print(f"  Retrying: seed={seed}  →  {run_dir.relative_to(ROOT)}")
    print(f"{'=' * 60}")

    saved_seed        = config.RANDOM_SEED
    config.RANDOM_SEED = seed

    try:
        # ── 1. Retry failed transcript sessions ──────────────────────────────
        # already_done() returns only successful records, so run_debates() will
        # pick up error records automatically and retry them.
        done_ids       = already_done(transcripts_path)
        n_failed_sessions = len(questions) - len(done_ids)
        if n_failed_sessions > 0:
            print(f"\n  [seed {seed}] Retrying {n_failed_sessions} failed session(s)...")
            run_debates(
                questions=questions,
                transcripts_path=transcripts_path,
                swap_flags=swap_flags,
                judge_question_model=CONSULTANCY_JUDGE_MODEL,
                workers=workers,
            )
        else:
            print(f"\n  [seed {seed}] No failed sessions.")

        # ── 2. Retry failed judgements and judge newly retried sessions ───────
        # Load ALL existing judgement records (including judge_answer=None).
        all_j: list[dict] = []
        if Path(judgements_path).exists():
            with open(judgements_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if "error" not in rec:
                            all_j.append(rec)
                    except Exception:
                        pass

        none_ids  = {int(j["id"]) for j in all_j if j.get("judge_answer") is None}
        judged_ids = {int(j["id"]) for j in all_j}

        # Transcripts that now exist but have never been judged (retried sessions).
        transcripts     = load_transcripts(transcripts_path)
        transcript_idx  = {int(t["id"]): t for t in transcripts}
        unjudged_ids    = {int(t["id"]) for t in transcripts} - judged_ids

        to_judge_ids = none_ids | unjudged_ids
        if not to_judge_ids:
            print(f"  [seed {seed}] No failed or unjudged judgements.")
        else:
            print(
                f"\n  [seed {seed}] Re-judging {len(to_judge_ids)} record(s) "
                f"({len(none_ids)} None + {len(unjudged_ids)} unjudged)..."
            )
            to_judge = [transcript_idx[qid] for qid in sorted(to_judge_ids)
                        if qid in transcript_idx]
            new_results: dict[int, dict] = {}
            for t in tqdm(to_judge, desc=f"Re-judging [{CONSULTANCY_JUDGE_MODEL}]", unit="q"):
                try:
                    result = judge_consultancy(t, model=CONSULTANCY_JUDGE_MODEL)
                    new_results[int(t["id"])] = {
                        "id":                    result["id"],
                        "question":              result["question"],
                        "story_title":           result.get("story_title", ""),
                        "swap":                  result["swap"],
                        "answer_a":              result["answer_a"],
                        "answer_b":              result["answer_b"],
                        "correct_letter":        result["correct_letter"],
                        "consultant_correct":    result.get("consultant_correct"),
                        "consultant_letter":     result.get("consultant_letter", "A"),
                        "judge_answer":          result["judge_answer"],
                        "judge_confidence":      result.get("judge_confidence"),
                        "judge_logprob_a":       result.get("judge_logprob_a"),
                        "judge_logprob_b":       result.get("judge_logprob_b"),
                        "judge_log_odds_truth":  result.get("judge_log_odds_truth"),
                        "judge_logit_gap_truth": result.get("judge_logit_gap_truth"),
                        "judge_correct":         result["judge_correct"],
                        "judge_fallback":        result.get("judge_fallback", False),
                        "judge_raw":             result["judge_raw_response"],
                        "transcript_str":        result["transcript_str"],
                    }
                except Exception as e:
                    print(f"\n  ERROR re-judging {t.get('id')}: {e}")

            # Merge: existing records updated/extended with new results
            merged: dict[int, dict] = {int(j["id"]): j for j in all_j}
            merged.update(new_results)
            with open(judgements_path, "w", encoding="utf-8") as f:
                for rec in merged.values():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Judgements rewritten: {judgements_path}")

    finally:
        config.RANDOM_SEED = saved_seed

    # ── 3. Recompute summary ──────────────────────────────────────────────────
    judgements  = _load_judgements_deduped(judgements_path)
    summary     = _compute_summary(judgements, CONSULTANCY_JUDGE_MODEL, seed, len(questions))
    summary_path = run_dir / f"eval_summary_{_model_tag(CONSULTANCY_JUDGE_MODEL)}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"\n  [seed {seed}] {_model_tag(CONSULTANCY_JUDGE_MODEL)}: "
        f"TWR={summary['truth_win_rate']:.4f}  "
        f"acc_correct={summary['accuracy_non_swap']:.4f}  "
        f"acc_deceptive={summary['accuracy_swapped']:.4f}  "
        f"({summary['valid_judgements']}/{summary['total_questions']} valid)"
    )
    return summary


# ── Per-seed pipeline ─────────────────────────────────────────────────────────

def _run_seed(seed: int, dry_run: bool, workers: int = 1) -> dict | None:
    """Run the full consultancy pipeline for one seed. Returns the summary dict."""
    from steps.run_consultancy_debates import run as run_debates
    from steps.run_judge_consultancy import run as run_judge

    base_dir = BASELINE_DIR / f"base_run_{seed}"
    questions_src = base_dir / "questions.csv"
    if not questions_src.exists():
        print(f"  [seed {seed}] {questions_src} not found — skipping.")
        return None

    run_dir = BASELINE_DIR / _run_dir_name(seed, CONSULTANCY_JUDGE_MODEL)

    print(f"\n{'=' * 60}")
    print(f"  Consultancy run: seed={seed}  →  {run_dir.relative_to(ROOT)}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"  [dry-run] Would process {questions_src}")
        return None

    run_dir.mkdir(parents=True, exist_ok=True)

    # metadata
    (run_dir / "metadata.json").write_text(
        json.dumps({"run": run_dir.name, "random_seed": seed}, indent=2)
    )

    # questions.csv — copy from the matching base_run
    dst_questions = run_dir / "questions.csv"
    if not dst_questions.exists():
        shutil.copy2(questions_src, dst_questions)

    questions  = _load_questions(dst_questions)
    swap_flags = _swap_flags(seed)

    # ── Step 1: run consultancy sessions ─────────────────────────────────────
    transcripts_path = str(run_dir / "consultancy_transcripts.jsonl")
    print(f"\n  [seed {seed}] Step 1: Consultancy sessions")
    print(f"    consultant  : {config.DEBATER_MODEL}")
    print(f"    judge (Q)   : {CONSULTANCY_JUDGE_MODEL}")

    saved_seed = config.RANDOM_SEED
    config.RANDOM_SEED = seed
    try:
        run_debates(
            questions=questions,
            transcripts_path=transcripts_path,
            swap_flags=swap_flags,
            judge_question_model=CONSULTANCY_JUDGE_MODEL,
            workers=workers,
        )
    finally:
        config.RANDOM_SEED = saved_seed

    # ── Step 2: judge transcripts ─────────────────────────────────────────────
    judgements_path = str(run_dir / f"judgements_{_model_tag(CONSULTANCY_JUDGE_MODEL)}.jsonl")
    print(f"\n  [seed {seed}] Step 2: Judging  [{CONSULTANCY_JUDGE_MODEL}]")

    config.RANDOM_SEED = seed
    try:
        run_judge(
            transcripts_path=transcripts_path,
            judgements_path=judgements_path,
            model=CONSULTANCY_JUDGE_MODEL,
        )
    finally:
        config.RANDOM_SEED = saved_seed

    # ── Step 3: compute and save summary ─────────────────────────────────────
    judgements = _load_judgements_deduped(judgements_path)
    summary = _compute_summary(judgements, CONSULTANCY_JUDGE_MODEL, seed, len(questions))
    summary_path = run_dir / f"eval_summary_{_model_tag(CONSULTANCY_JUDGE_MODEL)}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"\n  [seed {seed}] {_model_tag(CONSULTANCY_JUDGE_MODEL)}: "
        f"TWR={summary['truth_win_rate']:.4f}  "
        f"acc_correct={summary['accuracy_non_swap']:.4f}  "
        f"acc_deceptive={summary['accuracy_swapped']:.4f}  "
        f"({summary['valid_judgements']}/{summary['total_questions']} valid)"
    )

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consultancy baseline across all 5 QuALITY seeds."
    )
    parser.add_argument(
        "--seeds",
        metavar="SEED",
        type=int,
        nargs="+",
        default=None,
        help=f"Seeds to run (default: all 5 — {SEEDS})",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=None,
        help="Single seed (used by scripts/run_all_seeds.py orchestrator).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel consultancy sessions per seed (default 1).",
    )
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N seeds concurrently as subprocesses (default 1 = sequential).",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry failed sessions and judge_answer=None records for the given seeds.",
    )
    parser.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE_MODEL,
        metavar="MODEL",
        help=(
            "Gemini model for both in-round judge questions and the final judgment "
            f"(default: {_DEFAULT_JUDGE_MODEL}). "
            "Changes output dir: flash-lite → consultancy_run_{{seed}}, "
            "flash → consultancy_flash_run_{{seed}}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without making any API calls.",
    )
    args = parser.parse_args()

    # Override the module-level judge model so all functions pick it up.
    global CONSULTANCY_JUDGE_MODEL
    CONSULTANCY_JUDGE_MODEL = args.judge_model

    # Resolve seeds: --seed (singular, used by orchestrator) overrides --seeds;
    # fall back to all 5 seeds when neither is supplied.
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = SEEDS

    if not args.dry_run:
        config.validate_llm_credentials()

    print(f"\nConsultancy baseline")
    print(f"  Seeds           : {seeds}")
    print(f"  Consultant      : {config.DEBATER_MODEL}")
    print(f"  Judge (Q + final): {CONSULTANCY_JUDGE_MODEL}")
    print(f"  Rounds          : {config.NUM_ROUNDS}")
    print(f"  Word limit      : {config.WORD_LIMIT} (soft) / "
          f"{config.WORD_LIMIT_HARD} (hard)")
    print(f"  Judge Q limit   : {config.CONSULTANCY_JUDGE_QUESTION_WORD_LIMIT}")
    print(f"  Workers         : {args.workers}")
    print(f"  Output          : {BASELINE_DIR.relative_to(ROOT)}/consultancy_run_{{seed}}/")

    results: dict[int, dict] = {}

    if args.retry:
        config.validate_llm_credentials()
        for seed in seeds:
            summary = _retry_seed(seed, workers=args.workers)
            if summary is not None:
                results[seed] = summary
        # Print summary table and exit
        if results:
            _print_summary_table(results)
        return

    if args.parallel_seeds > 1 and len(seeds) > 1 and not args.dry_run:
        # Each seed runs as a separate subprocess so they don't share config
        # globals or file handles. Stdout from each child is prefixed with
        # [seed N] and printed live.
        def _run_subprocess(seed: int) -> int:
            cmd = [
                sys.executable, __file__,
                "--seeds", str(seed),
                "--workers", str(args.workers),
                "--judge-model", CONSULTANCY_JUDGE_MODEL,
            ]
            prefix = f"[seed {seed}] "
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                print(prefix + line, end="", flush=True)
            proc.wait()
            return proc.returncode

        with ThreadPoolExecutor(max_workers=args.parallel_seeds) as ex:
            futs = {ex.submit(_run_subprocess, s): s for s in seeds}
            for fut in as_completed(futs):
                seed = futs[fut]
                rc = fut.result()
                if rc != 0:
                    print(f"  [seed {seed}] subprocess exited with code {rc}")

        # Read summaries written by the subprocesses for the final table
        for seed in seeds:
            summary_path = BASELINE_DIR / _run_dir_name(seed, CONSULTANCY_JUDGE_MODEL) / f"eval_summary_{_model_tag(CONSULTANCY_JUDGE_MODEL)}.json"
            if summary_path.exists():
                results[seed] = json.loads(summary_path.read_text())
    else:
        for seed in seeds:
            summary = _run_seed(seed, dry_run=args.dry_run, workers=args.workers)
            if summary is not None:
                results[seed] = summary

    if results and not args.dry_run:
        _print_summary_table(results)


if __name__ == "__main__":
    main()
