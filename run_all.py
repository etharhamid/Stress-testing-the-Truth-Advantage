#!/usr/bin/env python
# run_all.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs pipeline steps in sequence. Each invocation uses a fresh folder
# under results/runs/<run_id>/ for artifacts and appends one row to
# results/experiments_results.csv.
#
# Usage:
#   python run_all.py                      # normal fresh pipeline
#   python run_all.py --rejudge-baseline   # re-judge qd_baseline/ transcripts
#                                          # with both flash + flash-lite judges
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _configure_experiment_run() -> tuple[str, Path]:
    """Create results/runs/<id>/ and point config outputs there. Returns (run_id, run_abs)."""
    run_id = _new_run_id()
    run_abs = ROOT / "results" / "runs" / run_id
    run_abs.mkdir(parents=True, exist_ok=True)
    config.set_run_output_directory(str(run_abs))
    return run_id, run_abs



def _compute_baseline_summary(judgements: list[dict], judge_model: str, random_seed: int) -> dict:
    from steps.score_results import _wilson_ci  # type: ignore[import-not-found]

    valid   = [j for j in judgements if j.get("judge_answer") is not None]
    correct = [j for j in valid if j.get("judge_correct") is True]
    n = len(valid)
    twr = len(correct) / n if n else 0.0
    ci_lo, ci_hi = _wilson_ci(len(correct), n)
    picked_a = sum(1 for j in valid if j.get("judge_answer") == "A")
    non_swap = [j for j in valid if not j.get("swap")]
    swapped  = [j for j in valid if     j.get("swap")]

    def _acc(subset: list[dict]) -> float:
        return sum(j.get("judge_correct", False) for j in subset) / len(subset) if subset else 0.0

    return {
        "judge_model":               judge_model,
        "random_seed":               random_seed,
        "total_questions":           len(judgements),
        "valid_judgements":          n,
        "invalid_judgements":        len(judgements) - n,
        "judge_correct":             len(correct),
        "judge_incorrect":           n - len(correct),
        "truth_win_rate":            round(twr, 4),
        "deceptive_win_rate":        round(1 - twr, 4),
        "asd":                       round(2 * twr - 1, 4),
        "base_rate_a":               round(picked_a / n, 4) if n else 0.0,
        "n_judge_a":                 picked_a,
        "n_judge_b":                 n - picked_a,
        "invalid_rate":              round((len(judgements) - n) / len(judgements), 4) if judgements else 0.0,
        "std_err":                   round(math.sqrt(twr * (1 - twr) / n), 4) if n else 0.0,
        "accuracy_95ci":             [round(ci_lo, 4), round(ci_hi, 4)],
        "accuracy_95ci_method":      "wilson",
        "position_adjusted_accuracy": round(0.5 * _acc(non_swap) + 0.5 * _acc(swapped), 4),
        "accuracy_non_swap":         round(_acc(non_swap), 4),
        "accuracy_swapped":          round(_acc(swapped), 4),
        "n_non_swap":                len(non_swap),
        "n_swapped":                 len(swapped),
    }


def _rejudge_baseline() -> None:
    """Re-judge all qd_baseline/base_run_{seed}/ transcripts with the QD eval judge.

    Uses the same judge call path as run_qd_eval (AI Studio, no logprobs, 30s retry).
    Writes qd_baseline/base_run_{seed}/eval_summary_{model_tag}.json per run per model.
    """
    config.validate_llm_credentials()

    from qd.fitness import run_eval_judge  # noqa: PLC0415

    BASELINE_DIR  = ROOT / "qd_baseline"
    JUDGE_MODELS  = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    run_dirs = sorted(BASELINE_DIR.glob("base_run_*/"))
    if not run_dirs:
        print("No base_run_{seed}/ directories found in qd_baseline/")
        return

    print(f"Re-judging {len(run_dirs)} baseline run(s) with {len(JUDGE_MODELS)} judge models\n")

    for judge_model in JUDGE_MODELS:
        model_tag = judge_model.replace("gemini-2.5-", "")
        print(f"\n{'='*60}")
        print(f"  Judge: {judge_model}")
        print(f"{'='*60}")

        config.QD_EVAL_JUDGE_MODEL = judge_model

        for run_dir in run_dirs:
            transcripts_path = run_dir / "transcripts.jsonl"
            if not transcripts_path.exists():
                print(f"  [{run_dir.name}] transcripts.jsonl not found — skipping")
                continue

            # Read per-run seed from metadata.json
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                run_seed = json.loads(metadata_path.read_text()).get("random_seed", config.RANDOM_SEED)
            else:
                run_seed = config.RANDOM_SEED
                print(f"  [{run_dir.name}] metadata.json not found — using default seed {run_seed}")

            transcripts = [
                json.loads(line)
                for line in transcripts_path.read_text().splitlines()
                if line.strip()
            ]
            print(f"\n  [{run_dir.name}] {len(transcripts)} transcripts  seed={run_seed}")

            saved_seed = config.RANDOM_SEED
            config.RANDOM_SEED = run_seed
            try:
                judgements: list[dict] = []
                for i, t in enumerate(transcripts):
                    print(f"    [{i + 1:>3}/{len(transcripts)}] id={t.get('id')} ... ", end="", flush=True)
                    try:
                        j = run_eval_judge(t)
                    except Exception as e:
                        print(f"FAIL ({e!r}) — retrying in 30s", end=" ", flush=True)
                        time.sleep(30)
                        try:
                            j = run_eval_judge(t)
                        except Exception as e2:
                            print(f"RETRY FAIL ({e2!r}) — skipping")
                            j = {**t, "judge_answer": None, "judge_correct": None}
                    print(f"→ {j.get('judge_answer')}  correct={j.get('judge_correct')}")
                    judgements.append(j)
            finally:
                config.RANDOM_SEED = saved_seed

            # Write judgements JSONL
            judgements_path = run_dir / f"judgements_{model_tag}.jsonl"
            with open(judgements_path, "w", encoding="utf-8") as jf:
                for j in judgements:
                    jf.write(json.dumps(j) + "\n")

            summary = _compute_baseline_summary(judgements, judge_model, run_seed)
            out_path = run_dir / f"eval_summary_{model_tag}.json"
            out_path.write_text(json.dumps(summary, indent=2))
            print(
                f"\n  [{run_dir.name}] {judgements_path.name} + {out_path.name} written — "
                f"TWR={summary['truth_win_rate']:.4f} ({summary['judge_correct']}/{summary['valid_judgements']} valid)"
            )

    print("\nBaseline re-judgment complete.")


def _new_baseline_run(seeds: list[int]) -> None:
    """Run full debate pipeline + re-judgment for new baseline runs with given seeds.

    For each seed, creates qd_baseline/base_run_{seed}/, runs prepare_data
    and run_debates (gemini-3.1-pro-preview debaters), then judges with both
    gemini-2.5-flash and gemini-2.5-flash-lite. The directory is named
    after the seed so the mapping is obvious from a `ls` and
    `SEED_TO_BASELINE` stays trivially correct without a sequence-counter
    look-up.
    """
    config.validate_llm_credentials()

    from steps import prepare_data, run_debates  # noqa: PLC0415
    from qd.fitness import run_eval_judge        # noqa: PLC0415

    BASELINE_DIR = ROOT / "qd_baseline"
    JUDGE_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    for seed in seeds:
        run_name = f"base_run_{seed}"
        run_dir  = BASELINE_DIR / run_name
        if run_dir.exists():
            raise SystemExit(
                f"[baseline] {run_dir} already exists — refusing to "
                "overwrite. Delete or move it first if a re-run is intended."
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  New baseline run: {run_name}  seed={seed}")
        print(f"{'='*60}")

        # Write metadata
        (run_dir / "metadata.json").write_text(
            json.dumps({"run": run_name, "random_seed": seed}, indent=2)
        )

        # Point all pipeline paths at the new run directory
        saved_seed = config.RANDOM_SEED
        config.RANDOM_SEED = seed
        config.set_run_output_directory(str(run_dir))

        try:
            print(f"\n  >>> STEP 1: Prepare data (seed={seed})")
            prepare_data.main()

            print(f"\n  >>> STEP 2: Run debates")
            run_debates.main()
        finally:
            config.RANDOM_SEED = saved_seed

        # Re-judge with both eval judge models
        transcripts_path = run_dir / "transcripts.jsonl"
        transcripts = [
            json.loads(line)
            for line in transcripts_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"\n  >>> STEP 3: Judge ({len(transcripts)} transcripts)")

        for judge_model in JUDGE_MODELS:
            model_tag = judge_model.replace("gemini-2.5-", "")
            print(f"\n    Judge: {judge_model}")
            config.QD_EVAL_JUDGE_MODEL = judge_model

            config.RANDOM_SEED = seed
            try:
                judgements: list[dict] = []
                for i, t in enumerate(transcripts):
                    print(f"      [{i + 1:>3}/{len(transcripts)}] id={t.get('id')} ... ", end="", flush=True)
                    try:
                        j = run_eval_judge(t)
                    except Exception as e:
                        print(f"FAIL ({e!r}) — retrying in 30s", end=" ", flush=True)
                        time.sleep(30)
                        try:
                            j = run_eval_judge(t)
                        except Exception as e2:
                            print(f"RETRY FAIL ({e2!r}) — skipping")
                            j = {**t, "judge_answer": None, "judge_correct": None}
                    print(f"→ {j.get('judge_answer')}  correct={j.get('judge_correct')}")
                    judgements.append(j)
            finally:
                config.RANDOM_SEED = saved_seed

            judgements_path = run_dir / f"judgements_{model_tag}.jsonl"
            with open(judgements_path, "w", encoding="utf-8") as jf:
                for j in judgements:
                    jf.write(json.dumps(j) + "\n")

            summary = _compute_baseline_summary(judgements, judge_model, seed)
            summary_path = run_dir / f"eval_summary_{model_tag}.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(
                f"\n    {run_name}/{model_tag}: TWR={summary['truth_win_rate']:.4f} "
                f"({summary['valid_judgements']}/{summary['total_questions']} valid)"
            )

        print(f"\n  {run_name} complete.")

    print("\nAll new baseline runs complete.")


def _retry_debate_failures() -> None:
    """Re-debate any error records in qd_baseline/base_run_{seed}/transcripts.jsonl.

    run_debates.main() already skips successful records, so pointing config at
    a run directory and calling it will only re-debate the failed questions.
    After re-debating, rewrites transcripts.jsonl to deduplicate (error record
    replaced by the new successful one) and keeps stable id order.
    """
    config.validate_llm_credentials()

    from steps import run_debates  # noqa: PLC0415

    BASELINE_DIR = ROOT / "qd_baseline"
    run_dirs = sorted(BASELINE_DIR.glob("base_run_*/"))

    if not run_dirs:
        print("No base_run_{seed}/ directories found in qd_baseline/")
        return

    # Save original config paths to restore after each run
    saved_seed    = config.RANDOM_SEED
    saved_results = config.RESULTS_DIR
    saved_questions  = config.QUESTIONS_FILE
    saved_transcripts = config.TRANSCRIPTS_FILE
    saved_judgements  = config.JUDGEMENTS_FILE

    found_any = False
    for run_dir in run_dirs:
        transcripts_path = run_dir / "transcripts.jsonl"
        if not transcripts_path.exists():
            continue

        records = [
            json.loads(line)
            for line in transcripts_path.read_text().splitlines()
            if line.strip()
        ]
        error_records = [r for r in records if r.get("error")]
        if not error_records:
            print(f"  [{run_dir.name}] no debate errors — skipping")
            continue

        found_any = True
        error_ids = [r.get("id") for r in error_records]
        print(f"\n[{run_dir.name}] {len(error_records)} error record(s): ids={error_ids}")

        metadata_path = run_dir / "metadata.json"
        run_seed = (
            json.loads(metadata_path.read_text()).get("random_seed", config.RANDOM_SEED)
            if metadata_path.exists() else config.RANDOM_SEED
        )

        config.RANDOM_SEED = run_seed
        config.set_run_output_directory(str(run_dir))
        try:
            run_debates.main()
        finally:
            # Restore config
            config.RANDOM_SEED      = saved_seed
            config.RESULTS_DIR      = saved_results
            config.QUESTIONS_FILE   = saved_questions
            config.TRANSCRIPTS_FILE = saved_transcripts
            config.JUDGEMENTS_FILE  = saved_judgements

        # Deduplicate: for each id keep the non-error record if one now exists
        fresh = [
            json.loads(line)
            for line in transcripts_path.read_text().splitlines()
            if line.strip()
        ]
        best: dict[str, dict] = {}
        for r in fresh:
            rid = str(r.get("id"))
            prev = best.get(rid)
            if prev is None or (prev.get("error") and not r.get("error")):
                best[rid] = r

        deduped = sorted(best.values(), key=lambda r: int(r.get("id", 0)))
        with open(transcripts_path, "w", encoding="utf-8") as f:
            for r in deduped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        still_errors = sum(1 for r in deduped if r.get("error"))
        print(
            f"  [{run_dir.name}] transcripts rewritten — "
            f"{len(deduped)} records, {still_errors} error(s) remaining"
        )

    if not found_any:
        print("No error debate records found across all runs.")

    print("\nDebate retry complete.")


def _retry_baseline_failures() -> None:
    """Retry any judge_answer=None records across all qd_baseline/base_run_{seed}/ runs.

    Reads existing judgements_{model}.jsonl files, re-judges only the failed
    records, overwrites the file, and rewrites eval_summary_{model}.json.
    Safe to run multiple times — skips runs/models with no failures.
    """
    config.validate_llm_credentials()

    from qd.fitness import run_eval_judge  # noqa: PLC0415

    BASELINE_DIR = ROOT / "qd_baseline"
    JUDGE_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    run_dirs = sorted(BASELINE_DIR.glob("base_run_*/"))
    if not run_dirs:
        print("No base_run_{seed}/ directories found in qd_baseline/")
        return

    for judge_model in JUDGE_MODELS:
        model_tag = judge_model.replace("gemini-2.5-", "")
        config.QD_EVAL_JUDGE_MODEL = judge_model

        for run_dir in run_dirs:
            judgements_path = run_dir / f"judgements_{model_tag}.jsonl"
            if not judgements_path.exists():
                print(f"  [{run_dir.name}/{model_tag}] no judgements file — run --rejudge-baseline first")
                continue

            judgements = [
                json.loads(line)
                for line in judgements_path.read_text().splitlines()
                if line.strip()
            ]
            failed = [j for j in judgements if j.get("judge_answer") is None]

            if not failed:
                print(f"  [{run_dir.name}/{model_tag}] no failures — skipping")
                continue

            print(f"\n  [{run_dir.name}/{model_tag}] {len(failed)} failed record(s) to retry")

            metadata_path = run_dir / "metadata.json"
            run_seed = (
                json.loads(metadata_path.read_text()).get("random_seed", config.RANDOM_SEED)
                if metadata_path.exists() else config.RANDOM_SEED
            )

            # Build transcript index so we can fall back to the full transcript when
            # the judgement record is a bare error stub (missing transcript_str etc.)
            transcripts_path = run_dir / "transcripts.jsonl"
            transcript_index: dict[str, dict] = {}
            if transcripts_path.exists():
                for line in transcripts_path.read_text().splitlines():
                    if line.strip():
                        tr = json.loads(line)
                        transcript_index[str(tr.get("id"))] = tr

            saved_seed = config.RANDOM_SEED
            config.RANDOM_SEED = run_seed
            try:
                for j in failed:
                    qid = j.get("id")
                    # Use full transcript if judgement record is a bare error stub
                    source = j if j.get("transcript_str") else transcript_index.get(str(qid), j)
                    print(f"    retrying id={qid} ... ", end="", flush=True)
                    try:
                        result = run_eval_judge(source)
                    except Exception as e:
                        print(f"FAIL ({e!r}) — retrying in 30s", end=" ", flush=True)
                        time.sleep(30)
                        try:
                            result = run_eval_judge(source)
                        except Exception as e2:
                            print(f"RETRY FAIL ({e2!r}) — still None")
                            continue
                    print(f"→ {result.get('judge_answer')}  correct={result.get('judge_correct')}")
                    # Update in-place by matching id
                    for idx, jj in enumerate(judgements):
                        if jj.get("id") == qid:
                            judgements[idx] = result
                            break
            finally:
                config.RANDOM_SEED = saved_seed

            # Overwrite judgements file
            with open(judgements_path, "w", encoding="utf-8") as jf:
                for j in judgements:
                    jf.write(json.dumps(j) + "\n")

            # Rewrite summary
            summary = _compute_baseline_summary(judgements, judge_model, run_seed)
            summary_path = run_dir / f"eval_summary_{model_tag}.json"
            summary_path.write_text(json.dumps(summary, indent=2))

            still_failed = sum(1 for j in judgements if j.get("judge_answer") is None)
            print(
                f"  [{run_dir.name}/{model_tag}] done — "
                f"TWR={summary['truth_win_rate']:.4f} "
                f"({summary['valid_judgements']}/{summary['total_questions']} valid, "
                f"{still_failed} still None)"
            )

    print("\nRetry complete.")


def _generate_run_plots(run_dir: Path) -> None:
    """Write ``<run_dir>/plots/*.png`` via ``results/plot_results.py`` (single-run only)."""
    plot_script = ROOT / "results" / "plot_results.py"
    spec = importlib.util.spec_from_file_location("baseline_plot_results", plot_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plotting module: {plot_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.plot_single_run(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline debate pipeline")
    parser.add_argument(
        "--rejudge-baseline",
        action="store_true",
        help="Re-judge qd_baseline/ transcripts with the QD eval judge (both flash models). "
             "Skips the normal debate pipeline.",
    )
    parser.add_argument(
        "--retry-debate-failures",
        action="store_true",
        help="Re-debate any error records in qd_baseline/ transcripts.jsonl files. "
             "Run before --retry-failures when debates themselves failed.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry any judge_answer=None records in existing qd_baseline/ judgement files. "
             "Run after --rejudge-baseline to fill in API failures.",
    )
    parser.add_argument(
        "--new-baseline-run",
        metavar="SEED",
        type=int,
        nargs="+",
        help="Run full debate pipeline + re-judgment for new qd_baseline runs. "
             "Pass one or more seeds, e.g. --new-baseline-run 13 73",
    )
    args = parser.parse_args()

    if args.rejudge_baseline:
        _rejudge_baseline()
        sys.exit(0)

    if args.retry_debate_failures:
        _retry_debate_failures()
        sys.exit(0)

    if args.retry_failures:
        _retry_baseline_failures()
        sys.exit(0)

    if args.new_baseline_run:
        _new_baseline_run(args.new_baseline_run)
        sys.exit(0)

    run_id, run_abs = _configure_experiment_run()
    run_dir_posix = run_abs.relative_to(ROOT).as_posix()
    started_at = datetime.now(timezone.utc).isoformat()
    failed_step = ""
    err_message = ""
    status = "success"
    exc: BaseException | None = None

    print(f"\nExperiment output directory: {run_dir_posix}\n")

    from steps import prepare_data, run_debates, run_judge, score_results

    step = "prepare_data"
    try:
        print(">>> STEP 1: Prepare data\n")
        prepare_data.main()

        step = "run_debates"
        print("\n>>> STEP 2: Run debates\n")
        run_debates.main()

        step = "run_judge"
        print("\n>>> STEP 3: Judge debates\n")
        run_judge.main()

        step = "score_results"
        print("\n>>> STEP 4: Score results\n")
        score_results.main()

        step = "plot_results"
        print("\n>>> STEP 5: Publication plots\n")
        try:
            _generate_run_plots(run_abs)
            rel = (run_abs / "plots").relative_to(ROOT)
            print(f"  Figures → {rel.as_posix()}/")
        except BaseException as plot_exc:
            print(
                f"\nWarning: plotting failed ({type(plot_exc).__name__}: {plot_exc}); "
                "scores and registry are still written. Regenerate with:\n"
                f"  python results/plot_results.py --run-dir {run_dir_posix}\n",
                file=sys.stderr,
            )
            traceback.print_exc()

        print("\nDone!")
    except BaseException as e:
        exc = e
        err_message = f"{type(e).__name__}: {e}"
        failed_step = step
        if isinstance(e, KeyboardInterrupt):
            status = "interrupted"
            print("\nInterrupted.", file=sys.stderr)
        else:
            status = "error"
            print(f"\nRun stopped with error at step '{step}': {e}", file=sys.stderr)
            traceback.print_exc()
    finally:
        pass

    if exc is not None:
        raise exc
