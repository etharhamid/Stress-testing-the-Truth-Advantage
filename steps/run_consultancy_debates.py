#!/usr/bin/env python
# steps/run_consultancy_debates.py
# ─────────────────────────────────────────────────────────────────────────────
# Run consultancy sessions for all questions and save transcripts to
# consultancy_transcripts.jsonl.
#
# Designed to be called directly by run_consultancy_baseline.py with explicit
# file paths (no global config mutations — multiple seeds run in the same
# process without stepping on each other).
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from core.consultancy_engine import run_consultancy


def already_done(transcripts_path: str) -> dict[int, bool]:
    """Return {qid: swap} for questions with a successful transcript (not error rows)."""
    done: dict[int, bool] = {}
    if not os.path.exists(transcripts_path):
        return done
    with open(transcripts_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                done[int(rec["id"])] = bool(rec.get("swap", False))
            except Exception:
                pass
    return done


def run(
    questions: list[dict],
    transcripts_path: str,
    swap_flags: dict[int, bool],
    model: str | None = None,
    judge_question_model: str | None = None,
    workers: int = 1,
) -> None:
    """
    Run all consultancy sessions and write to transcripts_path (append-only).

    Skips questions already present in transcripts_path (resume-safe).
    Error records are written per question so a crash mid-run loses at most
    one session.

    Args:
        questions:            List of question dicts (from questions.csv).
        transcripts_path:     Path to the output JSONL file.
        swap_flags:           {qid: swap} mapping pre-computed by the entry point.
        model:                Consultant model override (default: config.DEBATER_MODEL).
        judge_question_model: In-round judge model override
                              (default: gemini-2.5-flash-lite).
        workers:              Number of parallel consultancy sessions (default 1).
    """
    os.makedirs(os.path.dirname(os.path.abspath(transcripts_path)), exist_ok=True)

    done_ids = already_done(transcripts_path)
    if done_ids:
        print(f"  Resuming — {len(done_ids)} already completed.")

    # Guard: on resume, verify that the saved swap for each completed question
    # matches the current swap_flags schedule.  A mismatch means questions.csv
    # or the seed changed since the previous run; proceeding would silently mix
    # incompatible positional assignments.
    mismatches = []
    for qid, saved_swap in done_ids.items():
        cur_swap = swap_flags.get(qid)
        if cur_swap is None:
            print(
                f"  WARNING: completed question {qid} is missing from the "
                f"current swap_flags (questions.csv may have changed)."
            )
        elif cur_swap != saved_swap:
            mismatches.append((qid, saved_swap, cur_swap))
    if mismatches:
        print(
            f"  ERROR: swap-flag schedule changed since the previous run for "
            f"{len(mismatches)} completed question(s); first few: "
            f"{mismatches[:3]}. Delete {transcripts_path} before resuming, "
            f"or restore the original questions.csv and seed."
        )
        sys.exit(1)

    pending = [q for q in questions if int(q["id"]) not in done_ids]
    print(f"  {len(pending)} questions to run  (workers={workers}).")

    write_lock = threading.Lock()

    def _run_one(row: dict) -> None:
        qid  = int(row["id"])
        swap = swap_flags[qid]
        try:
            result = run_consultancy(
                row,
                swap=swap,
                model=model,
                judge_question_model=judge_question_model,
            )
            result_to_save = {k: v for k, v in result.items() if k != "story"}
            line = json.dumps(result_to_save, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"\n  ERROR on question {qid}: {e}")
            line = json.dumps({
                "id": qid, "error": str(e),
                "swap": swap, "question": row["question"],
            }, ensure_ascii=False) + "\n"
        with write_lock:
            out_f.write(line)
            out_f.flush()

    with open(transcripts_path, "a", encoding="utf-8") as out_f:
        if workers <= 1:
            for row in tqdm(pending, desc="Consultancy", unit="q"):
                _run_one(row)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_run_one, row): row for row in pending}
                for _ in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="Consultancy", unit="q",
                ):
                    pass

    print(f"  Transcripts saved to {transcripts_path}")
