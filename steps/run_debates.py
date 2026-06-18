#!/usr/bin/env python
# steps/run_debates.py
# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run debates for all questions and save transcripts to
#         results/transcripts.jsonl
#
# Positional-bias mitigation:
#   For each question we randomly decide whether to place the correct answer
#   at position A or B (swap=True/False).  This is seeded so results are
#   reproducible.  We do NOT run both orderings per question — just one random
#   assignment, which is the simplest unbiased approach.
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import os
import random
import sys
from tqdm import tqdm

# Allow running this file directly: python steps/run_debates.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.debate_engine import run_debate
from core.gemini_client import check_model_available


def load_questions(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(transcripts_file: str) -> dict[int, bool]:
    """Return {qid: swap} for questions with a successful transcript (not error rows)."""
    done: dict[int, bool] = {}
    if not os.path.exists(transcripts_file):
        return done
    with open(transcripts_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                done[int(rec["id"])] = bool(rec.get("swap", False))
            except Exception:
                pass
    return done


def main():
    print("=" * 60)
    print("Step 2: Running debates")
    print("=" * 60)
    config.validate_llm_credentials()
    print("  Debater: Google Gemini via Vertex AI (google.genai)")
    print(f"  GCP project: {config.GCP_PROJECT}")
    print(f"  GCP location: {config.GCP_LOCATION}")
    print(f"  Debater model: {config.DEBATER_MODEL}")

    if not os.path.exists(config.QUESTIONS_FILE):
        print(f"  ERROR: {config.QUESTIONS_FILE} not found. Run prepare_data.py first.")
        sys.exit(1)

    if not check_model_available(config.DEBATER_MODEL, profile="debater"):
        print(
            f"  WARNING: Model '{config.DEBATER_MODEL}' not listed by the debater endpoint.\n"
            "  Check the model id and that the API endpoint is reachable with your key."
        )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    questions = load_questions(config.QUESTIONS_FILE)
    done_ids  = already_done(config.TRANSCRIPTS_FILE)

    if done_ids:
        print(f"  Resuming — {len(done_ids)} already completed.")

    # ── Assign swap flags with guaranteed 50/50 balance ─────────────────
    # Half the questions get swap=True (truth=B), half get swap=False (truth=A).
    # This eliminates positional bias from skewing aggregate accuracy.
    rng = random.Random(config.RANDOM_SEED)
    n = len(questions)
    balanced = [False] * (n // 2) + [True] * (n - n // 2)
    rng.shuffle(balanced)
    swap_flags = {int(questions[i]["id"]): balanced[i] for i in range(n)}

    # Sanity check: on resume, the recomputed schedule must agree with whatever
    # was written to transcripts.jsonl on the previous run. Divergence means
    # questions.csv was regenerated with a different seed / NUM_QUESTIONS / order
    # and resuming would silently mix incompatible assignments.
    mismatches = []
    for qid, prev_swap in done_ids.items():
        cur_swap = swap_flags.get(qid)
        if cur_swap is None:
            print(
                f"  WARNING: completed question {qid} is missing from the current "
                f"questions.csv (csv may have been regenerated)."
            )
        elif cur_swap != prev_swap:
            mismatches.append((qid, prev_swap, cur_swap))
    if mismatches:
        print(
            f"  ERROR: swap-flag schedule changed since the previous run for "
            f"{len(mismatches)} completed question(s); first few: "
            f"{mismatches[:3]}. Delete {config.TRANSCRIPTS_FILE} before resuming, "
            f"or restore the original questions.csv."
        )
        sys.exit(1)

    pending = [q for q in questions if int(q["id"]) not in done_ids]
    print(f"  {len(pending)} questions to debate.")

    # ── Run debates ───────────────────────────────────────────────────────
    with open(config.TRANSCRIPTS_FILE, "a", encoding="utf-8") as out_f:
        for row in tqdm(pending, desc="Debates", unit="q"):
            qid  = int(row["id"])
            swap = swap_flags[qid]

            try:
                result = run_debate(row, swap=swap)
                result_to_save = {k: v for k, v in result.items() if k != "story"}
                out_f.write(json.dumps(result_to_save, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"\n  ERROR on question {qid}: {e}")
                err_rec = {
                    "id": qid,
                    "error": str(e),
                    "swap": swap,
                    "question": row["question"],
                }
                out_f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\n  Transcripts saved to {config.TRANSCRIPTS_FILE}")


if __name__ == "__main__":
    main()
