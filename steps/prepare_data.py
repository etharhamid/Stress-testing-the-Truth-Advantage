#!/usr/bin/env python
# steps/prepare_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download the QuALITY dataset, apply all filters from the paper,
#         and save 100 questions to results/questions.csv
# ─────────────────────────────────────────────────────────────────────────────

import csv
import os
import sys


def _refresh_if_stale(path: str, expected_rows: int) -> None:
    """Delete questions CSV when its row count doesn't match expected_rows."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, newline="", encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
    except Exception:
        return
    if n != expected_rows:
        os.remove(path)
        print(f"  Removed stale {path} ({n} rows ≠ NUM_QUESTIONS={expected_rows}).\n")

# Allow running this file directly: python steps/prepare_data.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.quality_loader import load_questions


def main():
    print("=" * 60)
    print("Step 1: Preparing QuALITY dataset")
    print("=" * 60)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    _refresh_if_stale(config.QUESTIONS_FILE, config.NUM_QUESTIONS)

    if os.path.exists(config.QUESTIONS_FILE):
        print(f"  {config.QUESTIONS_FILE} already exists — skipping download.")
        print("  Delete it to re-download and re-filter.")
        return

    print(f"  Filters: source={config.SOURCES}, difficulty={config.DIFFICULTY}, "
          f"max_answerability={config.MAX_ANSWERABILITY}, "
          f"min_untimed_acc={config.MIN_UNTIMED_ACCURACY}, "
          f"max_speed_acc={config.MAX_SPEED_ACCURACY}, "
          f"min_context={config.MIN_CONTEXT_REQUIRED}, "
          f"skip_conflicting={config.SKIP_CONFLICTING}, "
          f"ignore_nyu={config.IGNORE_NYU}")

    questions = load_questions(
        splits=["train", "dev"],
        sources=config.SOURCES,
        difficulty=config.DIFFICULTY,
        max_answerability=config.MAX_ANSWERABILITY,
        min_untimed_accuracy=config.MIN_UNTIMED_ACCURACY,
        max_speed_accuracy=config.MAX_SPEED_ACCURACY,
        min_context_required=config.MIN_CONTEXT_REQUIRED,
        skip_conflicting=config.SKIP_CONFLICTING,
        ignore_nyu=config.IGNORE_NYU,
        max_from_same_story=config.MAX_FROM_SAME_STORY,
        limit=config.NUM_QUESTIONS,
        seed=config.RANDOM_SEED,
    )

    if not questions:
        print("  ERROR: No questions passed the filter. Check your config.")
        sys.exit(1)

    fieldnames = [
        "id", "question", "correct_answer", "negative_answer",
        "story", "story_title", "question_set_id", "split",
    ]

    with open(config.QUESTIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)

    print(f"  Saved {len(questions)} questions to {config.QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
