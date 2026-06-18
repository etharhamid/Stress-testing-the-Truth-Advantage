#!/usr/bin/env python
# steps/run_judge.py
# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Judge all debate transcripts and append results to
#         results/judgements.jsonl
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from tqdm import tqdm

# Allow running this file directly: python steps/run_judge.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.judge_engine import judge_debate
from core.gemini_client import check_model_available


def load_transcripts(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    records.append(rec)
            except Exception:
                pass
    return records


def already_judged(path: str) -> set[int]:
    """IDs with a successful judgement line (skip error rows so reruns retry failures)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                done.add(int(rec["id"]))
            except Exception:
                pass
    return done


def main():
    print("=" * 60)
    print("Step 3: Judging debates")
    print("=" * 60)
    config.validate_llm_credentials()
    print("  Judge: Google Gemini via Vertex AI (google.genai)")
    print(f"  GCP project: {config.GCP_PROJECT}")
    print(f"  GCP location: {config.GCP_LOCATION}")
    print(f"  Judge model: {config.JUDGE_MODEL}")

    if not os.path.exists(config.TRANSCRIPTS_FILE):
        print(f"  ERROR: {config.TRANSCRIPTS_FILE} not found. Run run_debates.py first.")
        sys.exit(1)

    if not check_model_available(config.JUDGE_MODEL, profile="judge"):
        print(
            f"  WARNING: Model '{config.JUDGE_MODEL}' not listed by the judge endpoint.\n"
            "  Check the model id and that the API endpoint is reachable with your key."
        )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    transcripts = load_transcripts(config.TRANSCRIPTS_FILE)
    done_ids    = already_judged(config.JUDGEMENTS_FILE)

    if done_ids:
        print(f"  Resuming — {len(done_ids)} already judged.")

    pending = [t for t in transcripts if int(t["id"]) not in done_ids]
    print(f"  {len(pending)} transcripts to judge.")

    with open(config.JUDGEMENTS_FILE, "a", encoding="utf-8") as out_f:
        for debate in tqdm(pending, desc="Judging", unit="q"):
            try:
                result = judge_debate(debate)
                rec = {
                    "id":                   result["id"],
                    "question":             result["question"],
                    "story_title":          result.get("story_title", ""),
                    "swap":                 result["swap"],
                    "answer_a":             result["answer_a"],
                    "answer_b":             result["answer_b"],
                    "correct_letter":       result["correct_letter"],
                    "judge_answer":         result["judge_answer"],
                    "judge_confidence":     result.get("judge_confidence"),
                    "judge_logprob_a":      result.get("judge_logprob_a"),
                    "judge_logprob_b":      result.get("judge_logprob_b"),
                    "judge_log_odds_truth": result.get("judge_log_odds_truth"),
                    "judge_logit_gap_truth":result.get("judge_logit_gap_truth"),
                    "judge_correct":        result["judge_correct"],
                    "judge_fallback":       result.get("judge_fallback", False),
                    "judge_raw":            result["judge_raw_response"],
                    "transcript_str":       result["transcript_str"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"\n  ERROR judging question {debate.get('id')}: {e}")
                err = {"id": debate.get("id"), "error": str(e)}
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\n  Judgements saved to {config.JUDGEMENTS_FILE}")


if __name__ == "__main__":
    main()
