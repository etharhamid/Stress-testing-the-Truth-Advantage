#!/usr/bin/env python
# steps/run_judge_consultancy.py
# ─────────────────────────────────────────────────────────────────────────────
# Judge all consultancy transcripts and write results to judgements JSONL.
#
# One set of transcripts (consultancy_transcripts.jsonl) can be judged by
# multiple models independently — same pattern as the debate baseline's
# run_judge.py passing over the same transcripts.jsonl.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from core.consultancy_judge_engine import judge_consultancy


def load_transcripts(transcripts_path: str) -> list[dict]:
    """Load non-error transcript records from a JSONL file."""
    records = []
    with open(transcripts_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "error" not in rec:
                    records.append(rec)
            except Exception:
                pass
    return records


def already_judged(judgements_path: str) -> set[int]:
    """IDs with a successful judgement line (error rows are not skipped so
    reruns can retry partial failures)."""
    done: set[int] = set()
    if not os.path.exists(judgements_path):
        return done
    with open(judgements_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                done.add(int(rec["id"]))
            except Exception:
                pass
    return done


def run(
    transcripts_path: str,
    judgements_path: str,
    model: str,
) -> None:
    """
    Judge all transcripts in transcripts_path and write to judgements_path.

    Skips already-judged IDs (resume-safe). Writes one JSON record per
    question, flushed immediately so a crash loses at most one judgement.

    The output schema matches judge_engine.judge_debate() — the same fields
    expected by steps/score_results.py — plus the consultancy-specific
    `consultant_correct` field.

    Args:
        transcripts_path: Path to consultancy_transcripts.jsonl.
        judgements_path:  Path to the output judgements JSONL file.
        model:            Judge model to use (e.g. "gemini-2.5-flash-lite").
    """
    if not os.path.exists(transcripts_path):
        print(f"  ERROR: {transcripts_path} not found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(judgements_path)), exist_ok=True)

    transcripts = load_transcripts(transcripts_path)
    done_ids    = already_judged(judgements_path)

    if done_ids:
        print(f"  Resuming — {len(done_ids)} already judged.")

    pending = [t for t in transcripts if int(t["id"]) not in done_ids]
    print(f"  {len(pending)} transcripts to judge.")

    with open(judgements_path, "a", encoding="utf-8") as out_f:
        for consultancy in tqdm(pending, desc=f"Judging [{model}]", unit="q"):
            try:
                result = judge_consultancy(consultancy, model=model)
                rec = {
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
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"\n  ERROR judging question {consultancy.get('id')}: {e}")
                err = {"id": consultancy.get("id"), "error": str(e)}
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"  Judgements saved to {judgements_path}")
