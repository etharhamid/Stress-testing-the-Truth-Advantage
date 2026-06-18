"""Warm-start initialization for MAP-Elites.

Seeds every empty archive cell with one candidate before the main loop starts.
Only runs on fresh archives (iter_count == 0, no occupied cells). All logged
records use iter=0 to distinguish them from main-loop records (iter ≥ 1).
"""
from __future__ import annotations

import random
import traceback
from typing import Callable

import config

from .archive import Archive, Elite, QUESTION_TYPES, MUTATION_TYPES
from .bleu import passes_filter
from .fitness import fitness_from_judgement, run_search_debate, run_search_judge
from .logging import JsonlWriter, make_failure_record
from .mutator import call_mutator
from .validator import call_validator, classify_question_type


def _cell_index(qtype: str, mtype: str) -> int:
    return QUESTION_TYPES.index(qtype) * len(MUTATION_TYPES) + MUTATION_TYPES.index(mtype)


def _pick_pool_row(pool: list[dict], target_qtype: str, rng: random.Random) -> dict:
    typed = [r for r in pool if r.get("question_type") == target_qtype]
    return dict(rng.choice(typed if typed else pool))


def warm_start_archive(
    archive: Archive,
    pool: list[dict],
    qd,
    transcripts_writer: JsonlWriter,
    judgements_writer: JsonlWriter,
    failures_writer: JsonlWriter,
    swap_fn: Callable[[str, int], bool],
) -> tuple[int, int]:
    """Try to seed every empty cell once. Returns (successes, accepted).

    Uses iter=0 for all logged records. Does not increment archive.iter_count.
    """
    successes = 0
    accepted = 0

    cells = [(qt, mt) for qt in QUESTION_TYPES for mt in MUTATION_TYPES]

    for qt, mt in cells:
        if archive.get((qt, mt)) is not None:
            continue  # already occupied (shouldn't happen on fresh archive)

        print(f"[QD-init] seeding cell ({qt}, {mt}) ...")
        cidx = _cell_index(qt, mt)
        cell_rng = random.Random(qd.seed + cidx)
        parent_row = _pick_pool_row(pool, qt, cell_rng)
        parent_qid = str(parent_row["id"])

        mutated: dict | None = None
        bleu_score_val: float | None = None
        stage_failures: list[tuple[str, str, str | None, str | None]] = []

        for attempt in range(qd.max_retries):
            seed = qd.seed + cidx * 31 + attempt
            m = call_mutator(
                parent_question=parent_row["question"],
                correct_answer=parent_row["correct_answer"],
                parent_distractor=parent_row["negative_answer"],
                mutation_type=mt,
                target_question_type=qt,
                seed=seed,
            )
            if m is None:
                stage_failures.append(("mutator_parse", f"attempt={attempt}", None, None))
                continue

            parent_combined = parent_row["question"] + " " + parent_row["negative_answer"]
            mutated_combined = m["mutated_question"] + " " + m["mutated_distractor"]
            bleu_score_val, ok = passes_filter(parent_combined, mutated_combined, qd.bleu_threshold)
            if not ok:
                stage_failures.append(("bleu_filter",
                                       f"attempt={attempt} score={bleu_score_val:.3f}",
                                       m["mutated_question"], m["mutated_distractor"]))
                continue

            detected_type = classify_question_type(m["mutated_question"], seed=seed, target_type=qt)
            if detected_type != qt:
                stage_failures.append((
                    "descriptor_mismatch",
                    f"attempt={attempt} expected={qt} got={detected_type}",
                    m["mutated_question"], m["mutated_distractor"],
                ))
                continue

            valid = call_validator(
                story=parent_row["story"],
                original_question=parent_row["question"],
                correct_answer=parent_row["correct_answer"],
                distractor=parent_row["negative_answer"],
                mutated_question=m["mutated_question"],
                mutated_distractor=m["mutated_distractor"],
                mutation_type=mt,
                seed=seed,
            )
            if not valid:
                stage_failures.append(("validity", f"attempt={attempt} bleu={bleu_score_val:.3f}",
                                       m["mutated_question"], m["mutated_distractor"]))
                continue

            mutated = m
            break

        if mutated is None:
            for stage, detail, mq, md in stage_failures:
                failures_writer.write_record(make_failure_record(
                    iter_no=0, qid=parent_qid,
                    target_cell=(qt, mt), stage=stage, detail=detail,
                    mutated_question=mq, mutated_distractor=md,
                ))
            continue

        candidate_row = dict(parent_row)
        candidate_row["question"] = mutated["mutated_question"]
        candidate_row["negative_answer"] = mutated["mutated_distractor"]
        swap = swap_fn(str(candidate_row["id"]), 0)

        try:
            transcript = run_search_debate(candidate_row, swap)
            judgement = run_search_judge(transcript)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            failures_writer.write_record(make_failure_record(
                iter_no=0, qid=parent_qid,
                target_cell=(qt, mt), stage="debate_or_judge",
                detail=f"{e!r}\n{tb}",
            ))
            continue

        transcripts_writer.write_record({
            **transcript,
            "iter": 0,
            "target_cell": [qt, mt],
            "parent_qid": parent_qid,
            "bleu_to_parent": bleu_score_val,
        })
        j_log = dict(judgement)
        j_log.pop("rounds", None)
        j_log.pop("transcript_str", None)
        judgements_writer.write_record({
            **j_log,
            "iter": 0,
            "target_cell": [qt, mt],
            "parent_qid": parent_qid,
            "bleu_to_parent": bleu_score_val,
        })

        fit = fitness_from_judgement(judgement)
        if fit is not None:
            successes += 1
            candidate = Elite(
                qid=str(candidate_row["id"]),
                parent_qid=parent_qid,
                mutated_question=mutated["mutated_question"],
                mutated_distractor=mutated["mutated_distractor"],
                correct_answer=candidate_row["correct_answer"],
                fitness=fit,
                iteration=0,
                bleu_to_parent=float(bleu_score_val or 0.0),
                swap=swap,
                judge_log_odds_truth=judgement.get("judge_log_odds_truth"),
                judge_correct=judgement.get("judge_correct"),
            )
            if archive.try_update((qt, mt), candidate):
                accepted += 1

    archive.save(qd.archive_path)
    return successes, accepted
