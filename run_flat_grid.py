#!/usr/bin/env python
"""Flat per-qid mutation grid — iterative stepping-stone, resumable.

For every qid in the seed's pool, runs an outer iteration loop. In each
iteration, every (qid, mutation_type) slot is given a fixed 3-attempt
budget to produce one valid mutated framing. The parent fed to the
mutator is either the cell's current best framing (stepping-stone) or
the original pool question (cold start on iteration 1, or whenever the
cell still has no valid candidate).

Unlike MAP-Elites this does NOT use the 5×4 archive or cell selection —
it is a per-qid fairness sweep designed so every pool question gets
the same iterative refinement under all four mutation operators.

Pool:
  * Default: qd_baseline/base_run_{seed}/both_correct.csv (the per-seed
    both-correct subset — 62/76/64/69/69 rows for seeds 42/7/13/73/123).
    Resolved via qd.config.pool_csv_for_seed(seed). Every row carries a
    hand-labelled `question_type` so the mutator and descriptor check
    receive the correct type and step 1 of the mutator (type adaptation)
    is a no-op.
  * --pool-source PATH overrides path resolution and accepts any CSV
    (e.g. the legacy 100-row questions.csv).

Output:
    qd_results/seed_{seed}/flat_grid_run_{NNN}/
        flat_grid_state.json        # iteration-boundary resume state
        flat_per_mutation.csv       # pool_size × 4 rows (one per qid/mtype)
        flat_best_per_qid.csv       # pool_size rows (one per qid)
        search_flat/
            flat_transcripts.jsonl  # full transcripts (rounds + transcript_str)
            flat_attempts.jsonl     # judgement + per-attempt metadata
            flat_failures.jsonl     # mutator/BLEU/descriptor/validity failures
            flat_grid_report.md     # human-readable summary

Resume:
  Auto-detect. If the latest flat_grid_run_NNN under qd_results/seed_N/ has
  a flat_grid_state.json that is incomplete, the script resumes into that
  directory: it replays flat_attempts.jsonl to rebuild cell bests, and
  skips any (qid, mtype) slot in the in-progress iteration that already
  has a record at that iteration. Pass --force-new to start a fresh run.

────────────────────────────────────────────────────────────────────────────
Parallel execution & thread-safety contract
────────────────────────────────────────────────────────────────────────────
Within each iteration, slots are dispatched in parallel to a
ThreadPoolExecutor with `--workers` threads (default 8). Within a slot,
the 3-attempt retry loop is strictly sequential — attempt k+1 only runs
if attempt k failed. Iteration boundaries are full synchronisation
points: the outer loop waits for all slot futures to complete before
writing CSVs, the report, and the state file.

Locking discipline:
  * JsonlWriter has an internal per-instance lock — workers may call
    `write_record` without an external lock.
  * Models: pass the QD search models via the `model=` kwarg of
    `run_search_debate` / `run_search_judge`. Workers MUST NOT call
    `qd.fitness._temporarily_set_models` — it mutates module globals.
  * `_FlatState` is guarded by `state.lock`. All read-compare-write of
    per-cell bests and global qid bests happens inside the lock.
  * Per-task seed: SHA-256 of `(RANDOM_SEED, qid, mutation_type,
    iteration, attempt_idx)`. Used for the mutator/classifier/validator
    seeds so retries are reproducible.
  * Swap flag: loaded once from `baseline_swap_for_seed(seed)` and
    locked per QID for every iteration, every attempt, and every
    re-run. This matches `run_flat_eval.py` (which uses the same map)
    so the search and eval phases operate under the same positional
    setup.

`config.RANDOM_SEED` and the QD model constants are READ from workers
but never mutated — `set_seed` runs once before the executor starts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Allow running this file directly from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from qd.archive import MUTATION_TYPES, QUESTION_TYPES
from qd.bleu import passes_filter
from qd.config import (
    QDConfig,
    SEED_TO_BASELINE,
    baseline_swap_for_seed,
    flat_seed_dir,
    latest_flat_run_dir,
    next_flat_run_dir,
    pool_csv_for_seed,
    test_config,
)
from qd.fitness import (
    fitness_from_judgement,
    run_search_debate,
    run_search_judge,
)
from qd.logging import JsonlWriter, make_failure_record
from qd.mutator import call_mutator
from qd.notify import slack, slack_throttled
from qd.validator import call_validator, classify_question_type


# ── CSV column definitions ────────────────────────────────────────────────────

_PER_MUTATION_COLS = [
    "qid", "original_question_type", "mutation_type",
    "attempts_made", "valid_attempts",
    "best_attempt_index", "best_iteration",
    "best_fitness", "best_judge_log_odds_truth",
    "best_mutated_question", "best_mutated_distractor", "best_swap_flag",
    "best_judge_confidence_estimated",
    "original_question", "correct_answer", "original_distractor", "story_title",
    "covered",
]

_BEST_PER_QID_COLS = [
    "qid", "original_question_type", "covered",
    "best_mutation_type", "best_fitness", "best_judge_log_odds_truth",
    "best_mutated_question", "best_mutated_distractor", "best_swap_flag",
    "best_judge_confidence_estimated",
    "best_attempt_index", "best_iteration",
    "fitness_plain_reword", "fitness_abstraction_lift",
    "fitness_prime_inversion", "fitness_distractor_boost",
    "valid_attempts_total",
    "original_question", "correct_answer", "original_distractor", "story_title",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pool(csv_path: str) -> list[dict]:
    """Load a pool CSV. Missing/empty `question_type` → 'unknown'."""
    pool: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = str(row["id"])
            qt = row.get("question_type")
            if not qt:
                row["question_type"] = "unknown"
            pool.append(row)
    if not pool:
        raise RuntimeError(f"Question pool {csv_path!r} is empty.")
    return pool


def _task_seed(qid: str, mtype: str, iteration: int, attempt_idx: int) -> int:
    """Deterministic 31-bit seed for a (qid, mtype, iter, attempt) task."""
    key = f"{config.RANDOM_SEED}:{qid}:{mtype}:{iteration}:{attempt_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def _resolve_pool_path(seed: int | None, pool_source: str | None) -> str:
    """`--pool-source` wins; else resolve via qd.config.pool_csv_for_seed
    (qd_baseline/base_run_{seed}/both_correct.csv — the canonical per-seed
    flat-grid pool)."""
    if pool_source is not None:
        return pool_source
    if seed is None:
        raise SystemExit("--seed is required when --pool-source is not given.")
    if seed not in SEED_TO_BASELINE:
        raise SystemExit(
            f"seed={seed} is not in SEED_TO_BASELINE "
            f"({sorted(SEED_TO_BASELINE)}). Pass --pool-source PATH to override."
        )
    return str(pool_csv_for_seed(seed))


# ── State classes ─────────────────────────────────────────────────────────────

@dataclass
class _FlatState:
    """In-memory aggregator. All access goes through `self.lock` — workers
    do read-compare-write while holding it."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    # (qid, mtype) → best attempt details
    per_mutation: dict = field(default_factory=dict)
    # qid → global best across all 4 mutations
    best_per_qid: dict = field(default_factory=dict)

    def record_attempt(
        self,
        *,
        qid: str,
        mtype: str,
        iteration: int,
        attempt_idx: int,
        target_qtype: str,
        fitness: float | None,
        mutated_question: str | None,
        mutated_distractor: str | None,
        judgement: dict | None,
        swap: bool,
    ) -> None:
        """Update per-mutation and best-per-qid state."""
        with self.lock:
            key = (qid, mtype)
            if key not in self.per_mutation:
                self.per_mutation[key] = {
                    "attempts_made": 0,
                    "valid_attempts": 0,
                    "best_attempt_index": "",
                    "best_iteration": "",
                    "best_fitness": None,
                    "best_judge_log_odds_truth": "",
                    "best_mutated_question": "",
                    "best_mutated_distractor": "",
                    "best_swap_flag": "",
                    "best_judge_confidence_estimated": "",
                    "original_question_type": target_qtype,
                }
            cell = self.per_mutation[key]
            cell["attempts_made"] += 1

            if fitness is not None and judgement is not None:
                cell["valid_attempts"] += 1
                cur_best_fit = cell["best_fitness"]
                if cur_best_fit is None or fitness > cur_best_fit:
                    cell["best_attempt_index"] = attempt_idx
                    cell["best_iteration"] = iteration
                    cell["best_fitness"] = fitness
                    cell["best_judge_log_odds_truth"] = judgement.get("judge_log_odds_truth", "")
                    cell["best_mutated_question"] = mutated_question or ""
                    cell["best_mutated_distractor"] = mutated_distractor or ""
                    cell["best_swap_flag"] = swap
                    cell["best_judge_confidence_estimated"] = bool(
                        judgement.get("judge_confidence_estimated", False)
                    )

                # Update global best per qid
                cur_qid_best = self.best_per_qid.get(qid)
                if cur_qid_best is None or fitness > cur_qid_best["best_fitness"]:
                    self.best_per_qid[qid] = {
                        "qid": qid,
                        "original_question_type": target_qtype,
                        "best_mutation_type": mtype,
                        "best_fitness": fitness,
                        "best_judge_log_odds_truth": judgement.get("judge_log_odds_truth", ""),
                        "best_mutated_question": mutated_question or "",
                        "best_mutated_distractor": mutated_distractor or "",
                        "best_swap_flag": swap,
                        "best_attempt_index": attempt_idx,
                        "best_iteration": iteration,
                        "best_judge_confidence_estimated": bool(
                            judgement.get("judge_confidence_estimated", False)
                        ),
                    }

    def snapshot_parent_for(self, qid: str, mtype: str) -> tuple[str | None, str | None]:
        """Return (parent_question, parent_distractor) for the slot's stepping
        stone, or (None, None) if no valid candidate exists yet."""
        with self.lock:
            cell = self.per_mutation.get((qid, mtype))
            if (
                cell is not None
                and cell.get("valid_attempts", 0) > 0
                and cell.get("best_fitness") is not None
                and cell.get("best_mutated_question")
            ):
                return (cell["best_mutated_question"], cell["best_mutated_distractor"])
        return (None, None)


# ── CSV rebuild ───────────────────────────────────────────────────────────────

def _rebuild_csvs_from_state(
    state: _FlatState,
    pool_ids: list[str],
    pool_index: dict[str, dict],
    per_mut_path: Path,
    per_qid_path: Path,
) -> None:
    """Write both CSVs from scratch based on the current state snapshot."""
    with state.lock:
        per_mut_snap = {k: dict(v) for k, v in state.per_mutation.items()}
        best_snap = {k: dict(v) for k, v in state.best_per_qid.items()}

    with open(per_mut_path, "w", newline="", encoding="utf-8") as fmt, \
         open(per_qid_path, "w", newline="", encoding="utf-8") as fq:
        wmt = csv.DictWriter(
            fmt, fieldnames=_PER_MUTATION_COLS,
            extrasaction="ignore", quoting=csv.QUOTE_ALL,
        )
        wq = csv.DictWriter(
            fq, fieldnames=_BEST_PER_QID_COLS,
            extrasaction="ignore", quoting=csv.QUOTE_ALL,
        )
        wmt.writeheader()
        wq.writeheader()

        for qid in pool_ids:
            pool_row = pool_index[qid]
            qtype_from_pool = pool_row.get("question_type") or "unknown"
            original_question = pool_row.get("question", "")
            correct_answer = pool_row.get("correct_answer", "")
            original_distractor = pool_row.get("negative_answer", "")
            story_title = pool_row.get("story_title", "")

            for mtype in MUTATION_TYPES:
                cell = per_mut_snap.get((qid, mtype), {})
                covered = bool(cell.get("valid_attempts", 0) > 0)
                wmt.writerow({
                    "qid": qid,
                    "original_question_type": qtype_from_pool,
                    "mutation_type": mtype,
                    "attempts_made": cell.get("attempts_made", 0),
                    "valid_attempts": cell.get("valid_attempts", 0),
                    "best_attempt_index": cell.get("best_attempt_index", "") if covered else "",
                    "best_iteration": cell.get("best_iteration", "") if covered else "",
                    "best_fitness": cell.get("best_fitness", "") if covered else "",
                    "best_judge_log_odds_truth": cell.get("best_judge_log_odds_truth", "") if covered else "",
                    "best_mutated_question": cell.get("best_mutated_question", "") if covered else "",
                    "best_mutated_distractor": cell.get("best_mutated_distractor", "") if covered else "",
                    "best_swap_flag": cell.get("best_swap_flag", "") if covered else "",
                    "best_judge_confidence_estimated": cell.get("best_judge_confidence_estimated", "") if covered else "",
                    "original_question": original_question,
                    "correct_answer": correct_answer,
                    "original_distractor": original_distractor,
                    "story_title": story_title,
                    "covered": covered,
                })

            best_rec = best_snap.get(qid, {})
            covered_overall = bool(best_rec)
            fitness_by_col: dict[str, str | float] = {}
            for mtype in MUTATION_TYPES:
                cell = per_mut_snap.get((qid, mtype), {})
                fitness_by_col[f"fitness_{mtype}"] = (
                    cell.get("best_fitness", "")
                    if cell.get("valid_attempts", 0) > 0 else ""
                )
            valid_attempts_total = sum(
                per_mut_snap.get((qid, mtype), {}).get("valid_attempts", 0)
                for mtype in MUTATION_TYPES
            )
            wq.writerow({
                "qid": qid,
                "original_question_type": qtype_from_pool,
                "covered": covered_overall,
                "best_mutation_type": best_rec.get("best_mutation_type", "") if covered_overall else "",
                "best_fitness": best_rec.get("best_fitness", "") if covered_overall else "",
                "best_judge_log_odds_truth": best_rec.get("best_judge_log_odds_truth", "") if covered_overall else "",
                "best_mutated_question": best_rec.get("best_mutated_question", "") if covered_overall else "",
                "best_mutated_distractor": best_rec.get("best_mutated_distractor", "") if covered_overall else "",
                "best_swap_flag": best_rec.get("best_swap_flag", "") if covered_overall else "",
                "best_judge_confidence_estimated": best_rec.get("best_judge_confidence_estimated", "") if covered_overall else "",
                "best_attempt_index": best_rec.get("best_attempt_index", "") if covered_overall else "",
                "best_iteration": best_rec.get("best_iteration", "") if covered_overall else "",
                "fitness_plain_reword": fitness_by_col.get("fitness_plain_reword", ""),
                "fitness_abstraction_lift": fitness_by_col.get("fitness_abstraction_lift", ""),
                "fitness_prime_inversion": fitness_by_col.get("fitness_prime_inversion", ""),
                "fitness_distractor_boost": fitness_by_col.get("fitness_distractor_boost", ""),
                "valid_attempts_total": valid_attempts_total,
                "original_question": original_question,
                "correct_answer": correct_answer,
                "original_distractor": original_distractor,
                "story_title": story_title,
            })


# ── State save / log replay ───────────────────────────────────────────────────

def _save_state(
    state_path: Path,
    *,
    seed: int | None,
    pool_size: int,
    pool_source: str,
    iterations_requested: int,
    completed_iterations: int,
) -> None:
    """Atomically save flat_grid_state.json (tmp + os.replace)."""
    payload = {
        "seed": seed,
        "pool_size": pool_size,
        "pool_source": pool_source,
        "iterations_requested": iterations_requested,
        "completed_iterations": completed_iterations,
        "global_attempts": completed_iterations * pool_size * len(MUTATION_TYPES),
        "mutation_types": list(MUTATION_TYPES),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, state_path)


def _replay_log_to_state(
    attempts_path: Path,
    pool_index: dict[str, dict],
) -> tuple[_FlatState, set[tuple[int, str, str]]]:
    """Rebuild _FlatState by replaying flat_attempts.jsonl.

    Returns (state, logged_slots) where logged_slots is the set of
    (iteration, qid, mutation_type) tuples seen — used to skip already-done
    slots when resuming inside the in-progress iteration.
    """
    state = _FlatState()
    logged: set[tuple[int, str, str]] = set()
    if not attempts_path.exists():
        return state, logged

    with open(attempts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qid = str(rec.get("qid", ""))
            mtype = rec.get("mutation_type")
            iteration = int(rec.get("iteration", 0) or 0)
            if iteration <= 0 or mtype not in MUTATION_TYPES or qid not in pool_index:
                continue
            logged.add((iteration, qid, mtype))

            judgement_view = {
                "judge_log_odds_truth": rec.get("judge_log_odds_truth"),
                "judge_confidence_estimated": bool(rec.get("judge_confidence_estimated", False)),
                "judge_answer": rec.get("judge_answer"),
                "correct_letter": rec.get("correct_letter"),
            }
            fitness = fitness_from_judgement(judgement_view)
            target_qtype = pool_index[qid].get("question_type") or "unknown"
            state.record_attempt(
                qid=qid,
                mtype=mtype,
                iteration=iteration,
                attempt_idx=int(rec.get("attempt_index", 0) or 0),
                target_qtype=target_qtype,
                fitness=fitness,
                mutated_question=rec.get("mutated_question"),
                mutated_distractor=rec.get("mutated_distractor"),
                judgement=judgement_view,
                swap=bool(rec.get("swap", False)),
            )
    return state, logged


# ── Per-slot worker ───────────────────────────────────────────────────────────

def _run_one_slot(
    *,
    qid: str,
    mtype: str,
    iteration: int,
    pool_row: dict,
    qd: QDConfig,
    state: _FlatState,
    swap_map: dict[str, bool],
    transcripts_writer: JsonlWriter,
    attempts_writer: JsonlWriter,
    failures_writer: JsonlWriter,
) -> str:
    """Run one (qid, mutation_type) slot for the given iteration.

    Up to 3 internal attempts; stops after the first valid candidate is
    produced (debate + judge succeed and fitness can be computed → 'accepted',
    or fitness=None but record was written → 'no_fit').

    Returns one of:
      'accepted' — at least one valid candidate logged with computable fitness
      'no_fit' — candidate produced but fitness couldn't be computed
      'failed_all_retries' — all 3 attempts hit some pipeline failure
    """
    target_qtype_raw = pool_row.get("question_type") or "unknown"
    target_qtype = target_qtype_raw if target_qtype_raw in QUESTION_TYPES else QUESTION_TYPES[0]
    target_cell: tuple[str, str] = (target_qtype, mtype)

    # Parent selection at SLOT entry — fixed for all retries within this slot.
    p_q, p_d = state.snapshot_parent_for(qid, mtype)
    if p_q is None or p_d is None:
        parent_question = pool_row["question"]
        parent_distractor = pool_row["negative_answer"]
        stepping_stone = False
    else:
        parent_question = p_q
        parent_distractor = p_d
        stepping_stone = True

    def _log_fail(
        stage: str,
        detail: str,
        attempt_idx: int,
        mq: str | None = None,
        md: str | None = None,
    ) -> None:
        rec = make_failure_record(
            iter_no=iteration,
            qid=qid,
            target_cell=target_cell,
            stage=stage,
            detail=(
                f"task={qid}/{mtype}/iter={iteration}/attempt={attempt_idx} " + detail
            ),
            mutated_question=mq,
            mutated_distractor=md,
        )
        rec["iteration"] = iteration
        rec["attempt_index"] = attempt_idx
        rec["stepping_stone"] = stepping_stone
        failures_writer.write_record(rec)

    swap = swap_map.get(str(qid), False)
    for attempt_idx in range(3):
        seed_val = _task_seed(qid, mtype, iteration, attempt_idx)

        # a) Mutate from PARENT (stepping stone or original)
        try:
            m = call_mutator(
                parent_question=parent_question,
                correct_answer=pool_row["correct_answer"],
                parent_distractor=parent_distractor,
                mutation_type=mtype,
                target_question_type=target_qtype,
                seed=seed_val,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            _log_fail("mutator_parse", f"{e!r}\n{tb}", attempt_idx)
            continue
        if m is None:
            _log_fail("mutator_parse", "", attempt_idx)
            continue

        # b) BLEU vs PARENT (spec: diversity measured against actual parent)
        parent_combined = parent_question + " " + parent_distractor
        mutated_combined = m["mutated_question"] + " " + m["mutated_distractor"]
        bleu_score_val, ok = passes_filter(
            parent_combined, mutated_combined, qd.bleu_threshold,
        )
        if not ok:
            _log_fail(
                "bleu_filter",
                f"score={bleu_score_val:.3f}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue

        # c) Descriptor check
        try:
            detected_type = classify_question_type(
                m["mutated_question"], seed=seed_val, target_type=target_qtype,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            _log_fail(
                "descriptor_mismatch",
                f"classifier raised {e!r}\n{tb}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue
        if detected_type != target_qtype:
            _log_fail(
                "descriptor_mismatch",
                f"expected={target_qtype} got={detected_type}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue

        # d) Validity check — always against ORIGINAL pool_row (ground truth)
        try:
            valid_pair = call_validator(
                story=pool_row["story"],
                original_question=pool_row["question"],
                correct_answer=pool_row["correct_answer"],
                distractor=pool_row["negative_answer"],
                mutated_question=m["mutated_question"],
                mutated_distractor=m["mutated_distractor"],
                mutation_type=mtype,
                seed=seed_val,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            _log_fail(
                "validity",
                f"validator raised {e!r}\n{tb}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue
        if not valid_pair:
            _log_fail(
                "validity",
                f"bleu={bleu_score_val:.3f}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue

        # e) Debate + judge
        candidate_row = dict(pool_row)
        candidate_row["question"] = m["mutated_question"]
        candidate_row["negative_answer"] = m["mutated_distractor"]
        try:
            transcript = run_search_debate(candidate_row, swap)
            judgement = run_search_judge(transcript)
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
                    "vertex_quota",
                    300.0,
                    (
                        f":warning: flat_grid hitting Vertex quota "
                        f"(seed={config.RANDOM_SEED}, qid={qid}, mtype={mtype}, "
                        f"iter={iteration}): `{err_repr[:200]}`"
                    ),
                )
            _log_fail(
                "debate_or_judge",
                f"{e!r}\n{tb}",
                attempt_idx,
                mq=m["mutated_question"],
                md=m["mutated_distractor"],
            )
            continue

        # f) Log success + update state
        transcripts_writer.write_record({
            **transcript,
            "qid": qid,
            "mutation_type": mtype,
            "iteration": iteration,
            "attempt_index": attempt_idx,
            "stepping_stone": stepping_stone,
            "target_cell": list(target_cell),
            "bleu_to_parent": bleu_score_val,
        })
        judgement_for_log = dict(judgement)
        judgement_for_log.pop("rounds", None)
        judgement_for_log.pop("transcript_str", None)
        attempts_writer.write_record({
            **judgement_for_log,
            "qid": qid,
            "mutation_type": mtype,
            "iteration": iteration,
            "attempt_index": attempt_idx,
            "stepping_stone": stepping_stone,
            "target_cell": list(target_cell),
            "bleu_to_parent": bleu_score_val,
            "mutated_question": m["mutated_question"],
            "mutated_distractor": m["mutated_distractor"],
            "swap": swap,
        })

        fitness = fitness_from_judgement(judgement)
        state.record_attempt(
            qid=qid,
            mtype=mtype,
            iteration=iteration,
            attempt_idx=attempt_idx,
            target_qtype=target_qtype,
            fitness=fitness,
            mutated_question=m["mutated_question"],
            mutated_distractor=m["mutated_distractor"],
            judgement=judgement,
            swap=swap,
        )
        return "accepted" if fitness is not None else "no_fit"

    return "failed_all_retries"


# ── Report ────────────────────────────────────────────────────────────────────

def _render_report(
    report_path: Path,
    state: _FlatState,
    pool_ids: list[str],
    iterations_requested: int,
    completed_iterations: int,
    status_counts: dict[str, int],
    elapsed: float,
    pool_source: str,
) -> None:
    with state.lock:
        best_snap = dict(state.best_per_qid)
        per_mut_snap = {k: dict(v) for k, v in state.per_mutation.items()}

    pool_size = len(pool_ids)
    seed = config.RANDOM_SEED

    covered_by_mtype: dict[str, int] = {}
    for mtype in MUTATION_TYPES:
        covered_by_mtype[mtype] = sum(
            1 for qid in pool_ids
            if per_mut_snap.get((qid, mtype), {}).get("valid_attempts", 0) > 0
        )
    covered_any = sum(1 for qid in pool_ids if qid in best_snap)

    fitnesses_by_mtype: dict[str, list[float]] = {m: [] for m in MUTATION_TYPES}
    for mtype in MUTATION_TYPES:
        for qid in pool_ids:
            cell = per_mut_snap.get((qid, mtype), {})
            if cell.get("valid_attempts", 0) > 0 and cell.get("best_fitness") is not None:
                fitnesses_by_mtype[mtype].append(cell["best_fitness"])
    all_fitnesses = [f for fs in fitnesses_by_mtype.values() for f in fs]

    def _fmt_fitness_row(mtype: str) -> str:
        fs = fitnesses_by_mtype[mtype]
        if not fs:
            return f"| {mtype} | — | — | — | — |"
        return (
            f"| {mtype} | {min(fs):.3f} | {statistics.fmean(fs):.3f} | "
            f"{statistics.median(fs):.3f} | {max(fs):.3f} |"
        )

    winner_counts: dict[str, int] = {m: 0 for m in MUTATION_TYPES}
    for rec in best_snap.values():
        bm = rec.get("best_mutation_type")
        if bm in winner_counts:
            winner_counts[bm] += 1

    uncovered = [qid for qid in pool_ids if qid not in best_snap]

    fail_stages = [
        "failed_all_retries", "no_fit", "accepted",
    ]

    total_valid = sum(
        per_mut_snap.get((qid, mtype), {}).get("valid_attempts", 0)
        for qid in pool_ids for mtype in MUTATION_TYPES
    )
    total_attempts_made = sum(
        per_mut_snap.get((qid, mtype), {}).get("attempts_made", 0)
        for qid in pool_ids for mtype in MUTATION_TYPES
    )

    lines: list[str] = []
    lines.append(f"# Flat grid report — seed {seed}\n")
    lines.append("**Run metadata**\n")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    lines.append(f"| Seed | {seed} |")
    lines.append(f"| Pool source | {pool_source} |")
    lines.append(f"| Pool size | {pool_size} |")
    lines.append(f"| Iterations requested | {iterations_requested} |")
    lines.append(f"| Iterations completed | {completed_iterations} |")
    lines.append(f"| Total attempts made (incl retries) | {total_attempts_made} |")
    lines.append(f"| Total valid attempts | {total_valid} |")
    lines.append(f"| Wall time | {elapsed:.1f}s |")
    lines.append(f"| Mutator model | {config.QD_MUTATOR_MODEL} |")
    lines.append(f"| Search debater | {config.QD_SEARCH_DEBATER_MODEL} |")
    lines.append(f"| Search judge | {config.QD_SEARCH_JUDGE_MODEL} |")
    lines.append(f"| Validator | {config.QD_VALIDATOR_MODEL} |")
    lines.append("")
    lines.append("## Coverage\n")
    lines.append("| Mutation type | qids covered | coverage % |")
    lines.append("|---|---:|---:|")
    for mtype in MUTATION_TYPES:
        n = covered_by_mtype[mtype]
        pct = 100.0 * n / pool_size if pool_size else 0.0
        lines.append(f"| {mtype} | {n} / {pool_size} | {pct:.1f}% |")
    total_cells = pool_size * len(MUTATION_TYPES)
    cells_filled = sum(covered_by_mtype[m] for m in MUTATION_TYPES)
    cell_pct = 100.0 * cells_filled / total_cells if total_cells else 0.0
    any_pct = 100.0 * covered_any / pool_size if pool_size else 0.0
    lines.append(f"| **All cells (qid × mutation)** | **{cells_filled} / {total_cells}** | **{cell_pct:.1f}%** |")
    lines.append(f"| **Qids with ≥1 mutation (eval scope)** | **{covered_any} / {pool_size}** | **{any_pct:.1f}%** |")
    lines.append("")
    lines.append("## Fitness summary (covered qids only)\n")
    lines.append("| Mutation type | min | mean | median | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for mtype in MUTATION_TYPES:
        lines.append(_fmt_fitness_row(mtype))
    if all_fitnesses:
        lines.append(
            f"| Overall | {min(all_fitnesses):.3f} | {statistics.fmean(all_fitnesses):.3f} | "
            f"{statistics.median(all_fitnesses):.3f} | {max(all_fitnesses):.3f} |"
        )
    else:
        lines.append("| Overall | — | — | — | — |")
    lines.append("")
    lines.append("## Winning-mutation distribution\n")
    lines.append("| Mutation type | won for N qids | % of covered |")
    lines.append("|---|---:|---:|")
    for mtype in MUTATION_TYPES:
        n = winner_counts[mtype]
        pct = 100.0 * n / covered_any if covered_any else 0.0
        lines.append(f"| {mtype} | {n} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Uncovered qids\n")
    if uncovered:
        lines.append(", ".join(uncovered))
    else:
        lines.append("None — every pool question covered.")
    lines.append("")
    lines.append("## Slot outcome breakdown (this run's status counts)\n")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for stage in fail_stages:
        lines.append(f"| {stage} | {status_counts.get(stage, 0)} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flat per-qid mutation grid — iterative stepping-stone, resumable."
        ),
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Resolve pool from qd_baseline/base_run_{seed}/both_correct.csv "
                             "matching this seed. Sets RANDOM_SEED=N. Required unless "
                             "--pool-source is given.")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of outer stepping-stone iterations (default: 3). "
                             "Each iteration runs every (qid, mutation_type) slot "
                             "with up to 3 internal retries.")
    parser.add_argument("--pool-source", type=str, default=None,
                        help="Override pool CSV path (otherwise auto-resolved from --seed).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent slot workers (default: 8). Lower if you hit "
                             "Vertex RESOURCE_EXHAUSTED on the search judge.")
    parser.add_argument("--force-new", action="store_true",
                        help="Start a fresh flat_grid_run_NNN even if a resumable "
                             "state file exists.")
    parser.add_argument("--test", action="store_true",
                        help="Use qd_results_test/ paths.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Reserved for future plot rendering; currently a no-op.")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be ≥ 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be ≥ 1")
    if args.seed is None and args.pool_source is None:
        raise SystemExit("--seed is required (or pass --pool-source PATH).")

    if args.seed is not None:
        config.set_seed(args.seed)

    # ── Resolve pool path ──────────────────────────────────────────────────
    pool_path = _resolve_pool_path(args.seed, args.pool_source)
    pool = _load_pool(pool_path)
    pool_ids = [r["id"] for r in pool]
    pool_index = {r["id"]: r for r in pool}
    pool_size = len(pool_ids)
    print(f"[flat-grid] pool: {pool_path} ({pool_size} rows)")

    # ── Resolve baseline swap map ──────────────────────────────────────────
    # Per-QID swap is locked to the baseline assignment for the lifetime of
    # the run, so search and eval operate under the same positional setup.
    # Legacy --pool-source-only invocations (no --seed) fall back to all-
    # False (matches the historical default before per-seed pooling).
    if args.seed is not None:
        swap_map = baseline_swap_for_seed(args.seed)
        missing = [q for q in pool_ids if q not in swap_map]
        if missing:
            print(
                f"[flat-grid] WARN: {len(missing)} pool qids missing from "
                f"baseline swap map (e.g. {missing[:3]}). They will default "
                f"to swap=False."
            )
    else:
        swap_map = {}

    # ── Resolve run dir (resume vs fresh) ──────────────────────────────────
    # Flat-grid outputs live under qd_results/flat_results/seed_N/ — keeps
    # the flat-grid pipeline visually separate from the MAP-Elites runs
    # that live directly under qd_results/seed_N/.
    if args.seed is not None:
        base_results = flat_seed_dir(args.seed, test=args.test)
    else:
        # Legacy --pool-source-only path (no seed): fall back to the
        # historical layout (test root or production root) so old single-
        # pool invocations still work.
        base_results = Path(
            config.QD_RESULTS_TEST_DIR if args.test else config.QD_RESULTS_DIR
        )

    state: _FlatState
    already_logged_slots: set[tuple[int, str, str]]
    start_iteration: int

    resumed = False
    if not args.force_new:
        latest = latest_flat_run_dir(base_results)
        if latest is not None:
            state_path_candidate = latest / "flat_grid_state.json"
            if state_path_candidate.exists():
                try:
                    with open(state_path_candidate, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                except Exception as e:
                    print(f"[flat-grid] WARN: could not parse {state_path_candidate}: {e!r}")
                    saved = None
                if saved is not None:
                    completed_prev = int(saved.get("completed_iterations", 0) or 0)
                    if completed_prev >= args.iterations:
                        print(
                            f"[flat-grid] {latest.name} already completed "
                            f"{completed_prev} iterations (requested {args.iterations}) — "
                            f"nothing to do. Pass --force-new for a fresh run."
                        )
                        return
                    # Resume into this dir.
                    run_dir = latest
                    search_dir = run_dir / "search_flat"
                    if not search_dir.exists():
                        # Older run layout — bail rather than guess.
                        print(
                            f"[flat-grid] WARN: {run_dir.name} has no search_flat/ "
                            "subfolder; starting a fresh run instead."
                        )
                    else:
                        state, already_logged_slots = _replay_log_to_state(
                            search_dir / "flat_attempts.jsonl", pool_index,
                        )
                        start_iteration = completed_prev + 1
                        resumed = True
                        print(
                            f"[flat-grid] resuming {run_dir.name} from iteration "
                            f"{start_iteration} (requested {args.iterations}); "
                            f"replayed {len(already_logged_slots)} logged slots, "
                            f"covered={len(state.best_per_qid)}/{pool_size}"
                        )

    if not resumed:
        run_dir, _ = next_flat_run_dir(base_results)
        run_dir.mkdir(parents=True, exist_ok=True)
        search_dir = run_dir / "search_flat"
        search_dir.mkdir(parents=True, exist_ok=True)
        state = _FlatState()
        already_logged_slots = set()
        start_iteration = 1
        print(f"[flat-grid] fresh run directory: {run_dir}")
    else:
        # search_dir already set above
        run_dir.mkdir(parents=True, exist_ok=True)
        search_dir.mkdir(parents=True, exist_ok=True)
        print(f"[flat-grid] run directory: {run_dir}")

    qd = test_config() if args.test else QDConfig()
    qd.results_dir = str(run_dir)

    config.validate_llm_credentials()
    from core.gemini_client import format_pool_diagnostics
    print(format_pool_diagnostics(workers=args.workers))

    # ── Paths ──────────────────────────────────────────────────────────────
    per_mut_path = run_dir / "flat_per_mutation.csv"
    per_qid_path = run_dir / "flat_best_per_qid.csv"
    state_path = run_dir / "flat_grid_state.json"
    report_path = search_dir / "flat_grid_report.md"

    transcripts_writer = JsonlWriter(str(search_dir / "flat_transcripts.jsonl"))
    attempts_writer = JsonlWriter(str(search_dir / "flat_attempts.jsonl"))
    failures_writer = JsonlWriter(str(search_dir / "flat_failures.jsonl"))

    # If we resumed, rebuild the CSVs immediately so they reflect replayed state.
    if resumed:
        _rebuild_csvs_from_state(state, pool_ids, pool_index, per_mut_path, per_qid_path)

    # ── Run iteration loop ─────────────────────────────────────────────────
    slots_per_iter = pool_size * len(MUTATION_TYPES)
    print(
        f"[flat-grid] iterations {start_iteration}..{args.iterations} "
        f"({slots_per_iter} slots × {len(MUTATION_TYPES)} mutations / iter, "
        f"3-retry budget per slot, --workers={args.workers})"
    )
    status_counts: dict[str, int] = {}
    t_start = time.time()

    try:
        for iteration in range(start_iteration, args.iterations + 1):
            iter_t0 = time.time()
            slots = [
                (qid, mtype)
                for qid in pool_ids
                for mtype in MUTATION_TYPES
                if (iteration, qid, mtype) not in already_logged_slots
            ]
            skipped = slots_per_iter - len(slots)
            print(
                f"\n[flat-grid] ── iteration {iteration}/{args.iterations} ── "
                f"{len(slots)} slots to run ({skipped} skipped from log)"
            )

            completed_in_iter = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(
                        _run_one_slot,
                        qid=qid,
                        mtype=mtype,
                        iteration=iteration,
                        pool_row=pool_index[qid],
                        qd=qd,
                        state=state,
                        swap_map=swap_map,
                        transcripts_writer=transcripts_writer,
                        attempts_writer=attempts_writer,
                        failures_writer=failures_writer,
                    )
                    for (qid, mtype) in slots
                ]
                for fut in as_completed(futures):
                    status = fut.result()
                    status_counts[status] = status_counts.get(status, 0) + 1
                    completed_in_iter += 1
                    if completed_in_iter % 50 == 0 or completed_in_iter == len(slots):
                        with state.lock:
                            covered = len(state.best_per_qid)
                        print(
                            f"[flat-grid] iter {iteration}: "
                            f"{completed_in_iter}/{len(slots)} slots done "
                            f"({100.0 * completed_in_iter / max(1, len(slots)):.1f}%) — "
                            f"covered={covered}/{pool_size}  "
                            f"elapsed={time.time() - t_start:.0f}s"
                        )

            # End-of-iteration: CSVs, report, state file
            _rebuild_csvs_from_state(state, pool_ids, pool_index, per_mut_path, per_qid_path)
            _render_report(
                report_path=report_path,
                state=state,
                pool_ids=pool_ids,
                iterations_requested=args.iterations,
                completed_iterations=iteration,
                status_counts=status_counts,
                elapsed=time.time() - t_start,
                pool_source=pool_path,
            )
            _save_state(
                state_path,
                seed=args.seed,
                pool_size=pool_size,
                pool_source=pool_path,
                iterations_requested=args.iterations,
                completed_iterations=iteration,
            )

            with state.lock:
                covered_now = len(state.best_per_qid)
            print(
                f"[flat-grid] iter {iteration} done in {time.time() - iter_t0:.1f}s — "
                f"covered={covered_now}/{pool_size}  "
                f"status={dict(status_counts)}"
            )

            # Next iteration starts fresh w.r.t. log-skip: only slots logged in
            # iter+1 (from a partially-completed previous run, on resume of a
            # resume) would be present. Iter just finished is fully accounted
            # for in state already.
    except KeyboardInterrupt:
        raise
    except Exception:
        tb = traceback.format_exc()
        slack(
            f":x: flat_grid ABORTED — seed={args.seed}, run={run_dir.name}\n"
            f"```\n{tb[-1500:]}\n```"
        )
        raise
    finally:
        transcripts_writer.close()
        attempts_writer.close()
        failures_writer.close()

    elapsed = time.time() - t_start
    with state.lock:
        covered_count = len(state.best_per_qid)
        per_mut_snap = {k: dict(v) for k, v in state.per_mutation.items()}
    print(f"\n[flat-grid] all iterations done in {elapsed:.1f}s")
    print(f"[flat-grid] status counts: {status_counts}")
    print(f"[flat-grid] covered: {covered_count}/{pool_size} qids "
          f"({100.0 * covered_count / pool_size:.1f}%)")
    print(f"[flat-grid] per_mutation CSV → {per_mut_path}")
    print(f"[flat-grid] best_per_qid CSV → {per_qid_path}")
    print(f"[flat-grid] state            → {state_path}")
    print(f"[flat-grid] report           → {report_path}")

    fits_by_mtype: dict[str, list[float]] = {m: [] for m in MUTATION_TYPES}
    fallback_cells = 0
    for (_qid, mtype), cell in per_mut_snap.items():
        f = cell.get("best_fitness")
        if f is not None:
            fits_by_mtype[mtype].append(f)
        if cell.get("best_judge_confidence_estimated"):
            fallback_cells += 1
    mut_lines = []
    for m in MUTATION_TYPES:
        fs = fits_by_mtype[m]
        if fs:
            mut_lines.append(
                f"  - {m}: max={max(fs):.2f}, mean={statistics.fmean(fs):.2f}, "
                f"n={len(fs)}"
            )
        else:
            mut_lines.append(f"  - {m}: no valid attempts")
    fallback_line = (
        f"\n:warning: {fallback_cells} cells used LLM-confidence fallback"
        if fallback_cells else ""
    )
    slack(
        f":white_check_mark: flat_grid done — seed={args.seed}, "
        f"run={run_dir.name}\n"
        f"iterations: {args.iterations}, elapsed: {elapsed:.0f}s\n"
        f"covered: {covered_count}/{pool_size} qids "
        f"({100.0 * covered_count / pool_size:.1f}%)\n"
        f"slot status: {dict(status_counts)}\n"
        f"best fitness by mutation:\n" + "\n".join(mut_lines) + fallback_line
    )


if __name__ == "__main__":
    main()
