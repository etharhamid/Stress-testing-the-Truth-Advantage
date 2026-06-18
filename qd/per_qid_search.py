"""Per-QID 1D MAP-Elites with eval-gated removal.

Each QID owns a 1D archive of 4 cells, one per mutation type. Cells compete:
an elite from `plain_reword` can be the parent for mutating into
`abstraction_lift`, and vice versa. Warm-started from a flat_grid_run_NNN's
flat_per_mutation.csv so iter 1 already starts from the flat-grid bests.

After every `--checkpoint-every` search iterations per active QID, each
active QID's highest-fitness cell is run through a baseline-grade pro
debate (gemini-3.1-pro-preview) with the baseline-matched swap, then judged
by both gemini-2.5-flash and gemini-2.5-flash-lite. A QID is **confirmed
deceptive** iff both judges return judge_correct=False on that single best
framing; confirmed QIDs are removed from the active pool. The loop
continues until all QIDs are confirmed, exhausted, or no active set remains.

Layout (per seed, per run):

    qd_results/map_results/seed_{N}/
      per_qid_summary.json
      per_qid_run_{NNN}/
        per_qid_archive.json            # all QIDs' 4-cell archives
        per_qid_state.json              # per-QID iter_count + status (resume anchor)
        search/  search_{transcripts,judgements,failures}.jsonl
        checkpoint/  checkpoint_{transcripts,judgements_flash,judgements_flash-lite}.jsonl
                     checkpoint_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import config

from core.judge_engine import judge_debate

from qd.archive import Elite, MUTATION_TYPES
from qd.bleu import passes_filter
from qd.config import (
    baseline_swap_for_seed,
    flat_seed_dir,
    latest_flat_run_dir,
    map_seed_dir,
    pool_csv_for_seed,
)
from qd.fitness import (
    fitness_from_judgement,
    run_eval_debate,
    run_search_debate,
    run_search_judge,
)
from qd.logging import JsonlWriter, make_failure_record
from qd.mutator import call_mutator
from qd.notify import slack
from qd.selection import _P_EMPTY_PREF, _P_POOL_INJECT, _SOFTMAX_T
from qd.validator import call_validator, classify_question_type


CHECKPOINT_JUDGES: tuple[str, str] = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

# JSONL filename for each checkpoint judge (model id has a dot, file name keeps it as-is)
_JUDGE_FILE_TAG = {
    "gemini-2.5-flash":      "flash",
    "gemini-2.5-flash-lite": "flash-lite",
}

# Schema used when appending confirmed framings to qd_results/flat_results/
# seed_{N}/best_framings.csv. Must match scripts/clean_flat_runs.py:BEST_FRAMINGS_FIELDS.
_BEST_FRAMINGS_FIELDS = [
    "qid", "source_run", "confirmed_at_iter",
    "original_question_type", "confirming_question_type", "best_mutation_type",
    "best_fitness", "best_judge_log_odds_truth",
    "best_swap_flag", "baseline_swap",
    "best_mutated_question", "best_mutated_distractor",
    "original_question", "correct_answer", "original_distractor", "story_title",
    "best_attempt_index", "best_iteration", "best_judge_confidence_estimated",
    "fitness_plain_reword", "fitness_abstraction_lift",
    "fitness_prime_inversion", "fitness_distractor_boost",
    "valid_attempts_total", "covered",
    "eval_swap", "flash_judge_answer", "flash_judge_correct",
    "lite_judge_answer", "lite_judge_correct",
]


def _best_framings_path(seed: int) -> Path:
    return flat_seed_dir(seed) / "best_framings.csv"


def _migrate_best_framings_schema(path: Path) -> None:
    """One-shot upgrade of an existing best_framings.csv to the stage-3 schema.

    Adds the `confirming_question_type` column to every existing row,
    defaulting it to that row's `original_question_type` (stage-2 cells
    never crossed types, so original == confirming by construction).
    Rewrites the file atomically with the full new column list.

    Triggered lazily by `_append_confirmation_to_best_framings` the first
    time it sees an existing file whose header lacks the new column.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        if not r.get("confirming_question_type"):
            r["confirming_question_type"] = r.get("original_question_type", "")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=_BEST_FRAMINGS_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)


def _load_already_confirmed_qids(seed: int) -> set[str]:
    """Return the set of QIDs already locked in `best_framings.csv` for this seed.

    These QIDs are excluded from per-QID search entirely — their adversarial
    framings are already confirmed (both judges wrong under the pro debater)
    so spending further budget on them is wasted.
    """
    path = _best_framings_path(seed)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {str(r["qid"]) for r in csv.DictReader(f)}


def _append_confirmation_to_best_framings(
    *,
    seed: int,
    qid: str,
    elite: Elite,
    best_mtype: str,
    pool_row: dict,
    baseline_swap: dict[str, bool],
    eval_swap: bool,
    iter_count: int,
    source_run_name: str,
    confirming_qtype: str | None = None,
) -> None:
    """Append a newly-confirmed framing to the seed's best_framings.csv.

    Called from inside _checkpoint_one_qid after both judges return
    judge_correct=False on the QID's best cell. flash/lite judge_correct
    are definitionally False at confirmation, so we hardcode them.

    `confirming_qtype` is the question type of the cell that confirmed.
    For stage-2 (1D per-QID search) it equals `original_question_type`
    because every cell sits in the QID's original-type row. For stage-3
    (2D per-QID search) it may differ, telling readers whether the
    structural type-shift was what unlocked deception.
    """
    original_qtype = pool_row.get("question_type", "")
    if confirming_qtype is None:
        confirming_qtype = original_qtype
    path = _best_framings_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if not write_header:
        # Lazy schema migration. Check the on-disk header for the new
        # column; rewrite the file once if it's missing. The migration
        # is idempotent and the cost is one full read+write of the CSV
        # (small, a few dozen rows per seed).
        with open(path, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if "confirming_question_type" not in existing_header:
            _migrate_best_framings_schema(path)
    row = {
        "qid":                              qid,
        "source_run":                       source_run_name,
        "confirmed_at_iter":                iter_count,
        "original_question_type":           original_qtype,
        "confirming_question_type":         confirming_qtype,
        "best_mutation_type":               best_mtype,
        "best_fitness":                     elite.fitness,
        "best_judge_log_odds_truth":        elite.judge_log_odds_truth,
        "best_swap_flag":                   elite.swap,
        "baseline_swap":                    baseline_swap.get(qid, ""),
        "best_mutated_question":            elite.mutated_question,
        "best_mutated_distractor":          elite.mutated_distractor,
        "original_question":                pool_row.get("question", ""),
        "correct_answer":                   pool_row.get("correct_answer", ""),
        "original_distractor":              pool_row.get("negative_answer", ""),
        "story_title":                      pool_row.get("story_title", ""),
        "best_iteration":                   elite.iteration,
        "covered":                          True,
        "eval_swap":                        eval_swap,
        "flash_judge_answer":               "",
        "flash_judge_correct":              False,
        "lite_judge_answer":                "",
        "lite_judge_correct":               False,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BEST_FRAMINGS_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Atomic-write helpers ─────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)


# ── Per-QID archive (de)serialisation ────────────────────────────────────────

def _archive_to_jsonable(archive: dict[str, dict[str, Elite | None]]) -> dict:
    return {
        "qids": {
            qid: {
                "cells": {
                    mt: (asdict(e) if e is not None else None)
                    for mt, e in cells.items()
                }
            }
            for qid, cells in archive.items()
        }
    }


def _archive_from_jsonable(data: dict) -> dict[str, dict[str, Elite | None]]:
    out: dict[str, dict[str, Elite | None]] = {}
    for qid, payload in data.get("qids", {}).items():
        cells_d = payload.get("cells", {})
        out[qid] = {
            mt: (Elite.from_dict(cells_d[mt]) if cells_d.get(mt) is not None else None)
            for mt in MUTATION_TYPES
        }
    return out


def _save_archive(
    archive: dict[str, dict[str, Elite | None]],
    path: Path,
    archive_lock: threading.Lock,
) -> None:
    with archive_lock:
        data = _archive_to_jsonable(archive)
    _atomic_write_json(path, data)


def _save_state(state: dict, path: Path, state_lock: threading.Lock) -> None:
    with state_lock:
        snapshot = json.loads(json.dumps(state))
    _atomic_write_json(path, snapshot)


# ── Pool + warm-start ────────────────────────────────────────────────────────

def _load_pool(pool_csv: Path) -> tuple[dict[str, dict], list[str]]:
    pool_by_qid: dict[str, dict] = {}
    qids: list[str] = []
    with open(pool_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row["id"])
            pool_by_qid[qid] = row
            qids.append(qid)
    return pool_by_qid, qids


def _resolve_flat_run(seed: int, flat_run_n: int | None) -> Path:
    base = flat_seed_dir(seed)
    if flat_run_n is not None:
        cand = base / f"flat_grid_run_{flat_run_n:03d}"
        if not cand.exists():
            raise FileNotFoundError(f"Flat run not found: {cand}")
        return cand
    latest = latest_flat_run_dir(base)
    if latest is None:
        raise FileNotFoundError(f"No flat_grid_run_NNN under {base}/")
    return latest


def _build_warm_start_transcript_index(
    flat_run_dir: Path,
) -> dict[tuple[str, str], dict]:
    """One-time scan of `flat_grid_run_NNN/search_flat/flat_transcripts.jsonl`
    to build `{(qid, mutation_type): transcript}` for the cell-best rows.

    A cell's best is identified by `(qid, mutation_type, iteration ==
    best_iteration)` where best_iteration comes from `flat_per_mutation.csv`.
    Used by `_lazy_probe_best_cell` so warm-started cells reuse the
    existing flash-lite debate instead of running a fresh one.

    Returns an empty dict if either file is missing or unreadable.
    Callers fall back to a fresh debate when the lookup misses.
    """
    csv_path = flat_run_dir / "flat_per_mutation.csv"
    jsonl_path = flat_run_dir / "search_flat" / "flat_transcripts.jsonl"
    if not csv_path.exists() or not jsonl_path.exists():
        return {}
    # Step 1: map (qid, mtype) -> best_iteration for covered rows.
    target: dict[tuple[str, str], int] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("covered", "")).strip().lower() not in ("true", "1"):
                    continue
                try:
                    best_iter = int(row.get("best_iteration") or 0)
                except (ValueError, TypeError):
                    continue
                qid = str(row["qid"])
                mtype = row.get("mutation_type", "")
                if mtype not in MUTATION_TYPES:
                    continue
                target[(qid, mtype)] = best_iter
    except Exception:
        return {}
    # Step 2: scan transcripts JSONL, keep the row matching (qid, mtype,
    # iteration == best_iteration). Records past the first match for a
    # cell are ignored (stable on duplicate logging if it ever happened).
    index: dict[tuple[str, str], dict] = {}
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(rec.get("qid", ""))
                mtype = rec.get("mutation_type", "")
                key = (qid, mtype)
                if key not in target or key in index:
                    continue
                try:
                    it = int(rec.get("iteration") or 0)
                except (ValueError, TypeError):
                    continue
                if it == target[key]:
                    index[key] = rec
    except Exception:
        return index
    return index


def _warm_start_archive(
    flat_run_dir: Path,
    pool_by_qid: dict[str, dict],
) -> dict[str, dict[str, Elite | None]]:
    """Seed each QID's 4-cell archive from flat_per_mutation.csv.

    Only rows where `covered` is truthy are used. A missing/empty/uncovered
    row leaves that cell as None — the search will fill it via the empty-cell
    preference path.
    """
    archive: dict[str, dict[str, Elite | None]] = {
        qid: {m: None for m in MUTATION_TYPES} for qid in pool_by_qid
    }
    csv_path = flat_run_dir / "flat_per_mutation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"flat_per_mutation.csv not found in {flat_run_dir}. "
            f"The flat-grid run may be incomplete. Use --flat-run M to "
            f"pin a different flat_grid_run_NNN."
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = str(row["qid"])
            if qid not in archive:
                continue
            covered = str(row.get("covered", "")).strip().lower() in ("true", "1")
            if not covered:
                continue
            mtype = row.get("mutation_type", "")
            if mtype not in MUTATION_TYPES:
                continue
            try:
                fitness = float(row["best_fitness"])
            except (KeyError, ValueError, TypeError):
                continue
            mq = (row.get("best_mutated_question") or "").strip()
            md = (row.get("best_mutated_distractor") or "").strip()
            if not mq or not md:
                continue
            try:
                log_odds = float(row["best_judge_log_odds_truth"])
            except (KeyError, ValueError, TypeError):
                log_odds = None
            swap = str(row.get("best_swap_flag", "")).strip().lower() in ("true", "1")
            try:
                it = int(row.get("best_iteration") or 0)
            except (ValueError, TypeError):
                it = 0
            pool_row = pool_by_qid[qid]
            archive[qid][mtype] = Elite(
                qid=qid,
                parent_qid=qid,
                mutated_question=mq,
                mutated_distractor=md,
                correct_answer=pool_row["correct_answer"],
                fitness=fitness,
                iteration=it,
                bleu_to_parent=0.0,
                swap=swap,
                judge_log_odds_truth=log_odds,
                judge_correct=None,
            )
    return archive


# ── State helpers ────────────────────────────────────────────────────────────

def _make_initial_state(
    *,
    seed: int,
    pool_qids: list[str],
    pool_csv: Path,
    flat_run_dir: Path,
    per_qid_iterations: int,
    checkpoint_every: int,
) -> dict:
    return {
        "seed": seed,
        "pool_csv": str(pool_csv),
        "pool_size": len(pool_qids),
        "flat_run_used": str(flat_run_dir),
        "per_qid_iterations_cap": per_qid_iterations,
        "checkpoint_every": checkpoint_every,
        "run_start_iter": 0,
        "current_round": 0,
        "rounds_completed": 0,
        "timestamp": _now_iso(),
        "per_qid": [
            {
                "qid": qid,
                "iter_count": 0,
                "status": "active",
                "confirmed_round": None,
                "exhausted_round": None,
                "best_fitness": None,
                "best_mutation_type": None,
                # Tracks `best_cell_elite.iteration` at the time of the last
                # attempted pro checkpoint (None = no checkpoint yet).
                # The gate skips the pro debate when the live elite has the
                # same iteration value, meaning nothing has changed since
                # the last attempt. Decoupled from fitness so it stays
                # correct past search-judge saturation.
                "last_ckpt_elite_iter": None,
            }
            for qid in pool_qids
        ],
    }


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _best_cell(cells: dict[str, Elite | None]) -> tuple[str, Elite] | None:
    occupied = [(m, e) for m, e in cells.items() if e is not None]
    if not occupied:
        return None
    return max(occupied, key=lambda x: x[1].fitness)


def _refresh_best(state: dict, archive: dict[str, dict[str, Elite | None]]) -> None:
    for entry in state["per_qid"]:
        cells = archive.get(entry["qid"])
        if not cells:
            continue
        best = _best_cell(cells)
        if best is None:
            entry["best_fitness"] = None
            entry["best_mutation_type"] = None
        else:
            mt, elite = best
            entry["best_mutation_type"] = mt
            entry["best_fitness"] = round(float(elite.fitness), 6)


# ── 1D selection (mirrors qd/selection.py, single-QID variant) ───────────────

def _select_target_cell_1d(
    cells: dict[str, Elite | None], rng: random.Random
) -> str:
    empties = [m for m, e in cells.items() if e is None]
    occupied = [(m, e) for m, e in cells.items() if e is not None]
    if empties and (not occupied or rng.random() < _P_EMPTY_PREF):
        return rng.choice(empties)
    fits = [e.fitness for _, e in occupied]
    max_fit = max(fits)
    weights = [math.exp((max_fit - f) / _SOFTMAX_T) for f in fits]
    return rng.choices([m for m, _ in occupied], weights=weights, k=1)[0]


def _select_parent_1d(
    cells: dict[str, Elite | None],
    pool_row: dict,
    rng: random.Random,
) -> tuple[dict, str]:
    """Pool-injection (P_POOL_INJECT) or uniform pick from the 4 cells.

    Inverse-count weighting collapses to uniform because each cell holds at
    most one elite for this QID.
    """
    occupied = [(m, e) for m, e in cells.items() if e is not None]
    if not occupied or rng.random() < _P_POOL_INJECT:
        return dict(pool_row), str(pool_row["id"])
    _, elite = rng.choice(occupied)
    inherited = dict(pool_row)
    inherited["question"] = elite.mutated_question
    inherited["negative_answer"] = elite.mutated_distractor
    return inherited, elite.qid


# ── Deterministic per-(seed, qid, iter) RNG and salts ────────────────────────

def _stable_int_seed(qd_seed: int, qid: str, iter_no: int, salt: int = 0) -> int:
    h = hashlib.sha256(
        f"{qd_seed}:{qid}:{iter_no}:{salt}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _make_rng(qd_seed: int, qid: str, iter_no: int) -> random.Random:
    return random.Random(_stable_int_seed(qd_seed, qid, iter_no))


# ── Single search iteration on one QID ───────────────────────────────────────

def _do_per_qid_iteration(
    *,
    qid: str,
    iter_no: int,
    round_no: int,
    archive: dict,
    archive_lock: threading.Lock,
    pool_row: dict,
    target_qtype: str,
    swap: bool,
    qd_seed: int,
    max_retries: int,
    bleu_threshold: float,
    transcripts_writer: JsonlWriter,
    judgements_writer: JsonlWriter,
    failures_writer: JsonlWriter,
    probes_writer: JsonlWriter,
) -> str:
    rng = _make_rng(qd_seed, qid, iter_no)
    with archive_lock:
        cells = dict(archive[qid])
        target_mtype = _select_target_cell_1d(cells, rng)
        parent_row, parent_qid = _select_parent_1d(cells, pool_row, rng)
    target_cell = (target_qtype, target_mtype)

    mutated: dict | None = None
    bleu_score_val: float | None = None
    stage_failures: list[tuple[str, str, str | None, str | None]] = []

    for attempt in range(max_retries):
        m = call_mutator(
            parent_question=parent_row["question"],
            correct_answer=parent_row["correct_answer"],
            parent_distractor=parent_row["negative_answer"],
            mutation_type=target_mtype,
            target_question_type=target_qtype,
            seed=_stable_int_seed(qd_seed, qid, iter_no, salt=attempt + 1),
        )
        if m is None:
            stage_failures.append(("mutator_parse", f"attempt={attempt}", None, None))
            continue
        parent_combined = parent_row["question"] + " " + parent_row["negative_answer"]
        mutated_combined = m["mutated_question"] + " " + m["mutated_distractor"]
        bleu_score_val, ok = passes_filter(
            parent_combined, mutated_combined, bleu_threshold
        )
        if not ok:
            stage_failures.append((
                "bleu_filter",
                f"attempt={attempt} score={bleu_score_val:.3f}",
                m["mutated_question"], m["mutated_distractor"],
            ))
            continue
        detected_type = classify_question_type(
            m["mutated_question"],
            seed=_stable_int_seed(qd_seed, qid, iter_no, salt=100 + attempt),
            target_type=target_qtype,
        )
        if detected_type != target_qtype:
            stage_failures.append((
                "descriptor_mismatch",
                f"attempt={attempt} expected={target_qtype} got={detected_type}",
                m["mutated_question"], m["mutated_distractor"],
            ))
            continue
        valid = call_validator(
            story=pool_row["story"],
            original_question=pool_row["question"],
            correct_answer=pool_row["correct_answer"],
            distractor=pool_row["negative_answer"],
            mutated_question=m["mutated_question"],
            mutated_distractor=m["mutated_distractor"],
            mutation_type=target_mtype,
            seed=_stable_int_seed(qd_seed, qid, iter_no, salt=200 + attempt),
        )
        if not valid:
            stage_failures.append((
                "validity",
                f"attempt={attempt} bleu={bleu_score_val:.3f}",
                m["mutated_question"], m["mutated_distractor"],
            ))
            continue
        mutated = m
        break

    if mutated is None:
        for stage, detail, mq, md in stage_failures:
            failures_writer.write_record(make_failure_record(
                iter_no=iter_no, qid=qid, target_cell=target_cell,
                stage=stage, detail=detail,
                mutated_question=mq, mutated_distractor=md,
                round=round_no,
            ))
        return "failed"

    candidate_row = dict(pool_row)
    candidate_row["question"] = mutated["mutated_question"]
    candidate_row["negative_answer"] = mutated["mutated_distractor"]
    try:
        transcript = run_search_debate(candidate_row, swap)
        judgement = run_search_judge(transcript)
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        failures_writer.write_record(make_failure_record(
            iter_no=iter_no, qid=qid, target_cell=target_cell,
            stage="debate_or_judge", detail=f"{e!r}\n{tb}",
            round=round_no,
        ))
        return "failed"

    transcripts_writer.write_record({
        **transcript,
        "iter": iter_no, "round": round_no, "qid": qid,
        "target_cell": list(target_cell),
        "parent_qid": parent_qid,
        "bleu_to_parent": bleu_score_val,
        "swap": swap,
    })
    judgement_for_log = dict(judgement)
    judgement_for_log.pop("rounds", None)
    judgement_for_log.pop("transcript_str", None)
    judgements_writer.write_record({
        **judgement_for_log,
        "iter": iter_no, "round": round_no, "qid": qid,
        "target_cell": list(target_cell),
        "parent_qid": parent_qid,
        "bleu_to_parent": bleu_score_val,
        "swap": swap,
    })

    fit = fitness_from_judgement(judgement)
    if fit is None:
        return "no_fit"

    candidate = Elite(
        qid=qid,
        parent_qid=str(parent_qid),
        mutated_question=mutated["mutated_question"],
        mutated_distractor=mutated["mutated_distractor"],
        correct_answer=pool_row["correct_answer"],
        fitness=fit,
        iteration=iter_no,
        bleu_to_parent=float(bleu_score_val or 0.0),
        swap=swap,
        judge_log_odds_truth=judgement.get("judge_log_odds_truth"),
        judge_correct=judgement.get("judge_correct"),
    )

    # ── Pre-filter: skip the probe when the candidate cannot win ───────
    # Under the cascaded replacement rule below, a promotable elite is
    # only displaced by another promotable elite with higher fitness.
    # So if the existing cell is already promotable and we do not beat
    # its fitness, no probe outcome can change the decision. Reject
    # without spending the AI-Studio flash call.
    with archive_lock:
        existing_snapshot = archive[qid][target_mtype]
    if (existing_snapshot is not None
            and _probe_is_promotable(existing_snapshot)
            and candidate.fitness <= existing_snapshot.fitness):
        return "rejected"

    # ── Probe candidate ───────────────────────────────────────────────
    # Reuse the Vertex flash-lite verdict from the search-judge call we
    # just ran (its sign mirrors the AI-Studio flash-lite answer at
    # temperature 0.1). Only one new AI Studio call is needed for flash.
    flash_lite_pred = _judge_correct_from_log_odds(
        judgement.get("judge_log_odds_truth")
    )
    flash_pred = _probe_flash_on_transcript(transcript)
    candidate.probe_flash_judge_correct = flash_pred
    candidate.probe_lite_judge_correct  = flash_lite_pred
    candidate.probe_iter = iter_no

    # ── Cascaded replacement (probe-aware) ────────────────────────────
    # Priority: promotable beats non-promotable, regardless of fitness.
    # Within the same promotability class, higher fitness wins. This
    # keeps the cell aligned with the confirmation criterion rather
    # than blindly maximising search-judge log-odds past saturation.
    with archive_lock:
        current = archive[qid][target_mtype]
        if current is None:
            archive[qid][target_mtype] = candidate
            decision = "accepted_empty"
        else:
            cand_promo = _probe_is_promotable(candidate)
            cur_promo  = _probe_is_promotable(current)
            if cand_promo and not cur_promo:
                archive[qid][target_mtype] = candidate
                decision = "accepted_upgrade"
            elif (not cand_promo) and cur_promo:
                decision = "rejected_downgrade"
            elif candidate.fitness > current.fitness:
                archive[qid][target_mtype] = candidate
                decision = ("accepted_fitness_promo" if cand_promo
                            else "accepted_fitness_nonpromo")
            else:
                decision = "rejected_fitness"

    accepted = decision.startswith("accepted")
    probes_writer.write_record({
        "iter": iter_no, "round": round_no, "qid": qid,
        "target_cell": list(target_cell),
        "fitness": float(candidate.fitness),
        "swap": swap,
        "probe": {
            "gemini-2.5-flash":      flash_pred,
            "gemini-2.5-flash-lite": flash_lite_pred,
        },
        "promotable": (flash_pred is False and flash_lite_pred is False),
        "decision": decision,
    })
    return "accepted" if accepted else "rejected"


# ── Pre-confirmation probe (cheap eval-judge prediction on a search transcript)

def _judge_correct_from_log_odds(log_odds: float | None) -> bool | None:
    """Convert a Vertex flash-lite logprob signal into the equivalent
    `judge_correct` boolean. log_odds > 0 ⇒ truth had higher log-odds, so
    the judge picked truth (correct). log_odds < 0 ⇒ deceptive picked.
    None / NaN ⇒ unknown.

    Vertex and AI Studio share model weights, so this is a near-substitute
    for the AI Studio flash-lite verdict on the same transcript. Saves one
    judge call per probe.
    """
    if log_odds is None:
        return None
    try:
        v = float(log_odds)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v > 0.0


def _probe_flash_on_transcript(transcript: dict) -> bool | None:
    """Single AI Studio flash judge call on a transcript. One sleep+retry
    on failure (mirrors `_checkpoint_one_qid` recovery shape). Returns
    True (truth), False (deceptive), or None on both-attempts failure.
    """
    model = "gemini-2.5-flash"
    try:
        j = judge_debate(transcript, judge_profile="eval_judge",
                         use_logprobs=False, model=model)
    except Exception:
        time.sleep(30)
        try:
            j = judge_debate(transcript, judge_profile="eval_judge",
                             use_logprobs=False, model=model)
        except Exception:
            return None
    return j.get("judge_correct")


def _probe_eval_judges_on_transcript(
    transcript: dict,
    *,
    flash_lite_verdict: bool | None = None,
) -> dict[str, bool | None]:
    """Run the eval judges on a search transcript to predict whether the
    pro debate will fool both AI Studio judges.

    Returns {"gemini-2.5-flash": bool|None, "gemini-2.5-flash-lite": bool|None}
    where True == truth, False == deceptive.

    If `flash_lite_verdict` is provided (derived from the existing Vertex
    flash-lite logprob via `_judge_correct_from_log_odds`), the flash-lite
    AI Studio call is skipped. Vertex and AI Studio share model weights,
    so this substitution costs at most a small disagreement rate at
    temperature 0.1 and saves one judge call per probe.

    If `flash_lite_verdict` is None we fall back to calling AI Studio
    flash-lite explicitly so the gate still has a verdict to act on.
    """
    out: dict[str, bool | None] = {m: None for m in CHECKPOINT_JUDGES}
    out["gemini-2.5-flash-lite"] = (
        flash_lite_verdict
        if flash_lite_verdict is not None
        else _probe_flash_lite_fallback(transcript)
    )
    out["gemini-2.5-flash"] = _probe_flash_on_transcript(transcript)
    return out


def _probe_flash_lite_fallback(transcript: dict) -> bool | None:
    """Used when no existing Vertex flash-lite verdict is available, e.g.
    a fresh fallback debate for a missing flat-grid transcript. Single AI
    Studio flash-lite call with the same one-retry shape as `_probe_flash`.
    """
    model = "gemini-2.5-flash-lite"
    try:
        j = judge_debate(transcript, judge_profile="eval_judge",
                         use_logprobs=False, model=model)
    except Exception:
        time.sleep(30)
        try:
            j = judge_debate(transcript, judge_profile="eval_judge",
                             use_logprobs=False, model=model)
        except Exception:
            return None
    return j.get("judge_correct")


def _probe_is_promotable(elite: Elite | None) -> bool:
    """A cell is promotable iff both eval judges have been probed and both
    returned `judge_correct=False` (deceptive won on the search transcript)."""
    if elite is None:
        return False
    return (elite.probe_flash_judge_correct is False
            and elite.probe_lite_judge_correct is False)


def _lazy_probe_best_cell(
    *,
    qid: str,
    archive: dict,
    archive_lock: threading.Lock,
    pool_row: dict,
    swap: bool,
    round_no: int,
    probes_writer: JsonlWriter,
    warm_transcript_index: dict[tuple[str, str], dict],
) -> None:
    """Lazy probe the current best cell.

    Strategy (cheap path first, fallback when missing):
      1. If the flat-grid transcript for (qid, mutation_type) is in the
         warm-start index, use it. Derive the flash-lite verdict from the
         elite's `judge_log_odds_truth` (the existing Vertex flash-lite
         logprob) so we only pay for one AI Studio flash call.
      2. Otherwise, fall back to a fresh flash-lite search debate plus
         both AI Studio judge calls (the original behaviour). Used when
         the flat-grid run was deleted/moved, or the elite came from a
         per-qid search iter whose transcript wasn't cached.

    Updates the elite's probe fields in place under `archive_lock`, only
    if no concurrent worker replaced the elite while we were probing.
    Writes one record to `search_probes.jsonl` either way.
    """
    with archive_lock:
        cells = dict(archive[qid])
    best = _best_cell(cells)
    if best is None:
        return
    mtype, elite = best

    cached = warm_transcript_index.get((qid, mtype))
    if cached is not None:
        # Reuse cached transcript + Vertex flash-lite verdict.
        transcript = cached
        flash_lite_pred = _judge_correct_from_log_odds(elite.judge_log_odds_truth)
        flash_pred = _probe_flash_on_transcript(transcript)
        source = "cached_flat_grid"
    else:
        # Fallback: fresh flash-lite debate + both AI Studio judges.
        candidate_row = dict(pool_row)
        candidate_row["question"] = elite.mutated_question
        candidate_row["negative_answer"] = elite.mutated_distractor
        try:
            transcript = run_search_debate(candidate_row, swap)
        except Exception as e:
            probes_writer.write_record({
                "round": round_no, "qid": qid,
                "target_cell": ["lazy", mtype],
                "fitness": float(elite.fitness),
                "swap": swap, "lazy": True,
                "source": "redebate_failed", "error": repr(e),
            })
            return
        probe = _probe_eval_judges_on_transcript(transcript)
        flash_pred = probe.get("gemini-2.5-flash")
        flash_lite_pred = probe.get("gemini-2.5-flash-lite")
        source = "redebate"

    applied = False
    with archive_lock:
        cur = archive[qid].get(mtype)
        if cur is elite:
            cur.probe_flash_judge_correct = flash_pred
            cur.probe_lite_judge_correct  = flash_lite_pred
            cur.probe_iter = elite.iteration
            applied = True
    probes_writer.write_record({
        "round": round_no, "qid": qid,
        "target_cell": ["lazy", mtype],
        "fitness": float(elite.fitness),
        "swap": swap,
        "probe": {
            "gemini-2.5-flash":      flash_pred,
            "gemini-2.5-flash-lite": flash_lite_pred,
        },
        "promotable": (flash_pred is False and flash_lite_pred is False),
        "applied": applied,
        "lazy": True,
        "source": source,
    })


def _should_checkpoint(
    *,
    qid: str,
    archive: dict,
    archive_lock: threading.Lock,
    state_entry: dict,
) -> tuple[bool, str | None, str | None, float | None]:
    """Decide whether to spend a pro debate on this QID.

    Layered gate:
      Gate 2 (elite-unchanged): the live best-cell elite must be different
              from the one we pro-debated last round. We compare the
              elite's `iteration` value to `last_ckpt_elite_iter`; same
              iteration ⇒ identical framing ⇒ pro outcome would be
              deterministically the same as last round's, so skip.
              First-ever checkpoint (`last_ckpt_elite_iter is None`)
              always passes this gate.
      Gate 1 (probe-promotable): the best cell must be probe-promotable
              (both eval judges already chose deceptive on the cheap
              search transcript). Without this signal, pro is unlikely
              to confirm.

    Returns (should_run, skip_reason, best_mtype, best_fitness).
    """
    with archive_lock:
        cells = dict(archive[qid])
    best = _best_cell(cells)
    if best is None:
        return False, "no_elite", None, None
    best_mtype, best_elite = best
    last_iter = state_entry.get("last_ckpt_elite_iter")
    if last_iter is not None and int(best_elite.iteration) == int(last_iter):
        return False, "elite_unchanged", best_mtype, float(best_elite.fitness)
    if not _probe_is_promotable(best_elite):
        return False, "probe_not_promotable", best_mtype, float(best_elite.fitness)
    return True, None, best_mtype, float(best_elite.fitness)


# ── Checkpoint eval (one pro debate + 2 judge calls per QID) ─────────────────

def _checkpoint_one_qid(
    *,
    qid: str,
    archive: dict,
    archive_lock: threading.Lock,
    pool_row: dict,
    swap: bool,
    round_no: int,
    transcripts_writer: JsonlWriter,
    judge_writers: dict[str, JsonlWriter],
    checkpoint_cells: str = "best",
) -> dict:
    """Pro debate + flash & flash-lite judge on one or more of the QID's cells.

    Cell iteration is controlled by `checkpoint_cells`:
      * "best":     only the highest-fitness cell (1 debate, 2 judge calls).
      * "fallback": cells in fitness order, stop on first confirmation
                    (1-4 debates depending on whether early cells confirm).
      * "all":      every occupied cell (always 4 debates if all are filled).

    The return dict's `tries` list has one entry per cell debated:
      {rank, mutation_type, fitness, judge_correct: {flash: bool|None, lite: bool|None},
       confirmed: bool}
    `confirmed_at_rank` is the rank (1-indexed by fitness) where confirmation
    happened, or None. `best_mutation_type` / `best_fitness` describe the
    confirming cell when confirmed, otherwise the highest-fitness attempted.

    On no-elite / total debate failure: confirmed=False with reason/error.
    """
    with archive_lock:
        cells = dict(archive[qid])
    occupied = sorted(
        ((m, e) for m, e in cells.items() if e is not None),
        key=lambda x: x[1].fitness,
        reverse=True,
    )
    if not occupied:
        return {
            "qid": qid, "round": round_no, "confirmed": False,
            "best_mutation_type": None, "best_fitness": None,
            "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
            "reason": "no_elite",
            "tries": [], "confirmed_at_rank": None,
        }

    cells_to_try = [occupied[0]] if checkpoint_cells == "best" else occupied

    tries: list[dict] = []
    confirmed_overall = False
    confirming_mtype: str | None = None
    confirming_elite: Elite | None = None

    for rank, (mtype, elite) in enumerate(cells_to_try, start=1):
        candidate_row = dict(pool_row)
        candidate_row["question"] = elite.mutated_question
        candidate_row["negative_answer"] = elite.mutated_distractor

        try:
            transcript = run_eval_debate(candidate_row, swap)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            tries.append({
                "rank": rank, "mutation_type": mtype,
                "fitness": float(elite.fitness),
                "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
                "confirmed": False,
                "error": f"eval_debate_failed: {e!r}\n{tb}",
            })
            continue

        transcript_log = dict(transcript)
        transcripts_writer.write_record({
            **transcript_log,
            "qid": qid, "round": round_no, "rank": rank,
            "mutation_type": mtype,
            "fitness": float(elite.fitness),
            "swap": swap,
        })

        per_judge: dict[str, bool | None] = {m: None for m in CHECKPOINT_JUDGES}
        for judge_model in CHECKPOINT_JUDGES:
            j: dict | None = None
            try:
                j = judge_debate(
                    transcript, judge_profile="eval_judge",
                    use_logprobs=False, model=judge_model,
                )
            except Exception as e1:
                time.sleep(30)
                try:
                    j = judge_debate(
                        transcript, judge_profile="eval_judge",
                        use_logprobs=False, model=judge_model,
                    )
                except Exception as e2:
                    judge_writers[judge_model].write_record({
                        "qid": qid, "round": round_no, "rank": rank,
                        "mutation_type": mtype, "swap": swap,
                        "error": f"first={e1!r}; retry={e2!r}",
                        "judge_correct": None,
                    })
                    continue
            per_judge[judge_model] = j.get("judge_correct")
            rec = dict(j)
            rec.pop("rounds", None)
            rec.pop("transcript_str", None)
            judge_writers[judge_model].write_record({
                **rec,
                "qid": qid, "round": round_no, "rank": rank,
                "mutation_type": mtype,
                "fitness": float(elite.fitness),
                "swap": swap,
            })

        cell_confirmed = (
            per_judge[CHECKPOINT_JUDGES[0]] is False
            and per_judge[CHECKPOINT_JUDGES[1]] is False
        )
        tries.append({
            "rank": rank, "mutation_type": mtype,
            "fitness": float(elite.fitness),
            "judge_correct": per_judge,
            "confirmed": cell_confirmed,
        })

        if cell_confirmed and not confirmed_overall:
            confirmed_overall = True
            confirming_mtype = mtype
            confirming_elite = elite
            if checkpoint_cells == "fallback":
                break  # stop at first confirmation; "all" continues

    # Report the confirming cell when confirmed; otherwise the strongest attempted.
    if confirmed_overall and confirming_elite is not None:
        report_mtype = confirming_mtype
        report_fit = float(confirming_elite.fitness)
        report_judge = next(
            (t["judge_correct"] for t in tries if t.get("confirmed")),
            {m: None for m in CHECKPOINT_JUDGES},
        )
    else:
        head = tries[0] if tries else None
        report_mtype = head["mutation_type"] if head else cells_to_try[0][0]
        report_fit = head["fitness"] if head else float(cells_to_try[0][1].fitness)
        report_judge = head["judge_correct"] if head else {m: None for m in CHECKPOINT_JUDGES}

    return {
        "qid": qid, "round": round_no, "confirmed": confirmed_overall,
        "best_mutation_type": report_mtype,
        "best_fitness": report_fit,
        "judge_correct": report_judge,
        "tries": tries,
        "confirmed_at_rank": next(
            (t["rank"] for t in tries if t.get("confirmed")), None
        ),
        "n_cells_debated": len(tries),
    }


# ── Run-dir resolution ───────────────────────────────────────────────────────

def _per_qid_run_number(name: str) -> int | None:
    if not name.startswith("per_qid_run_"):
        return None
    suffix = name[len("per_qid_run_"):]
    return int(suffix) if suffix.isdigit() else None


def _resume_or_new_run_dir(seed: int) -> tuple[Path, bool]:
    """Resume into the highest-numbered per_qid_run_NNN that has a state file,
    otherwise create per_qid_run_{N+1}."""
    seed_root = map_seed_dir(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    existing = [
        d for d in seed_root.iterdir()
        if d.is_dir() and _per_qid_run_number(d.name) is not None
    ]
    existing.sort(key=lambda d: _per_qid_run_number(d.name), reverse=True)
    for d in existing:
        if (d / "per_qid_state.json").exists():
            return d, True
    next_num = (_per_qid_run_number(existing[0].name) + 1) if existing else 1
    new_dir = seed_root / f"per_qid_run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir, False


def _maybe_fork_completed_run(run_dir: Path, seed: int, new_cap: int) -> Path:
    """If run_dir is fully complete and new_cap exceeds the stored cap, copy
    only the two anchor files (state + archive) into the next per_qid_run_NNN
    and return that new path. The original run_dir is left intact as the
    clean log record for its own rounds. Returns run_dir unchanged otherwise."""
    state_file = run_dir / "per_qid_state.json"
    archive_file = run_dir / "per_qid_archive.json"
    if not (state_file.exists() and archive_file.exists()):
        return run_dir
    with open(state_file, encoding="utf-8") as f:
        st = json.load(f)
    old_cap = int(st.get("per_qid_iterations_cap", new_cap))
    if new_cap <= old_cap:
        return run_dir
    all_done = all(
        e["status"] in ("confirmed", "exhausted")
        for e in st.get("per_qid", [])
    )
    if not all_done:
        return run_dir
    seed_root = map_seed_dir(seed)
    existing = sorted(
        [d for d in seed_root.iterdir()
         if d.is_dir() and _per_qid_run_number(d.name) is not None],
        key=lambda d: _per_qid_run_number(d.name),
    )
    next_num = (_per_qid_run_number(existing[-1].name) + 1) if existing else 1
    new_dir = seed_root / f"per_qid_run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_file, new_dir / "per_qid_archive.json")
    # Patch the copied state: reset round counter and record where this run
    # starts so the target formula (run_start_iter + round * checkpoint_every)
    # gives the right absolute iter targets regardless of checkpoint_every.
    new_state = dict(st)
    new_state["run_start_iter"] = old_cap
    new_state["rounds_completed"] = 0
    new_state["current_round"] = 0
    new_state["timestamp"] = _now_iso()
    _atomic_write_json(new_dir / "per_qid_state.json", new_state)
    print(f"[per_qid] {run_dir.name} complete; forked to {new_dir.name} "
          f"(cap {old_cap}→{new_cap})")
    return new_dir


# ── Main orchestration ──────────────────────────────────────────────────────

def run_per_qid_mode(args: argparse.Namespace) -> int:
    seed = args.seed
    config.set_seed(seed)
    config.validate_llm_credentials()
    try:
        from core.gemini_client import format_pool_diagnostics
        print(format_pool_diagnostics(workers=args.workers))
    except Exception:
        pass

    pool_csv = pool_csv_for_seed(seed)
    pool_by_qid, full_pool_qids = _load_pool(pool_csv)

    # Exclude QIDs already confirmed by an earlier pipeline stage (flat-grid or
    # a prior per_qid_run). They live in seed_N/best_framings.csv and don't
    # need any further search budget.
    already_confirmed = _load_already_confirmed_qids(seed)
    pool_qids = [q for q in full_pool_qids if q not in already_confirmed]
    pool_by_qid_active = {q: pool_by_qid[q] for q in pool_qids}
    print(f"[per_qid] seed={seed}  pool={pool_csv}  "
          f"full={len(full_pool_qids)}  already_confirmed={len(already_confirmed)}  "
          f"active={len(pool_qids)}")

    if not pool_qids:
        print(f"[per_qid] seed={seed}: every QID is already confirmed — nothing to do.")
        return 0

    flat_run_dir = _resolve_flat_run(seed, args.flat_run)
    print(f"[per_qid] warm-start source: {flat_run_dir}")

    # One-time scan of flat_transcripts.jsonl so lazy probes can reuse
    # the existing flash-lite debate instead of running a fresh one. An
    # empty index just means every lazy probe will take the redebate
    # fallback path.
    warm_transcript_index = _build_warm_start_transcript_index(flat_run_dir)
    print(f"[per_qid] cached flat-grid transcripts: "
          f"{len(warm_transcript_index)} (cells with reusable transcripts)")

    run_dir, is_resume = _resume_or_new_run_dir(seed)
    if is_resume:
        run_dir = _maybe_fork_completed_run(run_dir, seed, args.per_qid_iterations)
    archive_path = run_dir / "per_qid_archive.json"
    state_path   = run_dir / "per_qid_state.json"
    search_dir   = run_dir / "search"
    ckpt_dir     = run_dir / "checkpoint"
    search_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = map_seed_dir(seed) / "per_qid_summary.json"

    archive_lock = threading.Lock()
    state_lock   = threading.Lock()

    if is_resume and archive_path.exists() and state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        with open(archive_path, encoding="utf-8") as f:
            archive = _archive_from_jsonable(json.load(f))
        # Reconcile: if best_framings.csv has grown since this run was started
        # (e.g. via a parallel run on the same seed), retroactively mark those
        # QIDs as already-confirmed so we don't keep iterating on them. Also
        # add any pool QIDs missing from state (extremely unlikely, but cheap).
        existing_qids = {e["qid"] for e in state["per_qid"]}
        for qid in pool_qids:
            if qid not in existing_qids:
                state["per_qid"].append({
                    "qid": qid, "iter_count": 0, "status": "active",
                    "confirmed_round": None, "exhausted_round": None,
                    "best_fitness": None, "best_mutation_type": None,
                    "last_ckpt_elite_iter": None,
                })
                archive[qid] = {m: None for m in MUTATION_TYPES}
        # Backfill `last_ckpt_elite_iter` for older state files. None means
        # the QID has never been pro-checkpointed under the elite-identity
        # gate, so the first round will fall through to the probe-only
        # decision. (Older files may also carry a now-unused
        # `last_ckpt_fitness` field; we leave it untouched so a downgrade
        # roll-back can still read it.)
        for entry in state["per_qid"]:
            entry.setdefault("last_ckpt_elite_iter", None)
        for entry in state["per_qid"]:
            if entry["status"] == "active" and entry["qid"] in already_confirmed:
                entry["status"] = "confirmed"
                entry["confirmed_round"] = entry.get("confirmed_round") or 0
        # If the iteration cap was raised, re-activate QIDs that were only
        # exhausted because they hit the old cap (not because they confirmed).
        old_cap = int(state.get("per_qid_iterations_cap", args.per_qid_iterations))
        if args.per_qid_iterations > old_cap:
            reactivated = 0
            for entry in state["per_qid"]:
                if (entry["status"] == "exhausted"
                        and entry["iter_count"] < args.per_qid_iterations):
                    entry["status"] = "active"
                    entry["exhausted_round"] = None
                    reactivated += 1
            if reactivated:
                print(f"[per_qid] cap raised {old_cap}→{args.per_qid_iterations}: "
                      f"re-activated {reactivated} exhausted QIDs")
        state["per_qid_iterations_cap"] = args.per_qid_iterations
        # Rewind any partial round: re-enter round = rounds_completed + 1.
        state["current_round"] = int(state.get("rounds_completed", 0))
        print(f"[per_qid] resuming {run_dir.name}: "
              f"rounds_completed={state.get('rounds_completed', 0)}, "
              f"active="
              f"{sum(1 for e in state['per_qid'] if e['status'] == 'active')}")
    else:
        archive = _warm_start_archive(flat_run_dir, pool_by_qid_active)
        state = _make_initial_state(
            seed=seed, pool_qids=pool_qids, pool_csv=pool_csv,
            flat_run_dir=flat_run_dir,
            per_qid_iterations=args.per_qid_iterations,
            checkpoint_every=args.checkpoint_every,
        )
        _refresh_best(state, archive)
        _save_archive(archive, archive_path, archive_lock)
        _save_state(state, state_path, state_lock)
        print(f"[per_qid] fresh run at {run_dir.name}; "
              f"warm-started cells="
              f"{sum(1 for cells in archive.values() for e in cells.values() if e is not None)}"
              f"/{len(pool_qids) * len(MUTATION_TYPES)}")

    state_by_qid: dict[str, dict] = {e["qid"]: e for e in state["per_qid"]}
    run_start_iter = int(state.get("run_start_iter", 0))

    search_t  = JsonlWriter(search_dir / "search_transcripts.jsonl")
    search_j  = JsonlWriter(search_dir / "search_judgements.jsonl")
    search_f  = JsonlWriter(search_dir / "search_failures.jsonl")
    search_p  = JsonlWriter(search_dir / "search_probes.jsonl")
    ckpt_t    = JsonlWriter(ckpt_dir / "checkpoint_transcripts.jsonl")
    judge_writers: dict[str, JsonlWriter] = {
        m: JsonlWriter(ckpt_dir / f"checkpoint_judgements_{_JUDGE_FILE_TAG[m]}.jsonl")
        for m in CHECKPOINT_JUDGES
    }
    ckpt_summary_path = ckpt_dir / "checkpoint_summary.json"
    if ckpt_summary_path.exists():
        with open(ckpt_summary_path, encoding="utf-8") as f:
            ckpt_summary = json.load(f)
    else:
        ckpt_summary = {"seed": seed, "rounds": []}

    swap_map = baseline_swap_for_seed(seed)
    qd_seed_for_iters = config.RANDOM_SEED
    max_retries     = config.QD_MAX_RETRIES_PER_ITER
    bleu_threshold  = config.QD_BLEU_THRESHOLD

    t_start = time.time()

    try:
        while True:
            active = [e["qid"] for e in state["per_qid"] if e["status"] == "active"]
            if not active:
                print("[per_qid] no active QIDs — stopping.")
                break

            round_no = int(state.get("current_round", 0)) + 1
            state["current_round"] = round_no
            print(f"[per_qid] ── round {round_no} ─────────────────────"
                  f"  active={len(active)}")

            # ── Search phase ──────────────────────────────────────────────
            tasks: list[tuple[str, int]] = []
            target_after_round = run_start_iter + round_no * args.checkpoint_every
            cap = args.per_qid_iterations
            target = min(target_after_round, cap)
            for qid in active:
                already = state_by_qid[qid]["iter_count"]
                iters_this_round = max(0, target - already)
                for k in range(iters_this_round):
                    tasks.append((qid, already + k + 1))

            if tasks:
                print(f"[per_qid] search: dispatching {len(tasks)} iters "
                      f"across {args.workers} worker(s)")
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = {}
                    for qid, iter_no in tasks:
                        pool_row = pool_by_qid[qid]
                        target_qtype = pool_row.get("question_type") or "Identity"
                        swap = bool(swap_map.get(qid, False))
                        fut = ex.submit(
                            _do_per_qid_iteration,
                            qid=qid, iter_no=iter_no, round_no=round_no,
                            archive=archive, archive_lock=archive_lock,
                            pool_row=pool_row, target_qtype=target_qtype, swap=swap,
                            qd_seed=qd_seed_for_iters,
                            max_retries=max_retries,
                            bleu_threshold=bleu_threshold,
                            transcripts_writer=search_t,
                            judgements_writer=search_j,
                            failures_writer=search_f,
                            probes_writer=search_p,
                        )
                        futures[fut] = (qid, iter_no)
                    done = 0
                    for fut in as_completed(futures):
                        qid, iter_no = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"[per_qid] ITERATION CRASH qid={qid} iter={iter_no}: {e!r}")
                        with state_lock:
                            entry = state_by_qid[qid]
                            if iter_no > entry["iter_count"]:
                                entry["iter_count"] = iter_no
                        done += 1
                        if done % 50 == 0:
                            print(f"[per_qid] search: {done}/{len(tasks)} iters done")

            _refresh_best(state, archive)
            _save_archive(archive, archive_path, archive_lock)
            _save_state(state, state_path, state_lock)

            # ── Lazy-probe phase ──────────────────────────────────────────
            # Probe any active QID whose current best cell has never been
            # probed (typically a warm-start cell the search has not
            # surpassed). Without this pass the gate below would skip
            # those cells permanently as "probe_not_promotable".
            lazy_targets: list[str] = []
            for qid in active:
                with archive_lock:
                    cells = dict(archive[qid])
                best = _best_cell(cells)
                if best is None:
                    continue
                if best[1].probe_iter is None:
                    lazy_targets.append(qid)
            if lazy_targets:
                print(f"[per_qid] lazy probe: {len(lazy_targets)} cell(s) "
                      f"need first-time probe")
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    lp_futures = {}
                    for qid in lazy_targets:
                        pool_row = pool_by_qid[qid]
                        swap = bool(swap_map.get(qid, False))
                        lp_futures[ex.submit(
                            _lazy_probe_best_cell,
                            qid=qid, archive=archive, archive_lock=archive_lock,
                            pool_row=pool_row, swap=swap, round_no=round_no,
                            probes_writer=search_p,
                            warm_transcript_index=warm_transcript_index,
                        )] = qid
                    for fut in as_completed(lp_futures):
                        try:
                            fut.result()
                        except Exception as e:
                            qid = lp_futures[fut]
                            print(f"[per_qid] LAZY PROBE CRASH qid={qid}: {e!r}")
                _save_archive(archive, archive_path, archive_lock)

            # ── Gate phase ────────────────────────────────────────────────
            # Decide which active QIDs spend a pro debate this round.
            #   - probe_not_promotable: skip (cheap probe says eval judges
            #     would not flip on this framing).
            #   - elite_unchanged: skip (the live best-cell elite is the
            #     same one we already pro-debated last round, so the pro
            #     outcome is deterministically identical).
            to_checkpoint: list[str] = []
            ckpt_results: list[dict] = []
            confirmed_now: list[str] = []
            skip_counts = {
                "no_elite": 0,
                "probe_not_promotable": 0,
                "elite_unchanged": 0,
            }
            for qid in active:
                decision, reason, mtype, fit = _should_checkpoint(
                    qid=qid, archive=archive, archive_lock=archive_lock,
                    state_entry=state_by_qid[qid],
                )
                if decision:
                    to_checkpoint.append(qid)
                else:
                    skip_counts[reason] = skip_counts.get(reason, 0) + 1
                    ckpt_results.append({
                        "qid": qid, "round": round_no,
                        "confirmed": False, "skipped": True,
                        "skip_reason": reason,
                        "best_mutation_type": mtype,
                        "best_fitness": fit,
                        "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
                        "tries": [], "confirmed_at_rank": None,
                        "n_cells_debated": 0,
                    })

            # ── Checkpoint phase ──────────────────────────────────────────
            print(f"[per_qid] checkpoint: dispatching {len(to_checkpoint)}/"
                  f"{len(active)} pro debates  "
                  f"(skipped: not_promotable={skip_counts['probe_not_promotable']}, "
                  f"elite_unchanged={skip_counts['elite_unchanged']}, "
                  f"no_elite={skip_counts['no_elite']}) "
                  f"across {args.workers} worker(s)")
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {}
                for qid in to_checkpoint:
                    pool_row = pool_by_qid[qid]
                    swap = bool(swap_map.get(qid, False))
                    fut = ex.submit(
                        _checkpoint_one_qid,
                        qid=qid, archive=archive, archive_lock=archive_lock,
                        pool_row=pool_row, swap=swap, round_no=round_no,
                        transcripts_writer=ckpt_t,
                        judge_writers=judge_writers,
                        checkpoint_cells=getattr(args, "checkpoint_cells", "best"),
                    )
                    futures[fut] = qid
                done = 0
                for fut in as_completed(futures):
                    qid = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        tb = traceback.format_exc(limit=2)
                        print(f"[per_qid] CHECKPOINT CRASH qid={qid}: {e!r}\n{tb}")
                        res = {
                            "qid": qid, "round": round_no, "confirmed": False,
                            "error": f"{e!r}",
                        }
                    ckpt_results.append(res)
                    if res.get("confirmed"):
                        with state_lock:
                            entry = state_by_qid[qid]
                            entry["status"] = "confirmed"
                            entry["confirmed_round"] = round_no
                            entry["confirmed_at_rank"] = res.get("confirmed_at_rank")
                            entry["n_cells_debated_at_confirm"] = res.get("n_cells_debated")
                            confirmed_iter_count = int(entry["iter_count"])
                        confirmed_now.append(qid)
                        # Append to seed_N/best_framings.csv so the manifest
                        # stays the single source of truth across flat-grid +
                        # per-QID. Serialised by this main loop (no extra lock
                        # needed — futures only return data).
                        best_mtype = res.get("best_mutation_type")
                        if best_mtype:
                            with archive_lock:
                                elite_for_append = archive[qid][best_mtype]
                            if elite_for_append is not None:
                                try:
                                    _append_confirmation_to_best_framings(
                                        seed=seed, qid=qid,
                                        elite=elite_for_append,
                                        best_mtype=best_mtype,
                                        pool_row=pool_by_qid_active[qid],
                                        baseline_swap=swap_map,
                                        eval_swap=bool(swap_map.get(qid, False)),
                                        iter_count=confirmed_iter_count,
                                        source_run_name=run_dir.name,
                                    )
                                except Exception as e:
                                    print(f"[per_qid] WARN: failed to append "
                                          f"qid={qid} to best_framings.csv: {e!r}")
                    done += 1
                    if done % 20 == 0:
                        print(f"[per_qid] checkpoint: {done}/{len(to_checkpoint)} done")

            # Update last_ckpt_elite_iter for every QID we considered this
            # round (skipped + dispatched). The gate compares this to the
            # live elite's iteration next round so the same elite is never
            # pro-debated twice in a row. Confirmed QIDs leave the active
            # set right after so the field is irrelevant for them, but
            # writing it keeps the schema consistent.
            with state_lock, archive_lock:
                for qid in active:
                    cells = archive[qid]
                    occ = [e for e in cells.values() if e is not None]
                    if not occ:
                        continue
                    best_elite = max(occ, key=lambda e: e.fitness)
                    state_by_qid[qid]["last_ckpt_elite_iter"] = int(best_elite.iteration)

            # ── Budget check ──────────────────────────────────────────────
            exhausted_now: list[str] = []
            with state_lock:
                for entry in state["per_qid"]:
                    if (entry["status"] == "active"
                            and entry["iter_count"] >= args.per_qid_iterations):
                        entry["status"] = "exhausted"
                        entry["exhausted_round"] = round_no
                        exhausted_now.append(entry["qid"])

            state["rounds_completed"] = round_no
            state["timestamp"] = _now_iso()
            _refresh_best(state, archive)
            _save_state(state, state_path, state_lock)

            n_confirmed_total = sum(1 for e in state["per_qid"] if e["status"] == "confirmed")
            n_exhausted_total = sum(1 for e in state["per_qid"] if e["status"] == "exhausted")
            n_active_after    = sum(1 for e in state["per_qid"] if e["status"] == "active")

            # Cost + rank stats for this round (drives the smart-fallback vs
            # best-only A/B comparison).
            rank_hist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
            debates_this_round = 0
            for r in ckpt_results:
                debates_this_round += int(r.get("n_cells_debated") or 0)
                rk = r.get("confirmed_at_rank")
                if isinstance(rk, int):
                    rank_hist[rk] = rank_hist.get(rk, 0) + 1
            n_best_only_equiv_round = rank_hist.get(1, 0)

            ckpt_summary["rounds"].append({
                "round": round_no,
                "n_active_before":   len(active),
                "confirmed":         sorted(confirmed_now),
                "exhausted":         sorted(exhausted_now),
                "n_confirmed_total": n_confirmed_total,
                "n_exhausted_total": n_exhausted_total,
                "n_active_after":    n_active_after,
                "pro_debates_this_round": debates_this_round,
                "rank_at_confirm_hist":   rank_hist,
                "n_best_only_equiv_round": n_best_only_equiv_round,
                "gate_dispatched":   len(to_checkpoint),
                "gate_skipped":      skip_counts,
                "gate_skipped_total": sum(skip_counts.values()),
                "lazy_probes":       len(lazy_targets),
                "results":           ckpt_results,
            })
            _atomic_write_json(ckpt_summary_path, ckpt_summary)

            # Cumulative A/B numbers for at-a-glance comparison
            n_best_only_cum = sum(
                1 for e in state["per_qid"]
                if e.get("confirmed_at_rank") == 1
            )
            print(f"[per_qid] round {round_no} done: "
                  f"+{len(confirmed_now)} confirmed (ranks={rank_hist}), "
                  f"+{len(exhausted_now)} exhausted, active={n_active_after}, "
                  f"total_confirmed={n_confirmed_total}, "
                  f"best_only_equiv={n_best_only_cum}, "
                  f"pro_debates_this_round={debates_this_round}, "
                  f"gate_dispatched={len(to_checkpoint)}/{len(active)}, "
                  f"gate_skipped="
                  f"{skip_counts['probe_not_promotable']}probe/"
                  f"{skip_counts['elite_unchanged']}stale, "
                  f"lazy_probes={len(lazy_targets)}")
            try:
                slack(
                    f":dart: per_qid seed={seed} round={round_no}: "
                    f"+{len(confirmed_now)} confirmed "
                    f"(rank1={rank_hist[1]}/2={rank_hist[2]}/3={rank_hist[3]}/4={rank_hist[4]}), "
                    f"+{len(exhausted_now)} exhausted, active={n_active_after}, "
                    f"best_only_equiv={n_best_only_cum}, debates={debates_this_round}, "
                    f"gate={len(to_checkpoint)}/{len(active)} (skip "
                    f"{skip_counts['probe_not_promotable']}probe/"
                    f"{skip_counts['elite_unchanged']}stale), "
                    f"lazy={len(lazy_targets)}"
                )
            except Exception:
                pass

    finally:
        for w in (search_t, search_j, search_f, search_p, ckpt_t,
                  *judge_writers.values()):
            try:
                w.close()
            except Exception:
                pass

    # ── Final summary ────────────────────────────────────────────────────────
    n_confirmed = sum(1 for e in state["per_qid"] if e["status"] == "confirmed")
    n_exhausted = sum(1 for e in state["per_qid"] if e["status"] == "exhausted")
    n_active    = sum(1 for e in state["per_qid"] if e["status"] == "active")

    # A/B numbers: smart-fallback total vs what best-only would have caught
    rank_hist_total: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in state["per_qid"]:
        rk = e.get("confirmed_at_rank")
        if isinstance(rk, int):
            rank_hist_total[rk] = rank_hist_total.get(rk, 0) + 1
    n_best_only_equiv = rank_hist_total.get(1, 0)
    total_pro_debates = sum(
        int(rr.get("pro_debates_this_round") or 0)
        for rr in ckpt_summary.get("rounds", [])
    )
    # Best-only would do (1 pro debate per active QID per round)
    best_only_pro_debates = sum(
        int(rr.get("n_active_before") or 0)
        for rr in ckpt_summary.get("rounds", [])
    )

    elapsed = time.time() - t_start
    summary = {
        "seed":                    seed,
        "pool_csv":                str(pool_csv),
        "pool_size_full":          len(full_pool_qids),
        "pool_size_active":        len(state["per_qid"]),
        "already_confirmed":       sorted(_load_already_confirmed_qids(seed), key=lambda q: int(q) if q.isdigit() else q),
        "flat_run_used":           str(flat_run_dir),
        "run_dir":                 str(run_dir),
        "checkpoint_cells":        getattr(args, "checkpoint_cells", "best"),
        "per_qid_iterations_cap":  args.per_qid_iterations,
        "checkpoint_every":        args.checkpoint_every,
        "rounds_completed":        state.get("rounds_completed", 0),
        "confirmed":               n_confirmed,
        "exhausted":               n_exhausted,
        "remaining_active":        n_active,
        "rank_at_confirm_hist":    rank_hist_total,
        "n_best_only_equiv":       n_best_only_equiv,
        "extra_catches_vs_best":   n_confirmed - n_best_only_equiv,
        "pro_debates_total":       total_pro_debates,
        "pro_debates_best_only_equiv": best_only_pro_debates,
        "elapsed_sec":             round(elapsed, 1),
        "timestamp":               _now_iso(),
        "per_qid":                 state["per_qid"],
    }
    _atomic_write_json(summary_path, summary)
    print(f"[per_qid] DONE. confirmed={n_confirmed}/{len(pool_qids)} "
          f"(rank1={rank_hist_total[1]}, rank2={rank_hist_total[2]}, "
          f"rank3={rank_hist_total[3]}, rank4={rank_hist_total[4]}), "
          f"exhausted={n_exhausted}, active_left={n_active}.")
    if getattr(args, "checkpoint_cells", "best") != "best":
        extra = n_confirmed - n_best_only_equiv
        cost_mult = (total_pro_debates / best_only_pro_debates) if best_only_pro_debates else 0
        print(f"[per_qid] A/B: smart-fallback caught {n_confirmed} vs "
              f"best-only would have caught {n_best_only_equiv} "
              f"(+{extra} extra). Cost: {total_pro_debates} pro debates "
              f"vs {best_only_pro_debates} best-only ({cost_mult:.2f}x).")
    print(f"[per_qid] summary -> {summary_path}")
    try:
        slack(
            f":white_check_mark: per_qid done — seed={seed}\n"
            f"confirmed: {n_confirmed}/{len(pool_qids)} "
            f"(best_only_equiv={n_best_only_equiv}, +{n_confirmed - n_best_only_equiv} extra), "
            f"exhausted: {n_exhausted}, active_left: {n_active}\n"
            f"pro_debates: {total_pro_debates} (vs {best_only_pro_debates} best-only)\n"
            f"elapsed: {elapsed:.0f}s"
        )
    except Exception:
        pass
    return 0
