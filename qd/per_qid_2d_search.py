"""Per-QID 2D MAP-Elites with eval-gated removal (Stage 3).

Each unconfirmed QID from stage 2 (per_qid_search) gets its own independent
5 question_type x 4 mutation_type archive (20 cells). The QID's original
question-type row is pre-seeded from stage 2's 4 mutation cells, and the
other 16 cells are warm-started by asking the mutator to cross into the
target question type while applying the target mutation type. The search
then runs 2D MAP-Elites per QID with all 20 cells competing as parents
and targets.

Cells are eligible for the pro checkpoint only when the same two gates
from stage 2 pass:

  Gate 1 (probe-promotable): the best cell's eager-probed flash + flash-lite
                             verdicts both say deceptive won.
  Gate 2 (elite-unchanged): the best cell's elite.iteration has moved
                            past last_ckpt_elite_iter.

Confirmed QIDs are appended to seed_N/best_framings.csv with a
`confirming_question_type` field set to the confirming cell's question
type. When that field differs from the QID's `original_question_type`,
the confirmation came from a type-crossed row.

Layout (per seed, per run):

    qd_results/map_results/seed_{N}/
      per_qid_2d_summary.json
      per_qid_2d_run_{NNN}/
        per_qid_2d_archive.json
        per_qid_2d_state.json
        search/  search_{transcripts,judgements,failures,probes}.jsonl
                 warm_start_log.jsonl
        checkpoint/  checkpoint_{transcripts,judgements_flash,
                                 judgements_flash-lite}.jsonl
                     checkpoint_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
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

from qd.archive import Archive, Elite, MUTATION_TYPES, QUESTION_TYPES
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
from qd.selection import select_parent, select_target_cell
from qd.validator import call_validator, classify_question_type

# Reuse helpers from stage 2 unchanged. These are cell-shape-agnostic.
from qd.per_qid_search import (
    CHECKPOINT_JUDGES,
    _JUDGE_FILE_TAG,
    _atomic_write_json,
    _append_confirmation_to_best_framings,
    _build_warm_start_transcript_index,
    _judge_correct_from_log_odds,
    _lazy_probe_best_cell,
    _load_already_confirmed_qids,
    _load_pool,
    _now_iso,
    _probe_flash_on_transcript,
    _probe_is_promotable,
    _resolve_flat_run,
)


# ── Archive (de)serialisation ────────────────────────────────────────────────
#
# Per-QID shape: { "qids": { qid: Archive.to_jsonable() } }. Each per-QID
# block matches the on-disk layout produced by run_qd_search.py for the
# full 5x4 archive, so the same downstream readers work.

def _archive_to_jsonable(archive: dict[str, Archive]) -> dict:
    return {"qids": {qid: a.to_jsonable() for qid, a in archive.items()}}


def _archive_from_jsonable(data: dict) -> dict[str, Archive]:
    out: dict[str, Archive] = {}
    for qid, payload in data.get("qids", {}).items():
        a = Archive()
        a.iter_count = int(payload.get("iter_count", 0))
        for entry in payload.get("cells", []):
            cell = tuple(entry["cell"])  # (qtype, mtype)
            elite_d = entry.get("elite")
            a.cells[cell] = Elite.from_dict(elite_d) if elite_d is not None else None
        out[qid] = a
    return out


def _save_archive(
    archive: dict[str, Archive],
    path: Path,
    lock: threading.Lock,
) -> None:
    with lock:
        data = _archive_to_jsonable(archive)
    _atomic_write_json(path, data)


def _save_state(state: dict, path: Path, state_lock: threading.Lock) -> None:
    with state_lock:
        snapshot = json.loads(json.dumps(state))
    _atomic_write_json(path, snapshot)


# ── Stage-2 source loading ───────────────────────────────────────────────────

def _latest_per_qid_run_dir(seed: int, pin: int | None = None) -> Path:
    seed_root = map_seed_dir(seed)
    if pin is not None:
        cand = seed_root / f"per_qid_run_{pin:03d}"
        if not cand.exists():
            raise FileNotFoundError(f"Stage-2 source run not found: {cand}")
        return cand
    runs = sorted(
        [d for d in seed_root.iterdir()
         if d.is_dir() and d.name.startswith("per_qid_run_")],
        key=lambda d: d.name,
    )
    if not runs:
        raise FileNotFoundError(
            f"No per_qid_run_NNN found under {seed_root}/ — stage 3 "
            f"requires a completed stage-2 run to warm-start from."
        )
    return runs[-1]


def _load_stage2_archive(stage2_run: Path) -> dict[str, dict[str, Elite | None]]:
    """Read stage-2's per_qid_archive.json (1D shape: {qid: {mtype: Elite|None}})."""
    archive_path = stage2_run / "per_qid_archive.json"
    if not archive_path.exists():
        raise FileNotFoundError(f"Stage-2 archive missing: {archive_path}")
    data = json.loads(archive_path.read_text())
    out: dict[str, dict[str, Elite | None]] = {}
    for qid, payload in data.get("qids", {}).items():
        cells_d = payload.get("cells", {})
        out[qid] = {
            mt: (Elite.from_dict(cells_d[mt]) if cells_d.get(mt) is not None else None)
            for mt in MUTATION_TYPES
        }
    return out


def _load_stage2_state(stage2_run: Path) -> dict:
    sp = stage2_run / "per_qid_state.json"
    if not sp.exists():
        raise FileNotFoundError(f"Stage-2 state missing: {sp}")
    return json.loads(sp.read_text())


# ── Best-of-stage-2 per QID ──────────────────────────────────────────────────

def _stage2_best_elite(stage2_cells: dict[str, Elite | None]) -> Elite | None:
    """Highest-fitness elite across this QID's 4 stage-2 mutation cells."""
    occupied = [e for e in stage2_cells.values() if e is not None]
    if not occupied:
        return None
    return max(occupied, key=lambda e: e.fitness)


# ── Run-dir resolution (stage-3 numbering) ───────────────────────────────────

def _per_qid_2d_run_number(name: str) -> int | None:
    if not name.startswith("per_qid_2d_run_"):
        return None
    suffix = name[len("per_qid_2d_run_"):]
    return int(suffix) if suffix.isdigit() else None


def _resume_or_new_run_dir(seed: int) -> tuple[Path, bool]:
    """Resume into the highest-numbered per_qid_2d_run_NNN with a state file,
    otherwise create per_qid_2d_run_{N+1}."""
    seed_root = map_seed_dir(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    existing = [
        d for d in seed_root.iterdir()
        if d.is_dir() and _per_qid_2d_run_number(d.name) is not None
    ]
    existing.sort(key=lambda d: _per_qid_2d_run_number(d.name), reverse=True)
    for d in existing:
        if (d / "per_qid_2d_state.json").exists():
            return d, True
    next_num = (_per_qid_2d_run_number(existing[0].name) + 1) if existing else 1
    new_dir = seed_root / f"per_qid_2d_run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir, False


def _maybe_fork_completed_run(run_dir: Path, seed: int, new_cap: int) -> Path:
    """If run_dir is fully complete and new_cap > stored cap, fork to
    per_qid_2d_run_{N+1} with state + archive copied forward and round
    counters reset. Mirrors stage-2 behaviour."""
    state_file = run_dir / "per_qid_2d_state.json"
    archive_file = run_dir / "per_qid_2d_archive.json"
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
         if d.is_dir() and _per_qid_2d_run_number(d.name) is not None],
        key=lambda d: _per_qid_2d_run_number(d.name),
    )
    next_num = (_per_qid_2d_run_number(existing[-1].name) + 1) if existing else 1
    new_dir = seed_root / f"per_qid_2d_run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_file, new_dir / "per_qid_2d_archive.json")
    new_state = dict(st)
    new_state["run_start_iter"] = old_cap
    new_state["rounds_completed"] = 0
    new_state["current_round"] = 0
    new_state["timestamp"] = _now_iso()
    _atomic_write_json(new_dir / "per_qid_2d_state.json", new_state)
    print(f"[per_qid_2d] {run_dir.name} complete; forked to {new_dir.name} "
          f"(cap {old_cap}->{new_cap})")
    return new_dir


# ── Initial state schema ─────────────────────────────────────────────────────

def _make_initial_state(
    *,
    seed: int,
    pool_qids: list[str],
    pool_csv: Path,
    stage2_run: Path,
    flat_run_dir: Path,
    per_qid_iterations: int,
    checkpoint_every: int,
) -> dict:
    return {
        "seed": seed,
        "pool_csv": str(pool_csv),
        "pool_size": len(pool_qids),
        "stage2_source_run": str(stage2_run),
        "flat_run_used": str(flat_run_dir),
        "per_qid_iterations_cap": per_qid_iterations,
        "checkpoint_every": checkpoint_every,
        "run_start_iter": 0,
        "current_round": 0,
        "rounds_completed": 0,
        "warm_start_done": False,
        "timestamp": _now_iso(),
        "per_qid": [
            {
                "qid": qid,
                "iter_count": 0,
                "status": "active",
                "confirmed_round": None,
                "exhausted_round": None,
                "best_fitness": None,
                "best_question_type": None,
                "best_mutation_type": None,
                "last_ckpt_elite_iter": None,
            }
            for qid in pool_qids
        ],
    }


def _best_cell_2d(cells: dict) -> tuple[tuple[str, str], Elite] | None:
    """Highest-fitness occupied cell in a per-QID Archive's cells dict.

    Keys are (qtype, mtype) tuples. Returns ((qt, mt), Elite) or None.
    """
    occupied = [(k, e) for k, e in cells.items() if e is not None]
    if not occupied:
        return None
    return max(occupied, key=lambda x: x[1].fitness)


def _refresh_best(state: dict, archive: dict[str, Archive]) -> None:
    for entry in state["per_qid"]:
        a = archive.get(entry["qid"])
        if a is None:
            continue
        best = _best_cell_2d(a.cells)
        if best is None:
            entry["best_fitness"] = None
            entry["best_question_type"] = None
            entry["best_mutation_type"] = None
        else:
            (qt, mt), elite = best
            entry["best_question_type"] = qt
            entry["best_mutation_type"] = mt
            entry["best_fitness"] = round(float(elite.fitness), 6)


# ── Deterministic RNG ────────────────────────────────────────────────────────

def _stable_int_seed(qd_seed: int, qid: str, iter_no: int, salt: int = 0) -> int:
    h = hashlib.sha256(
        f"{qd_seed}:{qid}:{iter_no}:{salt}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _make_rng(qd_seed: int, qid: str, iter_no: int) -> random.Random:
    return random.Random(_stable_int_seed(qd_seed, qid, iter_no))


# ── Warm-start: one cell of the type-crossed grid ────────────────────────────

def _warm_start_one_cell(
    *,
    qid: str,
    target_qtype: str,
    target_mtype: str,
    parent_elite: Elite,
    pool_row: dict,
    swap: bool,
    qd_seed: int,
    max_retries: int,
    bleu_threshold: float,
    archive_for_qid: Archive,
    warm_log_writer: JsonlWriter,
    transcripts_writer: JsonlWriter,
    judgements_writer: JsonlWriter,
    failures_writer: JsonlWriter,
) -> str:
    """Fill one (target_qtype, target_mtype) cell of this QID's archive by
    mutating the QID's best stage-2 elite across the type boundary.

    Returns "accepted" if the cell was filled, "failed" otherwise. The
    parent is fixed (the QID's overall best stage-2 elite). The cell is
    always empty going in, so no cascaded rule is needed; if any attempt
    produces a valid candidate it lands directly.
    """
    target_cell = (target_qtype, target_mtype)
    stage_failures: list[tuple[str, str, str | None, str | None]] = []
    mutated: dict | None = None
    bleu_score_val: float | None = None

    for attempt in range(max_retries):
        m = call_mutator(
            parent_question=parent_elite.mutated_question,
            correct_answer=pool_row["correct_answer"],
            parent_distractor=parent_elite.mutated_distractor,
            mutation_type=target_mtype,
            target_question_type=target_qtype,
            seed=_stable_int_seed(qd_seed, qid, 0, salt=(attempt + 1) * 13 + 1),
        )
        if m is None:
            stage_failures.append(("mutator_parse", f"attempt={attempt}", None, None))
            continue
        parent_combined = (parent_elite.mutated_question + " "
                           + parent_elite.mutated_distractor)
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
            seed=_stable_int_seed(qd_seed, qid, 0, salt=100 + attempt),
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
            seed=_stable_int_seed(qd_seed, qid, 0, salt=200 + attempt),
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
                iter_no=0, qid=qid, target_cell=target_cell,
                stage=stage, detail=detail,
                mutated_question=mq, mutated_distractor=md,
                round=0,
            ))
        warm_log_writer.write_record({
            "phase": "warm_start", "qid": qid,
            "target_cell": list(target_cell),
            "parent_qid": parent_elite.qid,
            "parent_fitness": float(parent_elite.fitness),
            "status": "failed", "attempts": max_retries,
        })
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
            iter_no=0, qid=qid, target_cell=target_cell,
            stage="debate_or_judge", detail=f"{e!r}\n{tb}",
            round=0,
        ))
        warm_log_writer.write_record({
            "phase": "warm_start", "qid": qid,
            "target_cell": list(target_cell),
            "status": "debate_or_judge_failed",
            "error": repr(e),
        })
        return "failed"

    transcripts_writer.write_record({
        **transcript,
        "iter": 0, "round": 0, "qid": qid,
        "target_cell": list(target_cell),
        "parent_qid": parent_elite.qid,
        "bleu_to_parent": bleu_score_val,
        "swap": swap,
        "phase": "warm_start",
    })
    j_for_log = dict(judgement)
    j_for_log.pop("rounds", None)
    j_for_log.pop("transcript_str", None)
    judgements_writer.write_record({
        **j_for_log,
        "iter": 0, "round": 0, "qid": qid,
        "target_cell": list(target_cell),
        "parent_qid": parent_elite.qid,
        "bleu_to_parent": bleu_score_val,
        "swap": swap,
        "phase": "warm_start",
    })

    fit = fitness_from_judgement(judgement)
    if fit is None:
        warm_log_writer.write_record({
            "phase": "warm_start", "qid": qid,
            "target_cell": list(target_cell),
            "status": "no_fitness",
        })
        return "failed"

    # Eager probe: reuse Vertex flash-lite verdict, call flash AI Studio once.
    flash_lite_pred = _judge_correct_from_log_odds(
        judgement.get("judge_log_odds_truth")
    )
    flash_pred = _probe_flash_on_transcript(transcript)

    candidate = Elite(
        qid=qid,
        parent_qid=parent_elite.qid,
        mutated_question=mutated["mutated_question"],
        mutated_distractor=mutated["mutated_distractor"],
        correct_answer=pool_row["correct_answer"],
        fitness=fit,
        iteration=0,
        bleu_to_parent=float(bleu_score_val or 0.0),
        swap=swap,
        judge_log_odds_truth=judgement.get("judge_log_odds_truth"),
        judge_correct=judgement.get("judge_correct"),
        probe_flash_judge_correct=flash_pred,
        probe_lite_judge_correct=flash_lite_pred,
        probe_iter=0,
    )

    with archive_for_qid.lock:
        # Cell should be empty at warm-start time. Concurrent workers might
        # race on the same cell only if the same QID's warm-start cells
        # are dispatched to different threads — we serialise per-cell
        # dispatch in the outer loop, so this branch is only hit on
        # genuine empties.
        if archive_for_qid.cells[target_cell] is None:
            archive_for_qid.cells[target_cell] = candidate
            applied = True
        else:
            applied = False  # already filled by a concurrent attempt

    warm_log_writer.write_record({
        "phase": "warm_start", "qid": qid,
        "target_cell": list(target_cell),
        "parent_qid": parent_elite.qid,
        "parent_fitness": float(parent_elite.fitness),
        "candidate_fitness": float(fit),
        "probe": {
            "gemini-2.5-flash":      flash_pred,
            "gemini-2.5-flash-lite": flash_lite_pred,
        },
        "promotable": (flash_pred is False and flash_lite_pred is False),
        "applied": applied,
        "status": "accepted" if applied else "race_collision",
    })
    return "accepted" if applied else "failed"


# ── Search iteration (2D) ────────────────────────────────────────────────────

def _do_per_qid_2d_iteration(
    *,
    qid: str,
    iter_no: int,
    round_no: int,
    archive_for_qid: Archive,
    pool_row: dict,
    swap: bool,
    qd_seed: int,
    max_retries: int,
    bleu_threshold: float,
    transcripts_writer: JsonlWriter,
    judgements_writer: JsonlWriter,
    failures_writer: JsonlWriter,
    probes_writer: JsonlWriter,
) -> str:
    """One MAP-Elites iteration on a single QID's 2D archive.

    Mirrors stage-2's `_do_per_qid_iteration` but uses the 2D selection
    helpers from qd.selection directly. Probe-aware cascaded archive
    update: promotable beats non-promotable; same class falls back to
    fitness. Eager probe runs once per candidate that clears the
    pre-filter; one AI Studio flash call (Vertex flash-lite verdict is
    reused from the search judge's logprob).
    """
    rng = _make_rng(qd_seed, qid, iter_no)
    pool_single = [pool_row]
    with archive_for_qid.lock:
        target_cell = select_target_cell(archive_for_qid, rng)
        parent_row, parent_qid = select_parent(
            archive_for_qid, target_cell, pool_single, rng,
        )
    target_qtype, target_mtype = target_cell

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
    j_for_log = dict(judgement)
    j_for_log.pop("rounds", None)
    j_for_log.pop("transcript_str", None)
    judgements_writer.write_record({
        **j_for_log,
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

    # Pre-filter: if the existing cell is already promotable and we don't
    # beat its fitness, no probe outcome can change the cascaded decision.
    with archive_for_qid.lock:
        existing_snapshot = archive_for_qid.cells[target_cell]
    if (existing_snapshot is not None
            and _probe_is_promotable(existing_snapshot)
            and candidate.fitness <= existing_snapshot.fitness):
        return "rejected"

    # Eager probe.
    flash_lite_pred = _judge_correct_from_log_odds(
        judgement.get("judge_log_odds_truth")
    )
    flash_pred = _probe_flash_on_transcript(transcript)
    candidate.probe_flash_judge_correct = flash_pred
    candidate.probe_lite_judge_correct  = flash_lite_pred
    candidate.probe_iter = iter_no

    # Cascaded replacement.
    with archive_for_qid.lock:
        current = archive_for_qid.cells[target_cell]
        if current is None:
            archive_for_qid.cells[target_cell] = candidate
            decision = "accepted_empty"
        else:
            cand_promo = _probe_is_promotable(candidate)
            cur_promo  = _probe_is_promotable(current)
            if cand_promo and not cur_promo:
                archive_for_qid.cells[target_cell] = candidate
                decision = "accepted_upgrade"
            elif (not cand_promo) and cur_promo:
                decision = "rejected_downgrade"
            elif candidate.fitness > current.fitness:
                archive_for_qid.cells[target_cell] = candidate
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


# ── Checkpoint (2D — record qt + mt separately) ──────────────────────────────

def _checkpoint_one_qid_2d(
    *,
    qid: str,
    archive_for_qid: Archive,
    pool_row: dict,
    swap: bool,
    round_no: int,
    transcripts_writer: JsonlWriter,
    judge_writers: dict[str, JsonlWriter],
) -> dict:
    """One pro-debater debate + both eval judges on the QID's highest-fitness
    cell across all 20 (qtype, mtype) positions. Records the cell's qt
    AND mt so downstream analysis can attribute confirmation to either
    the original-type row or a crossed row.
    """
    with archive_for_qid.lock:
        cells = dict(archive_for_qid.cells)
    best = _best_cell_2d(cells)
    if best is None:
        return {
            "qid": qid, "round": round_no, "confirmed": False,
            "best_question_type": None, "best_mutation_type": None,
            "best_fitness": None,
            "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
            "reason": "no_elite",
        }
    (best_qtype, best_mtype), elite = best

    candidate_row = dict(pool_row)
    candidate_row["question"] = elite.mutated_question
    candidate_row["negative_answer"] = elite.mutated_distractor

    try:
        transcript = run_eval_debate(candidate_row, swap)
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return {
            "qid": qid, "round": round_no, "confirmed": False,
            "best_question_type": best_qtype,
            "best_mutation_type": best_mtype,
            "best_fitness": float(elite.fitness),
            "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
            "error": f"eval_debate_failed: {e!r}\n{tb}",
        }

    transcripts_writer.write_record({
        **transcript,
        "qid": qid, "round": round_no,
        "target_cell": [best_qtype, best_mtype],
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
                    "qid": qid, "round": round_no,
                    "target_cell": [best_qtype, best_mtype],
                    "swap": swap,
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
            "qid": qid, "round": round_no,
            "target_cell": [best_qtype, best_mtype],
            "fitness": float(elite.fitness),
            "swap": swap,
        })

    confirmed = (
        per_judge[CHECKPOINT_JUDGES[0]] is False
        and per_judge[CHECKPOINT_JUDGES[1]] is False
    )
    return {
        "qid": qid, "round": round_no, "confirmed": confirmed,
        "best_question_type": best_qtype,
        "best_mutation_type": best_mtype,
        "best_fitness": float(elite.fitness),
        "judge_correct": per_judge,
    }


# ── Gate (probe + elite-unchanged) for 2D ────────────────────────────────────

def _should_checkpoint_2d(
    *,
    archive_for_qid: Archive,
    state_entry: dict,
) -> tuple[bool, str | None, str | None, str | None, float | None]:
    """Same gates as stage 2:

      Gate 2: skip if the live best elite's iteration equals
              `last_ckpt_elite_iter` (the cell hasn't changed since the
              last pro attempt; outcome would be identical).
      Gate 1: skip if the best cell is not probe-promotable.

    Returns (should_run, skip_reason, best_qtype, best_mtype, best_fitness).
    """
    with archive_for_qid.lock:
        cells = dict(archive_for_qid.cells)
    best = _best_cell_2d(cells)
    if best is None:
        return False, "no_elite", None, None, None
    (best_qtype, best_mtype), best_elite = best
    last_iter = state_entry.get("last_ckpt_elite_iter")
    if last_iter is not None and int(best_elite.iteration) == int(last_iter):
        return False, "elite_unchanged", best_qtype, best_mtype, float(best_elite.fitness)
    if not _probe_is_promotable(best_elite):
        return False, "probe_not_promotable", best_qtype, best_mtype, float(best_elite.fitness)
    return True, None, best_qtype, best_mtype, float(best_elite.fitness)


# ── Main orchestration ───────────────────────────────────────────────────────

def run_per_qid_2d_mode(args: argparse.Namespace) -> int:
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

    already_confirmed = _load_already_confirmed_qids(seed)
    print(f"[per_qid_2d] seed={seed}  pool={pool_csv}  "
          f"full={len(full_pool_qids)}  already_confirmed={len(already_confirmed)}")

    # Stage 2 source: the latest per_qid_run_NNN unless pinned via --stage2-source-run.
    stage2_pin = getattr(args, "stage2_source_run", None)
    stage2_run = _latest_per_qid_run_dir(seed, pin=stage2_pin)
    stage2_archive_1d = _load_stage2_archive(stage2_run)
    stage2_state = _load_stage2_state(stage2_run)
    print(f"[per_qid_2d] stage-2 source: {stage2_run}")

    # Active set for stage 3: QIDs that stage 2 left unconfirmed (status
    # active or exhausted) and that aren't already in best_framings.csv.
    stage2_status: dict[str, str] = {
        e["qid"]: e["status"] for e in stage2_state.get("per_qid", [])
    }
    unconfirmed_qids = [
        qid for qid in full_pool_qids
        if qid not in already_confirmed
        and stage2_status.get(qid) in ("active", "exhausted")
    ]
    pool_by_qid_active = {q: pool_by_qid[q] for q in unconfirmed_qids}
    print(f"[per_qid_2d] active QIDs entering stage 3: {len(unconfirmed_qids)}")

    if not unconfirmed_qids:
        print(f"[per_qid_2d] seed={seed}: nothing to do.")
        return 0

    flat_run_dir = _resolve_flat_run(seed, getattr(args, "flat_run", None))
    warm_transcript_index = _build_warm_start_transcript_index(flat_run_dir)
    print(f"[per_qid_2d] flat-grid transcript cache: {len(warm_transcript_index)} entries")

    run_dir, is_resume = _resume_or_new_run_dir(seed)
    if is_resume:
        run_dir = _maybe_fork_completed_run(run_dir, seed, args.per_qid_iterations)
    archive_path = run_dir / "per_qid_2d_archive.json"
    state_path   = run_dir / "per_qid_2d_state.json"
    search_dir   = run_dir / "search"
    ckpt_dir     = run_dir / "checkpoint"
    search_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = map_seed_dir(seed) / "per_qid_2d_summary.json"

    state_lock = threading.Lock()

    if is_resume and archive_path.exists() and state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        with open(archive_path, encoding="utf-8") as f:
            archive = _archive_from_jsonable(json.load(f))
        # Reconcile with best_framings.csv for any cross-process confirmations
        existing_qids = {e["qid"] for e in state["per_qid"]}
        for qid in unconfirmed_qids:
            if qid not in existing_qids:
                state["per_qid"].append({
                    "qid": qid, "iter_count": 0, "status": "active",
                    "confirmed_round": None, "exhausted_round": None,
                    "best_fitness": None,
                    "best_question_type": None, "best_mutation_type": None,
                    "last_ckpt_elite_iter": None,
                })
                archive[qid] = Archive()
        for entry in state["per_qid"]:
            entry.setdefault("last_ckpt_elite_iter", None)
            entry.setdefault("best_question_type", None)
        for entry in state["per_qid"]:
            if entry["status"] == "active" and entry["qid"] in already_confirmed:
                entry["status"] = "confirmed"
                entry["confirmed_round"] = entry.get("confirmed_round") or 0
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
                print(f"[per_qid_2d] cap raised {old_cap}->{args.per_qid_iterations}: "
                      f"re-activated {reactivated} exhausted QIDs")
        state["per_qid_iterations_cap"] = args.per_qid_iterations
        state["current_round"] = int(state.get("rounds_completed", 0))
        print(f"[per_qid_2d] resuming {run_dir.name}: "
              f"rounds_completed={state.get('rounds_completed', 0)}, "
              f"warm_start_done={state.get('warm_start_done', False)}, "
              f"active="
              f"{sum(1 for e in state['per_qid'] if e['status'] == 'active')}")
    else:
        archive = {qid: Archive() for qid in unconfirmed_qids}
        state = _make_initial_state(
            seed=seed, pool_qids=unconfirmed_qids, pool_csv=pool_csv,
            stage2_run=stage2_run, flat_run_dir=flat_run_dir,
            per_qid_iterations=args.per_qid_iterations,
            checkpoint_every=args.checkpoint_every,
        )
        _save_archive(archive, archive_path, threading.Lock())
        _save_state(state, state_path, state_lock)
        print(f"[per_qid_2d] fresh run at {run_dir.name}")

    state_by_qid: dict[str, dict] = {e["qid"]: e for e in state["per_qid"]}
    run_start_iter = int(state.get("run_start_iter", 0))

    search_t  = JsonlWriter(search_dir / "search_transcripts.jsonl")
    search_j  = JsonlWriter(search_dir / "search_judgements.jsonl")
    search_f  = JsonlWriter(search_dir / "search_failures.jsonl")
    search_p  = JsonlWriter(search_dir / "search_probes.jsonl")
    warm_log  = JsonlWriter(search_dir / "warm_start_log.jsonl")
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
    max_retries      = config.QD_MAX_RETRIES_PER_ITER
    bleu_threshold   = config.QD_BLEU_THRESHOLD

    t_start = time.time()

    try:
        # ── Warm-start phase (only if not yet done) ───────────────────────────
        if not state.get("warm_start_done", False):
            _do_warm_start(
                archive=archive,
                state=state,
                state_lock=state_lock,
                stage2_archive_1d=stage2_archive_1d,
                pool_by_qid_active=pool_by_qid_active,
                swap_map=swap_map,
                qd_seed=qd_seed_for_iters,
                max_retries=max_retries,
                bleu_threshold=bleu_threshold,
                workers=args.workers,
                transcripts_writer=search_t,
                judgements_writer=search_j,
                failures_writer=search_f,
                warm_log_writer=warm_log,
            )
            state["warm_start_done"] = True
            _refresh_best(state, archive)
            _save_archive(archive, archive_path, threading.Lock())
            _save_state(state, state_path, state_lock)

        # ── Search + checkpoint rounds ────────────────────────────────────────
        while True:
            active = [e["qid"] for e in state["per_qid"] if e["status"] == "active"]
            if not active:
                print("[per_qid_2d] no active QIDs — stopping.")
                break

            round_no = int(state.get("current_round", 0)) + 1
            state["current_round"] = round_no
            print(f"[per_qid_2d] ── round {round_no} ─────────────────────"
                  f"  active={len(active)}")

            # Search phase ────────────────────────────────────────────────────
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
                print(f"[per_qid_2d] search: dispatching {len(tasks)} iters "
                      f"across {args.workers} worker(s)")
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = {}
                    for qid, iter_no in tasks:
                        pool_row = pool_by_qid[qid]
                        swap = bool(swap_map.get(qid, False))
                        fut = ex.submit(
                            _do_per_qid_2d_iteration,
                            qid=qid, iter_no=iter_no, round_no=round_no,
                            archive_for_qid=archive[qid],
                            pool_row=pool_row, swap=swap,
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
                            print(f"[per_qid_2d] ITER CRASH qid={qid} iter={iter_no}: {e!r}")
                        with state_lock:
                            entry = state_by_qid[qid]
                            if iter_no > entry["iter_count"]:
                                entry["iter_count"] = iter_no
                        done += 1
                        if done % 50 == 0:
                            print(f"[per_qid_2d] search: {done}/{len(tasks)} iters done")

            _refresh_best(state, archive)
            _save_archive(archive, archive_path, threading.Lock())
            _save_state(state, state_path, state_lock)

            # Lazy-probe phase ────────────────────────────────────────────────
            # For 2D archives, every cell created during stage-3 warm-start
            # or search has been eagerly probed. Lazy-probe only fires on
            # cells inherited from a stage 2 that pre-dates the probe code
            # (probe_iter is None). Those are the original-row cells from
            # stage 2 carried in at warm-start; they get probed once here.
            lazy_targets: list[str] = []
            for qid in active:
                with archive[qid].lock:
                    best = _best_cell_2d(archive[qid].cells)
                if best is None:
                    continue
                if best[1].probe_iter is None:
                    lazy_targets.append(qid)
            if lazy_targets:
                print(f"[per_qid_2d] lazy probe: {len(lazy_targets)} cell(s) "
                      f"need first-time probe")
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    lp_futures = {}
                    for qid in lazy_targets:
                        pool_row = pool_by_qid[qid]
                        swap = bool(swap_map.get(qid, False))
                        # The cell-shape-agnostic lazy probe takes a
                        # dict-like archive; we pass {qid: cells dict}.
                        lp_futures[ex.submit(
                            _lazy_probe_best_cell,
                            qid=qid,
                            archive={qid: archive[qid].cells},
                            archive_lock=archive[qid].lock,
                            pool_row=pool_row, swap=swap, round_no=round_no,
                            probes_writer=search_p,
                            warm_transcript_index=warm_transcript_index,
                        )] = qid
                    for fut in as_completed(lp_futures):
                        try:
                            fut.result()
                        except Exception as e:
                            qid = lp_futures[fut]
                            print(f"[per_qid_2d] LAZY PROBE CRASH qid={qid}: {e!r}")
                _save_archive(archive, archive_path, threading.Lock())

            # Gate phase ──────────────────────────────────────────────────────
            to_checkpoint: list[str] = []
            ckpt_results: list[dict] = []
            confirmed_now: list[str] = []
            skip_counts = {
                "no_elite": 0,
                "probe_not_promotable": 0,
                "elite_unchanged": 0,
            }
            for qid in active:
                decision, reason, best_qt, best_mt, best_fit = _should_checkpoint_2d(
                    archive_for_qid=archive[qid],
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
                        "best_question_type": best_qt,
                        "best_mutation_type": best_mt,
                        "best_fitness": best_fit,
                        "judge_correct": {m: None for m in CHECKPOINT_JUDGES},
                    })

            print(f"[per_qid_2d] checkpoint: dispatching {len(to_checkpoint)}/"
                  f"{len(active)} pro debates  "
                  f"(skipped: not_promotable={skip_counts['probe_not_promotable']}, "
                  f"elite_unchanged={skip_counts['elite_unchanged']}, "
                  f"no_elite={skip_counts['no_elite']}) "
                  f"across {args.workers} worker(s)")

            # Dispatch phase ──────────────────────────────────────────────────
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {}
                for qid in to_checkpoint:
                    pool_row = pool_by_qid[qid]
                    swap = bool(swap_map.get(qid, False))
                    fut = ex.submit(
                        _checkpoint_one_qid_2d,
                        qid=qid, archive_for_qid=archive[qid],
                        pool_row=pool_row, swap=swap, round_no=round_no,
                        transcripts_writer=ckpt_t,
                        judge_writers=judge_writers,
                    )
                    futures[fut] = qid
                done = 0
                for fut in as_completed(futures):
                    qid = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        tb = traceback.format_exc(limit=2)
                        print(f"[per_qid_2d] CHECKPOINT CRASH qid={qid}: {e!r}\n{tb}")
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
                            entry["confirming_question_type"] = res.get("best_question_type")
                            entry["confirming_mutation_type"]  = res.get("best_mutation_type")
                            confirmed_iter_count = int(entry["iter_count"])
                        confirmed_now.append(qid)
                        best_qt = res.get("best_question_type")
                        best_mt = res.get("best_mutation_type")
                        if best_mt and best_qt:
                            with archive[qid].lock:
                                elite_for_append = archive[qid].cells.get((best_qt, best_mt))
                            if elite_for_append is not None:
                                try:
                                    _append_confirmation_to_best_framings(
                                        seed=seed, qid=qid,
                                        elite=elite_for_append,
                                        best_mtype=best_mt,
                                        pool_row=pool_by_qid_active[qid],
                                        baseline_swap=swap_map,
                                        eval_swap=bool(swap_map.get(qid, False)),
                                        iter_count=confirmed_iter_count,
                                        source_run_name=run_dir.name,
                                        confirming_qtype=best_qt,
                                    )
                                except Exception as e:
                                    print(f"[per_qid_2d] WARN: failed to append "
                                          f"qid={qid} to best_framings.csv: {e!r}")
                    done += 1
                    if done % 20 == 0:
                        print(f"[per_qid_2d] checkpoint: {done}/{len(to_checkpoint)} done")

            # Update last_ckpt_elite_iter for every QID we considered ──────────
            with state_lock:
                for qid in active:
                    with archive[qid].lock:
                        best = _best_cell_2d(archive[qid].cells)
                    if best is not None:
                        state_by_qid[qid]["last_ckpt_elite_iter"] = int(best[1].iteration)

            # Budget check ────────────────────────────────────────────────────
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

            crossed = sum(
                1 for r in ckpt_results
                if r.get("confirmed")
                and r.get("best_question_type")
                and r.get("best_question_type")
                    != pool_by_qid[r["qid"]].get("question_type", "")
            )
            same_row = sum(
                1 for r in ckpt_results
                if r.get("confirmed")
                and r.get("best_question_type")
                    == pool_by_qid[r["qid"]].get("question_type", "")
            )

            ckpt_summary["rounds"].append({
                "round": round_no,
                "n_active_before":   len(active),
                "confirmed":         sorted(confirmed_now),
                "confirmed_crossed_type": crossed,
                "confirmed_same_row":     same_row,
                "exhausted":         sorted(exhausted_now),
                "n_confirmed_total": n_confirmed_total,
                "n_exhausted_total": n_exhausted_total,
                "n_active_after":    n_active_after,
                "pro_debates_this_round": len(to_checkpoint),
                "gate_dispatched":   len(to_checkpoint),
                "gate_skipped":      skip_counts,
                "gate_skipped_total": sum(skip_counts.values()),
                "lazy_probes":       len(lazy_targets),
                "results":           ckpt_results,
            })
            _atomic_write_json(ckpt_summary_path, ckpt_summary)

            print(f"[per_qid_2d] round {round_no} done: "
                  f"+{len(confirmed_now)} confirmed "
                  f"(crossed={crossed}, same_row={same_row}), "
                  f"+{len(exhausted_now)} exhausted, active={n_active_after}, "
                  f"total_confirmed={n_confirmed_total}, "
                  f"pro_debates_this_round={len(to_checkpoint)}, "
                  f"gate_skipped="
                  f"{skip_counts['probe_not_promotable']}probe/"
                  f"{skip_counts['elite_unchanged']}stale, "
                  f"lazy_probes={len(lazy_targets)}")
            try:
                slack(
                    f":sparkles: per_qid_2d seed={seed} round={round_no}: "
                    f"+{len(confirmed_now)} confirmed "
                    f"(crossed={crossed}, same_row={same_row}), "
                    f"+{len(exhausted_now)} exhausted, active={n_active_after}, "
                    f"gate={len(to_checkpoint)}/{len(active)}, "
                    f"lazy={len(lazy_targets)}"
                )
            except Exception:
                pass

    finally:
        for w in (search_t, search_j, search_f, search_p, warm_log,
                  ckpt_t, *judge_writers.values()):
            try:
                w.close()
            except Exception:
                pass

    # Final summary ───────────────────────────────────────────────────────────
    n_confirmed = sum(1 for e in state["per_qid"] if e["status"] == "confirmed")
    n_exhausted = sum(1 for e in state["per_qid"] if e["status"] == "exhausted")
    n_active    = sum(1 for e in state["per_qid"] if e["status"] == "active")
    confirmed_crossed = sum(
        1 for e in state["per_qid"]
        if e["status"] == "confirmed"
        and e.get("confirming_question_type")
        and e.get("confirming_question_type") != pool_by_qid[e["qid"]].get("question_type", "")
    )
    confirmed_same = n_confirmed - confirmed_crossed
    elapsed = time.time() - t_start
    summary = {
        "seed":                       seed,
        "pool_csv":                   str(pool_csv),
        "pool_size_full":             len(full_pool_qids),
        "pool_size_active":           len(state["per_qid"]),
        "already_confirmed":          sorted(
            _load_already_confirmed_qids(seed),
            key=lambda q: int(q) if q.isdigit() else q,
        ),
        "stage2_source_run":          str(stage2_run),
        "flat_run_used":              str(flat_run_dir),
        "run_dir":                    str(run_dir),
        "per_qid_iterations_cap":     args.per_qid_iterations,
        "checkpoint_every":           args.checkpoint_every,
        "rounds_completed":           state.get("rounds_completed", 0),
        "confirmed":                  n_confirmed,
        "confirmed_crossed_type":     confirmed_crossed,
        "confirmed_same_row":         confirmed_same,
        "exhausted":                  n_exhausted,
        "remaining_active":           n_active,
        "elapsed_sec":                round(elapsed, 1),
        "timestamp":                  _now_iso(),
        "per_qid":                    state["per_qid"],
    }
    _atomic_write_json(summary_path, summary)
    print(f"[per_qid_2d] DONE. confirmed={n_confirmed} "
          f"(crossed={confirmed_crossed}, same_row={confirmed_same}), "
          f"exhausted={n_exhausted}, active_left={n_active}.")
    print(f"[per_qid_2d] summary -> {summary_path}")
    try:
        slack(
            f":white_check_mark: per_qid_2d done — seed={seed}\n"
            f"confirmed: {n_confirmed} (crossed={confirmed_crossed}, "
            f"same_row={confirmed_same}), exhausted: {n_exhausted}, "
            f"active_left: {n_active}, elapsed: {elapsed:.0f}s"
        )
    except Exception:
        pass
    return 0


# ── Warm-start orchestrator ──────────────────────────────────────────────────

def _do_warm_start(
    *,
    archive: dict[str, Archive],
    state: dict,
    state_lock: threading.Lock,
    stage2_archive_1d: dict[str, dict[str, Elite | None]],
    pool_by_qid_active: dict[str, dict],
    swap_map: dict[str, bool],
    qd_seed: int,
    max_retries: int,
    bleu_threshold: float,
    workers: int,
    transcripts_writer: JsonlWriter,
    judgements_writer: JsonlWriter,
    failures_writer: JsonlWriter,
    warm_log_writer: JsonlWriter,
) -> None:
    """Populate every QID's 5x4 archive: original-type row copied directly
    from stage 2, the other 16 cells filled by cross-type mutator calls
    using each QID's overall best stage-2 elite as the fixed parent."""
    # Phase A: copy original-row cells. Cheap, no LLM calls, do it serially.
    n_copied = 0
    qids_to_warm_cross: list[str] = []
    for qid, archive_for_qid in archive.items():
        pool_row = pool_by_qid_active.get(qid)
        if pool_row is None:
            continue
        original_qtype = pool_row.get("question_type", "")
        if original_qtype not in QUESTION_TYPES:
            warm_log_writer.write_record({
                "phase": "warm_start_copy", "qid": qid,
                "status": "skipped_unknown_qtype",
                "qtype": original_qtype,
            })
            continue
        stage2_cells = stage2_archive_1d.get(qid, {})
        for mt in MUTATION_TYPES:
            src = stage2_cells.get(mt)
            if src is None:
                continue
            with archive_for_qid.lock:
                archive_for_qid.cells[(original_qtype, mt)] = src
                n_copied += 1
        # If this QID has no stage-2 elites, there is nothing to mutate
        # from. Skip crossed-row warm-start; the lazy-probe + search
        # phases will still run, but the archive starts with one or more
        # empty rows (the search will explore them via the empty-cell
        # preference path).
        best = _stage2_best_elite(stage2_cells)
        if best is not None:
            qids_to_warm_cross.append(qid)
        warm_log_writer.write_record({
            "phase": "warm_start_copy", "qid": qid,
            "original_qtype": original_qtype,
            "stage2_cells_copied": sum(1 for mt in MUTATION_TYPES
                                       if stage2_cells.get(mt) is not None),
            "best_stage2_fitness": (float(best.fitness) if best is not None else None),
        })
    print(f"[per_qid_2d] warm-start phase A (copy): copied {n_copied} stage-2 "
          f"cells into the original-type rows of {len(archive)} QIDs.")

    # Phase B: fill the 16 type-crossed cells per QID via the mutator.
    # Build the full task list, then dispatch in parallel.
    tasks: list[tuple[str, str, str]] = []  # (qid, target_qtype, target_mtype)
    for qid in qids_to_warm_cross:
        pool_row = pool_by_qid_active[qid]
        original_qtype = pool_row["question_type"]
        for tqt in QUESTION_TYPES:
            if tqt == original_qtype:
                continue
            for tmt in MUTATION_TYPES:
                tasks.append((qid, tqt, tmt))
    print(f"[per_qid_2d] warm-start phase B (cross-type): dispatching "
          f"{len(tasks)} cells across {workers} worker(s).")

    parents: dict[str, Elite] = {}
    for qid in qids_to_warm_cross:
        best = _stage2_best_elite(stage2_archive_1d.get(qid, {}))
        if best is not None:
            parents[qid] = best

    if not tasks:
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for qid, tqt, tmt in tasks:
            pool_row = pool_by_qid_active[qid]
            swap = bool(swap_map.get(qid, False))
            parent_elite = parents[qid]
            fut = ex.submit(
                _warm_start_one_cell,
                qid=qid,
                target_qtype=tqt,
                target_mtype=tmt,
                parent_elite=parent_elite,
                pool_row=pool_row,
                swap=swap,
                qd_seed=qd_seed,
                max_retries=max_retries,
                bleu_threshold=bleu_threshold,
                archive_for_qid=archive[qid],
                warm_log_writer=warm_log_writer,
                transcripts_writer=transcripts_writer,
                judgements_writer=judgements_writer,
                failures_writer=failures_writer,
            )
            futures[fut] = (qid, tqt, tmt)
        done = 0
        accepted = 0
        for fut in as_completed(futures):
            qid, tqt, tmt = futures[fut]
            try:
                res = fut.result()
                if res == "accepted":
                    accepted += 1
            except Exception as e:
                print(f"[per_qid_2d] WARM CELL CRASH qid={qid} "
                      f"({tqt},{tmt}): {e!r}")
            done += 1
            if done % 50 == 0:
                print(f"[per_qid_2d] warm-start: {done}/{len(tasks)} "
                      f"cells processed, {accepted} accepted")
    print(f"[per_qid_2d] warm-start phase B done: {accepted}/{len(tasks)} "
          f"cells filled.")
