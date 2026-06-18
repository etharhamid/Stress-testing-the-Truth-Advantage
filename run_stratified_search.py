"""Recursive MAP-Elites stratification.

Three modes:

* `--mode per_qid` (default): per-QID 1D MAP-Elites with eval-gated removal.
  Each QID owns a 4-cell archive (one per mutation type) warm-started from
  the latest flat_grid_run_NNN. Every `--checkpoint-every` iterations the
  highest-fitness cell of each active QID is debated by the pro debater and
  judged by both flash + flash-lite; QIDs that fool both judges are
  confirmed and removed from the active pool. Loop ends when the active set
  is empty or every QID hits the `--per-qid-iterations` cap.
* `--mode per_qid_2d` (Stage 3): per-QID 2D MAP-Elites. Each QID
  unconfirmed after stage-2 (`per_qid`) gets its own 5 question_type x
  4 mutation_type archive (20 cells). The QID's original-type row is
  pre-seeded directly from stage 2's 4 mutation cells; the other 16
  cells are warm-started by asking the mutator to cross into the
  target question type while applying the target mutation type. Search
  then runs MAP-Elites per QID across all 20 cells. Same gate (probe-
  promotable + elite-unchanged) at checkpoint. Confirmations are
  recorded with their cell's question type so we can distinguish
  "original row" from "type-crossed" confirmations.
* `--mode cell`: the legacy cell-stratified mode. Successive fresh MAP-Elites
  rounds where each round's pool is the original pool minus all QIDs that
  were "deceptive in both judges" in the same archive cell in any prior
  round. Round 1 is the existing run_001 — this script only adds rounds 2,
  3, …

Usage:
    # Per-QID mode (default):
    python run_stratified_search.py --seed 42 --per-qid-iterations 20 --workers 12

    # Per-QID 2D mode (Stage 3):
    python run_stratified_search.py --seed 42 --mode per_qid_2d \
        --per-qid-iterations 100 --checkpoint-every 100 --workers 12

    # Legacy cell-stratified mode:
    python run_stratified_search.py --seed 42 --mode cell \
        --iterations 500 --max-rounds 5 --workers 12

    # Resume after a crash (identical command — fully idempotent in all modes):
    python run_stratified_search.py --seed 42 ...
"""

from pathlib import Path
import argparse, csv, datetime, json, subprocess, sys, traceback

from qd.notify import slack

# ── archive.json field-name reference (confirmed from real archive) ───────────
# Top-level keys : iter_count, question_types, mutation_types, cells
# cells[i]       : {"cell": [question_type, mutation_type], "elite": {...} | null}
# elite fields   : qid (str), fitness (float), iteration (int), parent_qid (str),
#                  mutated_question, mutated_distractor, correct_answer,
#                  bleu_to_parent, swap, judge_log_odds_truth, judge_correct
# Pool CSV column "id" and archive elite field "qid" hold the same string value.
# ─────────────────────────────────────────────────────────────────────────────

MAP_RESULTS_ROOT = Path("qd_results/map_results")


# ── Path helpers ──────────────────────────────────────────────────────────────

def seed_dir(seed: int) -> Path:
    return MAP_RESULTS_ROOT / f"seed_{seed}"


def run_dir(seed: int, run_n: int) -> Path:
    return seed_dir(seed) / f"run_{run_n:03d}"


def archive_path(seed: int, run_n: int) -> Path:
    return run_dir(seed, run_n) / "search" / "archive.json"


def stratified_pool_csv(seed: int, round_n: int) -> Path:
    # round_n >= 2 always (round 1 uses the original both_correct.csv — no copy made)
    return seed_dir(seed) / f"stratified_pool_round_{round_n:03d}.csv"


def summary_json(seed: int) -> Path:
    return seed_dir(seed) / "stratified_summary.json"


def orchestrator_log_dir(seed: int) -> Path:
    return seed_dir(seed) / "orchestrator_logs"


def latest_run_number(seed: int) -> int:
    """Highest run_NNN that exists and has a search/archive.json."""
    sd = seed_dir(seed)
    candidates = [
        int(d.name.split("_")[1])
        for d in sd.iterdir()
        if d.is_dir()
        and d.name.startswith("run_")
        and d.name[4:].isdigit()
        and (d / "search" / "archive.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run dirs found under {sd}")
    return max(candidates)


# ── Archive inspection helpers ────────────────────────────────────────────────

def load_archive(archive_json: Path) -> dict:
    with open(archive_json) as f:
        return json.load(f)


def elite_qids(archive: dict) -> set:
    """Unique question IDs of all non-null archive cells."""
    return {
        cell["elite"]["qid"]         # archive stores question ID as "qid"
        for cell in archive["cells"]
        if cell.get("elite") is not None
    }


def fitness_stats(archive: dict) -> dict:
    vals = [
        cell["elite"]["fitness"]
        for cell in archive["cells"]
        if cell.get("elite") is not None
    ]
    if not vals:
        return {"min": None, "mean": None, "max": None}
    return {
        "min":  round(min(vals), 4),
        "mean": round(sum(vals) / len(vals), 4),
        "max":  round(max(vals), 4),
    }


def cells_occupied(archive: dict) -> int:
    return sum(1 for c in archive["cells"] if c.get("elite") is not None)


def iter_count(archive: dict) -> int:
    return archive["iter_count"]


# ── Pool CSV helpers ──────────────────────────────────────────────────────────

def count_csv_rows(csv_path: Path) -> int:
    with open(csv_path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def write_filtered_pool(source: Path, exclude_qids: set, dest: Path) -> int:
    """Write rows from source whose 'id' column is not in exclude_qids to dest.

    Pool CSV uses column 'id'; archive elites use field 'qid' — same string
    values, so comparison is correct.  Preserves all columns verbatim.
    Returns number of rows written.
    """
    with open(source, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r["id"] not in exclude_qids]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ── Summary helpers ───────────────────────────────────────────────────────────

def write_summary(path: Path, data: dict) -> None:
    """Atomic write: tmp file then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def build_round_record(
    round_n: int,
    run_n: int,
    seed: int,
    pool_csv: Path,
    pool_size: int,
    archive: dict,
    deceptive_qids: set,
    cumulative_excluded: set,
    original_pool_size: int,
    eval_completed: bool,
) -> dict:
    qids = elite_qids(archive)
    fs = fitness_stats(archive)
    return {
        "round":                        round_n,
        "run_number":                   run_n,
        "run_dir":                      str(run_dir(seed, run_n)),
        "pool_csv":                     str(pool_csv),
        "pool_size":                    pool_size,
        "elite_qids":                   sorted(qids),
        "n_elite_qids":                 len(qids),
        "eval_completed":               eval_completed,
        "deceptive_both_judges":        sorted(deceptive_qids),
        "n_deceptive_both":             len(deceptive_qids),
        "cells_occupied":               cells_occupied(archive),
        "fitness_min":                  fs["min"],
        "fitness_mean":                 fs["mean"],
        "fitness_max":                  fs["max"],
        "iter_count":                   iter_count(archive),
        "cumulative_excluded_qids":     sorted(cumulative_excluded),
        "cumulative_excluded_count":    len(cumulative_excluded),
        "cumulative_coverage_fraction": round(len(cumulative_excluded) / original_pool_size, 4),
    }


# ── Subprocess helper ─────────────────────────────────────────────────────────

def invoke_qd_search(
    seed: int,
    pool_csv: Path,
    iterations: int,
    workers: int,
    no_plot: bool,
    fresh_archive: bool,
    log_fh,
) -> int:
    """Run run_qd_search.py as a subprocess. Returns exit code."""
    cmd = [
        sys.executable, "run_qd_search.py",
        "--seed",        str(seed),
        "--pool-source", str(pool_csv),
        "--iterations",  str(iterations),
        "--workers",     str(workers),
    ]
    if fresh_archive:
        cmd.append("--fresh")
    if no_plot:
        cmd.append("--no-plot")

    msg = f"[stratified] $ {' '.join(cmd)}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)
    log_fh.flush()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()
    proc.wait()
    return proc.returncode


def invoke_qd_eval(seed: int, run_n: int, log_fh) -> int:
    """Run run_qd_eval.py for a specific run. Returns exit code."""
    cmd = [
        sys.executable, "run_qd_eval.py",
        "--seed",   str(seed),
        "--run",    str(run_n),
        "--judges", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ]
    msg = f"[stratified] $ {' '.join(cmd)}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg); log_fh.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line); log_fh.flush()
    proc.wait()
    return proc.returncode


def eval_csvs_exist(seed: int, run_n: int) -> bool:
    ed = run_dir(seed, run_n) / "eval"
    return (
        (ed / "eval_results_flash.csv").exists()
        and (ed / "eval_results_flash-lite.csv").exists()
    )


def deceptive_both_judges(seed: int, run_n: int) -> set:
    """Return QIDs where both judges were wrong in the same cell (same question_type + mutation_type).

    A QID fooling flash in one cell and flash-lite in a different cell does not
    qualify — the same framing must have deceived both models.
    """
    eval_dir = run_dir(seed, run_n) / "eval"
    flash_csv      = eval_dir / "eval_results_flash.csv"
    flash_lite_csv = eval_dir / "eval_results_flash-lite.csv"

    def wrong_cells(csv_path: Path) -> set:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return {
                (row["qid"], row["question_type"], row["mutation_type"])
                for row in reader
                if row["judge_correct"].strip().lower() in ("false", "0")
            }

    both_wrong_cells = wrong_cells(flash_csv) & wrong_cells(flash_lite_csv)
    return {qid for qid, _, _ in both_wrong_cells}


def _finish_eval(
    seed: int, run_n: int, summary: dict, op_size: int, log_fh, sf: Path
) -> bool:
    """Run eval (or read existing CSVs), update last summary record in place, write.

    Returns False if the eval subprocess fails.
    """
    if eval_csvs_exist(seed, run_n):
        msg = f"[stratified] Eval CSVs found for run_{run_n:03d} — reading directly.\n"
        print(msg, end="", flush=True)
        log_fh.write(msg); log_fh.flush()
    else:
        exit_code = invoke_qd_eval(seed, run_n, log_fh)
        if exit_code != 0:
            return False

    deceptive = deceptive_both_judges(seed, run_n)
    prior_excluded = {
        qid
        for r in summary["rounds"][:-1]
        if r.get("eval_completed")
        for qid in r.get("deceptive_both_judges", [])
    }
    new_excluded = prior_excluded | deceptive

    last = summary["rounds"][-1]
    last["eval_completed"]               = True
    last["deceptive_both_judges"]        = sorted(deceptive)
    last["n_deceptive_both"]             = len(deceptive)
    last["cumulative_excluded_qids"]     = sorted(new_excluded)
    last["cumulative_excluded_count"]    = len(new_excluded)
    last["cumulative_coverage_fraction"] = round(len(new_excluded) / op_size, 4)
    write_summary(sf, summary)
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratification across successive rounds — per-QID 1D "
                    "MAP-Elites (default) or legacy cell-stratified pools."
    )
    p.add_argument("--seed", type=int, required=True,
                   help="Seed to stratify (42 / 123 / 7 / 13 / 73).")
    p.add_argument("--mode", choices=["per_qid", "per_qid_2d", "cell"],
                   default="per_qid",
                   help="per_qid (default): per-QID 1D MAP-Elites with eval-gated "
                        "removal. per_qid_2d (Stage 3): per-QID 2D MAP-Elites with "
                        "type-crossed warm-start. cell: legacy cell-stratified "
                        "across pools.")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent workers (default 8).")

    # per_qid mode flags
    p.add_argument("--per-qid-iterations", type=int, default=25,
                   help="[per_qid] Max search iterations per QID before exhausted (default 25).")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="[per_qid] Run pro-debate checkpoint every N iters per QID (default 5).")
    p.add_argument("--flat-run", type=int, default=None,
                   help="[per_qid, per_qid_2d] Warm-start from flat_grid_run_{N:03d} "
                        "(default: highest-numbered).")
    p.add_argument("--stage2-source-run", type=int, default=None,
                   help="[per_qid_2d] Pin the stage-2 source to "
                        "per_qid_run_{N:03d} (default: highest-numbered).")
    p.add_argument("--checkpoint-cells", choices=["best", "fallback", "all"],
                   default="best",
                   help="[per_qid] Per-checkpoint eval strategy. "
                        "best (default): only the highest-fitness cell. "
                        "fallback: try cells in fitness order, stop on first "
                        "confirmation (smart-fallback). "
                        "all: debate every occupied cell regardless. "
                        "Fallback/all also record which rank confirmed each QID "
                        "so you can compute what 'best' would have caught.")

    # cell mode flags (legacy)
    p.add_argument("--iterations", type=int, default=500,
                   help="[cell] MAP-Elites iterations per round (default 500).")
    p.add_argument("--max-rounds", type=int, default=10,
                   help="[cell] Max additional rounds beyond round 1 (default 10).")
    p.add_argument("--min-pool-size", type=int, default=5,
                   help="[cell] Stop before a round if remaining pool < this (default 5).")
    p.add_argument("--no-plot", action="store_true",
                   help="[cell] Forwarded to run_qd_search.py.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.mode == "per_qid":
        from qd.per_qid_search import run_per_qid_mode
        sys.exit(run_per_qid_mode(args))

    if args.mode == "per_qid_2d":
        from qd.per_qid_2d_search import run_per_qid_2d_mode
        sys.exit(run_per_qid_2d_mode(args))

    seed = args.seed
    _t_start = datetime.datetime.now()

    from qd.config import pool_csv_for_seed
    original_pool_csv = Path(pool_csv_for_seed(seed))

    orchestrator_log_dir(seed).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = orchestrator_log_dir(seed) / f"stratified_{ts}.log"

    with open(log_path, "w") as log_fh:

        # ── Load or bootstrap summary ─────────────────────────────────────────
        sf = summary_json(seed)
        summary = None
        if sf.exists():
            with open(sf) as f:
                summary = json.load(f)
            if summary["rounds"] and "eval_completed" not in summary["rounds"][0]:
                print("[stratified] Old-format summary detected — deleting and re-bootstrapping.")
                log_fh.write("[stratified] Old-format summary detected. Deleting.\n")
                sf.unlink()
                summary = None
            else:
                print(f"[stratified] Resuming: {len(summary['rounds'])} round(s) in summary.")
                log_fh.write(f"[stratified] Resuming: {len(summary['rounds'])} round(s).\n")
        if summary is None:
            run1_n  = latest_run_number(seed)
            arc1    = load_archive(archive_path(seed, run1_n))
            op_size = count_csv_rows(original_pool_csv)
            summary = {
                "seed":               seed,
                "original_pool_size": op_size,
                "original_pool_csv":  str(original_pool_csv),
                "rounds": [
                    build_round_record(
                        round_n=1, run_n=run1_n, seed=seed,
                        pool_csv=original_pool_csv, pool_size=op_size,
                        archive=arc1,
                        deceptive_qids=set(),
                        cumulative_excluded=set(),
                        original_pool_size=op_size,
                        eval_completed=False,
                    )
                ],
            }
            write_summary(sf, summary)
            print(f"[stratified] Bootstrapped round 1 from run_{run1_n:03d}. "
                  f"Elite QIDs: {len(elite_qids(arc1))}/{op_size}")
            log_fh.write(f"[stratified] Bootstrap round 1: run_{run1_n:03d}, "
                         f"elite_qids={len(elite_qids(arc1))}\n")
        log_fh.flush()

        # ── Round loop ────────────────────────────────────────────────────────
        op_size = summary["original_pool_size"]

        for _ in range(args.max_rounds):
            last         = summary["rounds"][-1]
            next_round_n = last["round"] + 1

            # ── Resume: finish pending eval for most recent round ─────────────
            if last.get("eval_completed") is False:
                print(f"[stratified] Resuming eval for run_{last['run_number']:03d}...")
                log_fh.write(
                    f"[stratified] Resuming eval for run_{last['run_number']:03d}.\n"
                )
                if not _finish_eval(seed, last["run_number"], summary, op_size, log_fh, sf):
                    print("[stratified] Eval failed on resume. Stopping.")
                    log_fh.write("[stratified] Eval failed on resume.\n")
                    slack(
                        f":x: stratified_search FAILED — seed={seed}, "
                        f"round={last['round']}\neval subprocess failed on resume"
                    )
                    break
                last = summary["rounds"][-1]

            # Cumulative excluded = union of deceptive_both_judges from all eval-completed rounds
            excluded: set = set()
            for r in summary["rounds"]:
                if r.get("eval_completed"):
                    excluded.update(r.get("deceptive_both_judges", []))

            # Always filter from the original pool CSV
            next_pool_csv = stratified_pool_csv(seed, next_round_n)
            remaining = write_filtered_pool(original_pool_csv, excluded, next_pool_csv)

            print(f"[stratified] Round {next_round_n}: pool={remaining} "
                  f"(excluded {len(excluded)} QIDs total)")
            log_fh.write(f"[stratified] Round {next_round_n}: pool={remaining}\n")
            log_fh.flush()

            if remaining < args.min_pool_size:
                print(f"[stratified] Pool too small ({remaining} < {args.min_pool_size}). Done.")
                log_fh.write("[stratified] Pool too small. Done.\n")
                break

            # Decide whether the candidate next run dir already has a complete archive.
            current_latest    = latest_run_number(seed)
            candidate_run_n   = current_latest + 1
            candidate_archive = archive_path(seed, candidate_run_n)

            if candidate_archive.exists():
                arc = load_archive(candidate_archive)
                if iter_count(arc) >= args.iterations:
                    print(f"[stratified] run_{candidate_run_n:03d} already complete. "
                          f"Recording and running eval.")
                    log_fh.write(
                        f"[stratified] run_{candidate_run_n:03d} already complete.\n"
                    )
                    summary["rounds"].append(
                        build_round_record(
                            round_n=next_round_n, run_n=candidate_run_n, seed=seed,
                            pool_csv=next_pool_csv, pool_size=remaining,
                            archive=arc,
                            deceptive_qids=set(),
                            cumulative_excluded=excluded,
                            original_pool_size=op_size,
                            eval_completed=False,
                        )
                    )
                    write_summary(sf, summary)
                    if not _finish_eval(seed, candidate_run_n, summary, op_size, log_fh, sf):
                        print("[stratified] Eval failed. Stopping.")
                        log_fh.write("[stratified] Eval failed.\n")
                        slack(
                            f":x: stratified_search FAILED — seed={seed}, "
                            f"round={next_round_n}\neval subprocess failed"
                        )
                        break
                    log_fh.flush()
                    continue
                else:
                    # Partial run — resume by inheriting from it (no --fresh)
                    use_fresh = False
            else:
                # No run dir yet — start fresh
                use_fresh = True

            exit_code = invoke_qd_search(
                seed=seed, pool_csv=next_pool_csv,
                iterations=args.iterations, workers=args.workers,
                no_plot=args.no_plot, fresh_archive=use_fresh,
                log_fh=log_fh,
            )

            if exit_code != 0:
                print(f"[stratified] Subprocess failed (exit={exit_code}). Stopping.")
                log_fh.write(f"[stratified] Subprocess failed (exit={exit_code}).\n")
                slack(
                    f":x: stratified_search FAILED — seed={seed}, round={next_round_n}\n"
                    f"subprocess exit code: {exit_code}"
                )
                break

            new_run_n = latest_run_number(seed)
            new_arc   = load_archive(archive_path(seed, new_run_n))

            summary["rounds"].append(
                build_round_record(
                    round_n=next_round_n, run_n=new_run_n, seed=seed,
                    pool_csv=next_pool_csv, pool_size=remaining,
                    archive=new_arc,
                    deceptive_qids=set(),
                    cumulative_excluded=excluded,
                    original_pool_size=op_size,
                    eval_completed=False,
                )
            )
            write_summary(sf, summary)

            if not _finish_eval(seed, new_run_n, summary, op_size, log_fh, sf):
                print("[stratified] Eval failed. Stopping.")
                log_fh.write("[stratified] Eval failed.\n")
                slack(
                    f":x: stratified_search FAILED — seed={seed}, round={next_round_n}\n"
                    "eval subprocess failed"
                )
                break

            last_r = summary["rounds"][-1]
            fs = fitness_stats(new_arc)
            print(f"[stratified] Round {next_round_n} done. "
                  f"Elite QIDs: {last_r['n_elite_qids']}, "
                  f"deceptive both judges: {last_r['n_deceptive_both']}, "
                  f"fitness mean: {fs['mean']:.3f}, "
                  f"coverage: {last_r['cumulative_excluded_count']}/{op_size} "
                  f"({100 * last_r['cumulative_coverage_fraction']:.1f}%)")
            log_fh.write(
                f"[stratified] Round {next_round_n} done: "
                f"elite_qids={last_r['n_elite_qids']}, "
                f"deceptive={last_r['n_deceptive_both']}, "
                f"coverage={last_r['cumulative_excluded_count']}/{op_size}\n"
            )
            log_fh.flush()

        elapsed = (datetime.datetime.now() - _t_start).total_seconds()
        n_rounds = len(summary["rounds"])
        last_cov = summary["rounds"][-1]
        print(f"[stratified] Finished. Summary: {sf}")
        log_fh.write("[stratified] Finished.\n")
        slack(
            f":white_check_mark: stratified_search done — seed={seed}\n"
            f"rounds completed: {n_rounds}, "
            f"coverage: {last_cov['cumulative_excluded_count']}/{summary['original_pool_size']} "
            f"({100 * last_cov['cumulative_coverage_fraction']:.1f}%), "
            f"elapsed: {elapsed:.0f}s"
        )


if __name__ == "__main__":
    main()
