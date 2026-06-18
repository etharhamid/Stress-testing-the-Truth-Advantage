"""QD-specific configuration. Reads model defaults from the project-wide
`config.py` so a single edit point exists; QDConfig groups paths + thresholds
+ iteration count for the search and eval runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import config as _project_config


# ── Per-seed pool resolution ──────────────────────────────────────────────────
# Maps each baseline seed to its qd_baseline/base_run_{seed}/ directory.
# Each baseline run is named after its own random_seed so the dir name and
# the seed are trivially identifiable. The pool for a per-seed MAP-Elites
# run is that directory's both_correct.csv.
SEED_TO_BASELINE: dict[int, str] = {
    42:  "base_run_42",
    123: "base_run_123",
    7:   "base_run_7",
    13:  "base_run_13",
    73:  "base_run_73",
}


def baseline_dir_for_seed(seed: int, root: str | Path = "qd_baseline") -> Path:
    """Resolve seed → qd_baseline/base_run_{seed}/.

    Tries the static SEED_TO_BASELINE map first. If not found, scans the
    metadata.json files under `root` for a matching `random_seed` (so
    legacy / re-numbered layouts still resolve correctly).
    """
    root = Path(root)
    name = SEED_TO_BASELINE.get(seed)
    if name is not None:
        cand = root / name
        if cand.exists():
            return cand
    for meta in root.glob("base_run_*/metadata.json"):
        try:
            with open(meta) as f:
                data = json.load(f)
        except Exception:
            continue
        if int(data.get("random_seed", -1)) == seed:
            return meta.parent
    raise FileNotFoundError(
        f"No baseline run found for seed={seed} under {root}/"
    )


def pool_csv_for_seed(seed: int, root: str | Path = "qd_baseline") -> Path:
    return baseline_dir_for_seed(seed, root) / "both_correct.csv"


def baseline_swap_for_seed(seed: int, root: str | Path = "qd_baseline") -> dict[str, bool]:
    """Return {qid_str: swap_bool} from the baseline transcripts for a given seed.

    The swap flag assigned in steps/run_debates.py is the authoritative positional
    assignment for each QID. Using it in eval keeps the swap constant across
    baseline, QD, and flat-grid comparisons, so positional bias cannot confound
    per-QID deltas.
    """
    path = baseline_dir_for_seed(seed, root) / "transcripts.jsonl"
    result: dict[str, bool] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            result[str(rec["id"])] = bool(rec["swap"])
    return result


# ── Run-folder helpers ────────────────────────────────────────────────────────

def _run_number(name: str) -> int | None:
    """Return the integer suffix of 'run_NNN', or None if not matching."""
    m = re.fullmatch(r"run_(\d+)", name)
    return int(m.group(1)) if m else None


def latest_run_dir(base: str | Path) -> Path | None:
    """Return the highest-numbered run_NNN directory under base, or None."""
    base = Path(base)
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir() and _run_number(d.name) is not None]
    return max(runs, key=lambda d: _run_number(d.name)) if runs else None  # type: ignore[arg-type]


def next_run_dir(base: str | Path) -> tuple[Path, Path | None]:
    """Create and return the next run_NNN directory under base.

    Returns (new_run_dir, previous_run_dir_or_None). The caller is responsible
    for copying the previous run's archive into new_run_dir if desired."""
    base = Path(base)
    prev = latest_run_dir(base)
    next_num = (_run_number(prev.name) + 1) if prev is not None else 1  # type: ignore[arg-type]
    new_dir = base / f"run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir, prev


# ── Flat-grid run-folder helpers ─────────────────────────────────────────────
# Same shape as _run_number / latest_run_dir / next_run_dir but for the
# `flat_grid_run_NNN/` naming used by run_flat_grid.py. Lives alongside the
# MAP-Elites helpers so both numbering schemes share one resolution module.

def _flat_run_number(name: str) -> int | None:
    """Return the integer suffix of 'flat_grid_run_NNN', or None if not matching."""
    m = re.fullmatch(r"flat_grid_run_(\d+)", name)
    return int(m.group(1)) if m else None


def latest_flat_run_dir(base: str | Path) -> Path | None:
    """Return the highest-numbered flat_grid_run_NNN directory under base, or None."""
    base = Path(base)
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir() and _flat_run_number(d.name) is not None]
    return max(runs, key=lambda d: _flat_run_number(d.name)) if runs else None  # type: ignore[arg-type]


def next_flat_run_dir(base: str | Path) -> tuple[Path, Path | None]:
    """Create and return the next flat_grid_run_NNN directory under base.

    Returns (new_run_dir, previous_run_dir_or_None). Flat-grid runs do not
    inherit anything from previous runs; the prev tuple element is informational
    only (for parity with `next_run_dir`)."""
    base = Path(base)
    prev = latest_flat_run_dir(base)
    next_num = (_flat_run_number(prev.name) + 1) if prev is not None else 1  # type: ignore[arg-type]
    new_dir = base / f"flat_grid_run_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir, prev


# ── Per-pipeline output roots ─────────────────────────────────────────────────
# `qd_results/` is split by pipeline. Each side has its own per-seed tree so
# the two stress-test families can be compared at-a-glance without their
# artefacts intermixing:
#
#   qd_results/
#     map_results/seed_{seed}/run_NNN/ ...      ← run_qd_search / run_qd_eval
#     flat_results/seed_{seed}/flat_grid_run_NNN/ ...  ← run_flat_grid / run_flat_eval
#
# `test=True` swaps both roots to live under qd_results_test/ instead.

def map_results_root(test: bool = False) -> Path:
    """Root directory for all MAP-Elites outputs across seeds.

    Layout:
        qd_results/map_results/
            seed_42/
                run_001/
                run_002/
                orchestrator_logs/      # qd-search / qd-eval teed logs
            seed_7/, seed_13/, ...
            seeds_analysis.md            # cross-seed write-up
            aggregate_summary.json       # when --all-seeds eval runs
    """
    base = Path(_project_config.QD_RESULTS_TEST_DIR if test
                else _project_config.QD_RESULTS_DIR)
    return base / "map_results"


def map_seed_dir(seed: int, *, test: bool = False) -> Path:
    """Per-seed MAP-Elites directory: qd_results/map_results/seed_{seed}/."""
    return map_results_root(test=test) / f"seed_{seed}"


def flat_results_root(test: bool = False) -> Path:
    """Root directory for all flat-grid outputs across seeds.

    Layout:
        qd_results/flat_results/
            seed_42/
                flat_grid_run_001/
                flat_grid_run_002/
                orchestrator_logs/      # flat-grid / flat-eval teed logs
            seed_7/, seed_13/, ...
            flat_seeds_analysis.md
            flat_aggregate_summary.json (when --all-seeds eval runs)
    """
    base = Path(_project_config.QD_RESULTS_TEST_DIR if test
                else _project_config.QD_RESULTS_DIR)
    return base / "flat_results"


def flat_seed_dir(seed: int, *, test: bool = False) -> Path:
    """Per-seed flat-grid directory: qd_results/flat_results/seed_{seed}/."""
    return flat_results_root(test=test) / f"seed_{seed}"


@dataclass
class QDConfig:
    iterations: int = _project_config.QD_DEFAULT_ITERATIONS
    bleu_threshold: float = _project_config.QD_BLEU_THRESHOLD
    max_retries: int = _project_config.QD_MAX_RETRIES_PER_ITER
    seed: int = _project_config.RANDOM_SEED

    questions_pool_csv: str = "always_correct_questions.csv"
    baseline_dir: str = "qd_baseline"

    results_dir: str = _project_config.QD_RESULTS_DIR

    @property
    def search_dir(self) -> str:
        return str(Path(self.results_dir) / "search")

    @property
    def archive_path(self) -> str:
        return str(Path(self.results_dir) / "search" / "archive.json")

    @property
    def search_transcripts_path(self) -> str:
        return str(Path(self.results_dir) / "search" / "search_transcripts.jsonl")

    @property
    def search_judgements_path(self) -> str:
        return str(Path(self.results_dir) / "search" / "search_judgements.jsonl")

    @property
    def search_failures_path(self) -> str:
        return str(Path(self.results_dir) / "search" / "search_failures.jsonl")

    @property
    def plots_dir(self) -> str:
        return str(Path(self.results_dir) / "search" / "plots")


def test_config() -> QDConfig:
    """Smoke-test variant: tiny iteration count, separate output dir."""
    return QDConfig(
        iterations=6,
        results_dir=_project_config.QD_RESULTS_TEST_DIR,
    )
