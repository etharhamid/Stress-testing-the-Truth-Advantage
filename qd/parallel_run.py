"""Cross-seed subprocess orchestration helpers.

Two primitives, both used by `scripts/run_all_seeds.py` and by the
`--all-seeds` modes of `run_qd_eval.py` / `run_flat_eval.py`:

* `tee_subprocess(argv, log_path, prefix)` — launch one subprocess; tee its
  stdout (with stderr merged) line-by-line to both `log_path` and the parent
  process's stdout, prefixed with `prefix`. Returns the subprocess's exit
  code. Handles SIGINT in the parent by forwarding it to the child once,
  then escalating to SIGTERM if the child hasn't exited after a grace
  period.

* `run_seeds_in_parallel(seeds, build_argv, log_path_for, prefix_for,
  parallel_workers)` — run one `tee_subprocess` per seed concurrently via a
  `ThreadPoolExecutor`. Returns `[(seed, returncode), ...]` in completion
  order. On Ctrl-C in the parent, every in-flight child receives SIGINT and
  the executor drains cleanly.

Resume safety is unchanged: each subprocess writes to its own per-seed
folder and uses the existing per-script resume protocol (CLAUDE.md §7), so
re-running the same orchestrator command picks up wherever each seed left
off — no extra state lives in this module.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable


# Seconds to wait after sending SIGINT to a child before escalating to SIGTERM.
# The qd_search / flat_grid scripts close JsonlWriters and save archive.json
# inside `finally` blocks; that cleanup is normally fast (<1s), but a worker
# blocked on a 300s judge call needs slack to wind down.
_SIGINT_GRACE_SECONDS = 30.0


def _signal_process_tree(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's process group when possible, else the child itself."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _interrupt_child(proc: subprocess.Popen) -> None:
    """Ask a child process group to stop, then escalate if it lingers."""
    _signal_process_tree(proc, signal.SIGINT)
    try:
        proc.wait(timeout=_SIGINT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_tree(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_process_tree(proc, signal.SIGKILL)


def _stream_to_two_sinks(
    proc: subprocess.Popen,
    log_path: Path,
    prefix: str,
) -> None:
    """Read proc.stdout line-by-line; write to log file and parent stdout.
    Returns when the stream closes (i.e. the child exits)."""
    # Line-buffered file write; flush after every line so `tail -f` works.
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} :: "
                   f"{' '.join(proc.args)}\n")
        logf.flush()
        assert proc.stdout is not None
        for line in proc.stdout:
            # `print` here goes to the orchestrator's own stdout, not the
            # subprocess's. The flush=True keeps cross-seed lines interleaved
            # in roughly real-time order.
            print(f"{prefix} {line.rstrip()}", flush=True)
            logf.write(line)
            logf.flush()


def tee_subprocess(
    argv: list[str],
    log_path: Path,
    prefix: str,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    on_exit: Callable[[subprocess.Popen], None] | None = None,
) -> int:
    """Run one subprocess; tee its merged stdout/stderr to `log_path` and
    parent stdout. Returns the child's exit code.

    Forwards SIGINT once on KeyboardInterrupt, then escalates to SIGTERM
    after `_SIGINT_GRACE_SECONDS` if the child hasn't exited.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Force the child's Python stdout to be line-flushed instead of
    # block-buffered. Without this, child print() output is invisible to
    # the parent until the OS pipe buffer (~64 KB) fills — which can take
    # minutes for a slow producer like a search loop, masking live
    # progress and making the per-seed log appear stalled.
    child_env = dict(env) if env is not None else dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line buffered on the parent's read side
        text=True,
        env=child_env,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    if on_start is not None:
        on_start(proc)

    # Streaming runs in this thread; the subprocess runs in its own process.
    try:
        _stream_to_two_sinks(proc, log_path, prefix)
    except KeyboardInterrupt:
        # Forward SIGINT once, then wait briefly, then escalate.
        _interrupt_child(proc)
        raise
    finally:
        rc = proc.wait()
        if on_exit is not None:
            on_exit(proc)
    return rc


def run_seeds_in_parallel(
    seeds: Iterable[int],
    *,
    build_argv: Callable[[int], list[str]],
    log_path_for: Callable[[int], Path],
    prefix_for: Callable[[int], str],
    parallel_workers: int,
    env: dict[str, str] | None = None,
) -> list[tuple[int, int]]:
    """Run one subprocess per seed concurrently via a `ThreadPoolExecutor`.

    Args:
      seeds: iterable of integer seeds to dispatch.
      build_argv: seed → full subprocess argv (including python + script + args).
      log_path_for: seed → path where this child's teed log lives. Parent
        dirs are created on demand.
      prefix_for: seed → short prefix prepended to each line on the
        orchestrator's stdout. Typically f"[seed {seed:>3}]".
      parallel_workers: number of subprocesses allowed to run concurrently.
      env: optional env dict passed to every child; when None the child
        inherits the orchestrator's environment (so GOOGLE_API_KEYS /
        GOOGLE_CLOUD_PROJECTS propagate naturally).

    Returns:
      List of (seed, returncode) pairs in completion order. Use the
      returncodes to decide overall exit status — this helper never raises
      on a non-zero child exit.

    On KeyboardInterrupt: the in-flight children receive SIGINT (via
    `tee_subprocess`'s exception path), the executor cancels pending
    futures, and the exception is re-raised after cleanup.
    """
    seed_list = list(seeds)
    results: list[tuple[int, int]] = []
    results_lock = threading.Lock()
    active_procs: set[subprocess.Popen] = set()
    active_lock = threading.Lock()

    def _register_proc(proc: subprocess.Popen) -> None:
        with active_lock:
            active_procs.add(proc)

    def _unregister_proc(proc: subprocess.Popen) -> None:
        with active_lock:
            active_procs.discard(proc)

    def _interrupt_active_children() -> None:
        with active_lock:
            procs = list(active_procs)
        for proc in procs:
            if proc.poll() is None:
                _interrupt_child(proc)

    def _one(seed: int) -> tuple[int, int]:
        argv = build_argv(seed)
        log_path = log_path_for(seed)
        prefix = prefix_for(seed)
        rc = tee_subprocess(
            argv,
            log_path,
            prefix,
            env=env,
            on_start=_register_proc,
            on_exit=_unregister_proc,
        )
        return seed, rc

    with ThreadPoolExecutor(max_workers=max(1, parallel_workers)) as ex:
        futs = {ex.submit(_one, s): s for s in seed_list}
        try:
            for fut in as_completed(futs):
                seed, rc = fut.result()
                with results_lock:
                    results.append((seed, rc))
        except KeyboardInterrupt:
            # The signal is delivered to the main thread, not reliably to the
            # worker threads streaming child output, so stop running children
            # from here and cancel any not-yet-started futures.
            _interrupt_active_children()
            for f in futs:
                f.cancel()
            raise
    return results


def default_log_path(
    seed: int,
    subcommand: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Standard per-seed orchestrator log location:
       qd_results/seed_{seed}/orchestrator_logs/{subcommand}_{ts}.log

    Used by both `scripts/run_all_seeds.py` and the eval scripts' parallel
    `--all-seeds` modes so logs land in a consistent place regardless of
    which entry point spawned them. The orchestrator can't know the
    run_NNN/ folder up front (the subprocess creates it lazily), hence the
    sidecar location."""
    root = base_dir if base_dir is not None else Path("qd_results")
    log_dir = root / f"seed_{seed}" / "orchestrator_logs"
    ts = time.strftime("%Y%m%dT%H%M%S")
    return log_dir / f"{subcommand}_{ts}.log"
