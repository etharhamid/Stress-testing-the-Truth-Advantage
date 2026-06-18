# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the baseline debate experiment (local runs).
#
# Auth is split by role:
#   - Debater  → Google AI Studio (API key, GOOGLE_API_KEY env var).
#                Broader preview-model access (e.g. gemini-3.1-pro-preview).
#   - Judge    → Vertex AI (Application Default Credentials).
#                Required for `response_logprobs`, which is Vertex-only on
#                Gemini 2.5 / 3.x.
# Setup:
#     gcloud auth application-default login               # Vertex (judge)
#     # Create an API key at https://aistudio.google.com/app/apikey
# then set in shell or in a `.env` next to this file:
#     GOOGLE_API_KEY=<ai-studio-key>
#     GOOGLE_CLOUD_PROJECT=<gcp-project>
#     GOOGLE_CLOUD_LOCATION=us-central1   # optional, defaults to us-central1
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import sys
from pathlib import Path


def _caffeinate_self() -> None:
    """On macOS, re-exec the current entry script under `caffeinate -i` so a
    laptop sleep doesn't pause hours-long runs. We hold `i` (idle-prevent)
    only, so the display can still sleep; the CPU stays awake.

    Gated by:
      - macOS only (`sys.platform == 'darwin'`).
      - `caffeinate` must be on PATH (always true on stock macOS).
      - Only wraps when the entry script is a `run_*.py` file at the
        process root; library imports, REPL sessions, tests, and
        ad-hoc one-off scripts pass through unchanged.
      - Env-var `CAFFEINATED=1` prevents recursion in the re-execed
        child. Env-var `NO_CAFFEINATE=1` opts out entirely (useful on
        battery or when running inside a wrapper that already holds
        a wake assertion).

    Several caffeinate processes (e.g. from `scripts/run_all_seeds.py`
    spawning N children) coexist without conflict; each holds its own
    power assertion and macOS unions them.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("CAFFEINATED") == "1":
        return
    if os.environ.get("NO_CAFFEINATE") == "1":
        return
    if not shutil.which("caffeinate"):
        return
    if not sys.argv:
        return
    main_script = os.path.basename(sys.argv[0])
    if not (main_script.startswith("run_") and main_script.endswith(".py")):
        return
    os.environ["CAFFEINATED"] = "1"
    print(f"[caffeinate] wrapping {main_script} under `caffeinate -i` "
          f"(set NO_CAFFEINATE=1 to disable)", flush=True)
    os.execvp("caffeinate", ["caffeinate", "-i", sys.executable, "-u", *sys.argv])


_caffeinate_self()


def _load_local_env() -> None:
    """
    Load `.env` from this package directory (same folder as config.py), not from
    the process cwd — so entry-point scripts always pick up settings no matter
    where you run the command from.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return
    env_file = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_file)


_load_local_env()

# ── Models (Gemini model ids) ───────────────────────────────────────────────
# Strong debaters (truthful + deceptive): same backbone.
DEBATER_MODEL = "gemini-3.1-pro-preview"
# Weaker judge model.
JUDGE_MODEL = "gemini-2.0-flash"

# ── Vertex AI target (judge path) ───────────────────────────────────────────
GCP_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip()

# ── AI Studio target (debater path) ─────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()


# ── Credential pools (optional, for cross-project / multi-key throughput) ──
def _parse_csv_env(name: str, fallback: str) -> list[str]:
    """Parse a comma-separated env var into a list, falling back to the
    singleton when unset. Existing single-project / single-key setups keep
    working unchanged because the singleton becomes a 1-entry pool."""
    raw = os.environ.get(name, "").strip()
    if raw:
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [fallback] if fallback else []


# Vertex AI: judge-profile clients (logprobs). One client per project.
GCP_PROJECTS: list[str] = _parse_csv_env("GOOGLE_CLOUD_PROJECTS", GCP_PROJECT)

# AI Studio: debater / qd_validator / eval_judge profiles. One client per key.
GOOGLE_API_KEYS: list[str] = _parse_csv_env("GOOGLE_API_KEYS", GOOGLE_API_KEY)


def set_seed(seed: int) -> None:
    global RANDOM_SEED
    RANDOM_SEED = seed


def set_judge_model(model: str) -> None:
    global JUDGE_MODEL
    JUDGE_MODEL = model


def set_judge_output_directory(judge_dir: str) -> None:
    """Redirect judge-phase outputs without touching QUESTIONS_FILE or TRANSCRIPTS_FILE."""
    global RESULTS_DIR, JUDGEMENTS_FILE, BASELINE_RESULTS_FILE, SUMMARY_FILE
    j = os.path.normpath(judge_dir)
    RESULTS_DIR = j
    JUDGEMENTS_FILE = os.path.join(j, "judgements.jsonl")
    BASELINE_RESULTS_FILE = os.path.join(j, "baseline_results.csv")
    SUMMARY_FILE = os.path.join(j, "summary.json")


def validate_llm_credentials() -> None:
    """Fail fast if either backend's credentials are missing.

    - AI Studio (debater)  : requires GOOGLE_API_KEY *or* GOOGLE_API_KEYS.
    - Vertex AI  (judge)   : requires GOOGLE_CLOUD_PROJECT *or*
      GOOGLE_CLOUD_PROJECTS; ADC is checked lazily by the SDK on the first
      API call.
    """
    missing = []
    if not GOOGLE_API_KEYS:
        missing.append(
            "GOOGLE_API_KEY or GOOGLE_API_KEYS (AI Studio — debater). "
            "Create one at https://aistudio.google.com/app/apikey and add "
            "it to .env (comma-separated for multi-key pooling)."
        )
    if not GCP_PROJECTS:
        missing.append(
            "GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECTS (Vertex AI — "
            "judge). Set in shell or .env (comma-separated for multi-"
            "project pooling), and run `gcloud auth application-default "
            "login` once."
        )
    if missing:
        raise RuntimeError("Missing LLM credentials:\n  - " + "\n  - ".join(missing))

# ── Dataset (QuALITY) ────────────────────────────────────────────────────────
NUM_QUESTIONS         = 100
RANDOM_SEED           = 42

# Filters — matched to core/config/config.yaml in the Khan et al. repo
SOURCES               = ["Gutenberg"]   # only Gutenberg stories
DIFFICULTY            = 1               # "hard" questions (speed accuracy < 0.5)
MAX_ANSWERABILITY     = 1.0             # single unambiguous correct answer
MIN_UNTIMED_ACCURACY  = 1.0             # all untimed annotators correct
MAX_SPEED_ACCURACY    = 0.5             # ≤ 50 % timed annotators correct
MIN_CONTEXT_REQUIRED  = 1.5             # needs substantial story context
SKIP_CONFLICTING      = True            # writer label must == gold label
IGNORE_NYU            = False           # paper does NOT exclude NYU articles
MAX_FROM_SAME_STORY   = 1              # at most 1 question per story (diversity)

# ── Debate settings ──────────────────────────────────────────────────────────
NUM_ROUNDS       = 3      # Round 1: Opening, Round 2: Rebuttal, Round 3: Final
WORD_LIMIT       = 100    # soft limit mentioned in debater prompts
WORD_LIMIT_HARD  = 150    # hard truncation (grace margin above WORD_LIMIT)

# ── Output paths ─────────────────────────────────────────────────────────────
# Default layout (overridden by run_all.py per experiment).
RESULTS_DIR           = "results"
QUESTIONS_FILE        = "results/questions.csv"
TRANSCRIPTS_FILE      = "results/transcripts.jsonl"
JUDGEMENTS_FILE       = "results/judgements.jsonl"
BASELINE_RESULTS_FILE = "results/baseline_results.csv"
SUMMARY_FILE          = "results/summary.json"


def set_run_output_directory(run_dir: str) -> None:
    """
    Point all step outputs (questions, JSONL, CSV, summary) under run_dir.
    `run_dir` may be absolute or relative to the process cwd.
    """
    global RESULTS_DIR, QUESTIONS_FILE, TRANSCRIPTS_FILE, JUDGEMENTS_FILE
    global BASELINE_RESULTS_FILE, SUMMARY_FILE
    r = os.path.normpath(run_dir)
    RESULTS_DIR = r
    QUESTIONS_FILE = os.path.join(r, "questions.csv")
    TRANSCRIPTS_FILE = os.path.join(r, "transcripts.jsonl")
    JUDGEMENTS_FILE = os.path.join(r, "judgements.jsonl")
    BASELINE_RESULTS_FILE = os.path.join(r, "baseline_results.csv")
    SUMMARY_FILE = os.path.join(r, "summary.json")


# ── QD experiment settings ──────────────────────────────────────────────────
# Models per role. AI Studio = API key, Vertex = ADC (logprobs).
QD_MUTATOR_MODEL          = "gemma-4-31b-it"         # AI Studio
QD_VALIDATOR_MODEL        = "gemini-2.5-flash-lite" # AI Studio
QD_SEARCH_DEBATER_MODEL   = "gemini-2.5-flash-lite" # AI Studio
QD_SEARCH_JUDGE_MODEL     = "gemini-2.5-flash-lite" # Vertex (logprobs)
QD_EVAL_DEBATER_MODEL     = "gemini-3.1-pro-preview" # AI Studio (== baseline)
QD_EVAL_JUDGE_MODEL       = "gemini-2.5-flash"      # Vertex (== baseline)

QD_RESULTS_DIR            = "qd_results"
QD_RESULTS_TEST_DIR       = "qd_results_test"
QD_BLEU_THRESHOLD         = 0.8
QD_MAX_RETRIES_PER_ITER   = 3
QD_DEFAULT_ITERATIONS     = 500
QD_MUTATOR_TEMPERATURE    = 0.7
QD_VALIDATOR_TEMPERATURE  = 0.0



def set_debater_model(model: str) -> None:
    global DEBATER_MODEL
    DEBATER_MODEL = model


# ── Consultancy experiment settings ─────────────────────────────────────────
# Consultant arguments use the same WORD_LIMIT / WORD_LIMIT_HARD as debate.
# Judge in-round questions have their own limit (no existing equivalent).
CONSULTANCY_JUDGE_QUESTION_WORD_LIMIT = 100   # word limit for judge in-round questions
