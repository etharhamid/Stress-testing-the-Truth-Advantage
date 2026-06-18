# Stress-testing the Truth Advantage

Code release for experiments that evaluate the robustness of language-model debate under adversarial question framing.

The repository contains code for:

- A consultancy baseline.
- A simultaneous debate baseline.
- Stage 1: flat per-question adversarial framing search.
- Stage 2: per-question 1D MAP-Elites over mutation types.
- Stage 3: per-question 2D MAP-Elites over question type and mutation type.

This release intentionally excludes generated results, plots, thesis files, private notes, logs, and credentials.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file or export the required environment variables:

```bash
GOOGLE_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=us-central1
```

For higher-throughput runs, the code also supports comma-separated pools:

```bash
GOOGLE_API_KEYS=key1,key2
GOOGLE_CLOUD_PROJECTS=project1,project2
```

The `.env` file is ignored by git.

## Main Commands

Run the debate baseline:

```bash
python run_all.py
```

Run the consultancy baseline:

```bash
python run_consultancy_baseline.py
```

Run Stage 1 flat-grid search:

```bash
python run_flat_grid.py --seed 42
```

Evaluate Stage 1:

```bash
python run_flat_eval.py --seed 42
```

Run Stage 2 per-QID 1D MAP-Elites:

```bash
python run_stratified_search.py --mode per_qid --seed 42
```

Run Stage 3 per-QID 2D MAP-Elites:

```bash
python run_stratified_search.py --mode per_qid_2d --seed 42
```

Evaluate MAP-Elites outputs:

```bash
python run_qd_eval.py --seed 42
```

## Code Layout

```text
core/       debate, consultancy, judging, prompts, Gemini client
steps/      baseline preparation, debate, judging, scoring
qd/         QD archive, selection, mutation, validation, fitness, search helpers
```

Generated outputs are written to directories such as `results/`, `qd_baseline/`, and `qd_results/`, all of which are ignored by git in this release.

## Notes

The code expects QuALITY data access through the local data-loading path used by `core/quality_loader.py`. Generated experiment outputs are not included in this public code release.

