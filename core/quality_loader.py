# core/quality_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads the QuALITY dataset and applies the exact same filtering logic as
# core/load/quality.py in the Khan et al. repo.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import random
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, field_validator

# ── Excluded content (same as the repo) ──────────────────────────────────────
EXCLUDED_STORIES = [
    "Boys Do Bleed",
    "MONICA!",
    "The Girl in His Mind",
]

EXCLUDED_QUESTIONS = [
    ("What is the relationship between the humans and the Belphins?", "The Blue Tower"),
    ("What will happen if Anne becomes pregnant?", "Conditionally Human"),
    (
        "Why does Adrian think the Callistans will be willing to fight against the league?",
        "Conspiracy on Callisto",
    ),
    (
        "All of the following terms describe how Infield would characterize Price EXCEPT for:",
        "Name Your Symptom",
    ),
    ("How did the Mafia grow the business of prostitution on Mars?", "Mars Confidential"),
    ("What crime has Zeckler committed to warrant imprisonment?", "Letter of the Law"),
    (
        "The humans in the fourth dimension acquire all of the following remarkable abilities EXCEPT for:",
        "Judas Ram",
    ),
    (
        "The Movement believes all of the following EXCEPT: Questioning the failings of the old society, "
        "failings have put them in the dome; failure of foreign policy (self-containment)",
        "A Fall of Glass",
    ),
    ("What is the Boyar's ultimate goal for Flamme?", "The Desert and the Stars"),
]

# Articles used in the NYU human experiments — excluded to avoid data leakage
NYU_ARTICLE_IDS = [
    "63477", "61007", "63109", "62569", "55933", "61499", "63899", "52844",
    "63640", "63631", "50893", "61090", "63527", "62619", "62198", "61053",
    "63890", "53269", "61380", "51201", "43046", "30029", "63523", "63401",
    "61285", "62314", "61430", "62085", "61119", "61467", "52855", "60412",
    "63855", "63633", "61434", "61146", "63862", "63392", "63130", "62382",
    "63833", "20002", "63150", "63473", "51483", "51461", "50818", "51027",
    "51267", "51351", "51126", "51320", "51395", "51274", "51650", "51605",
    "51150", "51295", "51688", "51256",
]


# ── Pydantic models (minimal subset needed) ───────────────────────────────────
class ValidationUntimed(BaseModel):
    untimed_annotator_id: str
    untimed_answer: str
    untimed_eval1_answerability: int
    untimed_eval2_context: int
    untimed_best_distractor: int

    @field_validator("untimed_answer", mode="before")
    @classmethod
    def _coerce_untimed_answer(cls, v: object) -> str:
        # QuALITY JSON sometimes stores choice index as int; downstream uses int(answer).
        return str(v)


class SpeedValidation(BaseModel):
    speed_annotator_id: str
    speed_answer: str

    @field_validator("speed_answer", mode="before")
    @classmethod
    def _coerce_speed_answer(cls, v: object) -> str:
        return str(v)


class QualityQuestion(BaseModel):
    question: str
    options: list[str]
    gold_label: int
    writer_label: int
    validation: list[ValidationUntimed]
    speed_validation: list[SpeedValidation]
    difficult: int


class QualityArticle(BaseModel):
    article_id: str
    set_unique_id: str
    batch_num: str
    writer_id: str
    source: str
    title: str
    author: str
    topic: str
    url: str
    year: Optional[int] = None
    license: str
    article: str
    questions: list[QualityQuestion]
    split: Optional[str] = None


# ── Helper metrics (same as repo) ────────────────────────────────────────────
def answerability(q: QualityQuestion) -> float:
    return sum(v.untimed_eval1_answerability for v in q.validation) / len(q.validation)


def untimed_accuracy(q: QualityQuestion, gold: int) -> float:
    return sum(1 for v in q.validation if int(v.untimed_answer) == gold) / len(q.validation)


def speed_accuracy(q: QualityQuestion, gold: int) -> float:
    return sum(1 for v in q.speed_validation if int(v.speed_answer) == gold) / len(
        q.speed_validation
    )


def context_required(q: QualityQuestion) -> float:
    return sum(v.untimed_eval2_context for v in q.validation) / len(q.validation)


def best_distractor(q: QualityQuestion) -> Optional[str]:
    """Return the answer option that most annotators chose as the best distractor,
    or None if no annotator flagged a non-gold option (all agreed on gold).
    """
    distractors = [
        v.untimed_best_distractor
        for v in q.validation
        if v.untimed_best_distractor != q.gold_label
    ]
    if not distractors:
        return None
    best_idx = max(set(distractors), key=distractors.count) - 1  # 1-indexed → 0-indexed
    return q.options[best_idx]


def correct_answer(q: QualityQuestion) -> str:
    return q.options[q.gold_label - 1]  # 1-indexed → 0-indexed


def incompatible(answer: str) -> bool:
    lower = answer.lower()
    return any(
        phrase in lower
        for phrase in ["all of the above", "neither of these", "none of the above"]
    )




# ── Dataset download (persistent cache under $XDG_CACHE_HOME) ─────────────────
def _cache_dir() -> Path:
    """Persistent cache root. Respects XDG_CACHE_HOME; otherwise ~/.cache/."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "baseline_debate" / "quality"


def fetch_split(split: str = "train") -> list[QualityArticle]:
    name = f"QuALITY.v1.0.1.htmlstripped.{split}"
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / name

    if not cache.exists():
        url = (
            f"https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/{name}"
        )
        print(f"  Downloading QuALITY {split} split …")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        text = (
            resp.text.replace("\u2028", "")
            .replace("\u2029", "")
            .replace("\xa0", " ")
        )
        cache.write_text(text, encoding="utf-8")
    else:
        print(f"  Using cached QuALITY {split} split from {cache}")

    articles = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            articles.append(QualityArticle.model_validate_json(line))
    return articles


# ── Filter function (mirrors filter_question in the repo) ─────────────────────
def passes_filter(
    q: QualityQuestion,
    article: QualityArticle,
    *,
    sources: list[str],
    difficulty: int,
    max_answerability: float,
    min_untimed_accuracy: float,
    max_speed_accuracy: float,
    min_context_required: float,
    skip_conflicting: bool,
    ignore_nyu: bool,
    questions_to_avoid: list[tuple],
    stories_to_avoid: list[str],
) -> bool:
    if sources and article.source not in sources:
        return False
    if difficulty is not None and q.difficult != difficulty:
        return False
    if ignore_nyu and article.article_id in NYU_ARTICLE_IDS:
        return False
    if article.title.strip() in stories_to_avoid:
        return False
    if (q.question.strip(), article.title.strip()) in questions_to_avoid:
        return False

    gold = q.gold_label
    if answerability(q) > max_answerability:
        return False
    if untimed_accuracy(q, gold) < min_untimed_accuracy:
        return False
    if speed_accuracy(q, gold) > max_speed_accuracy:
        return False
    if context_required(q) < min_context_required:
        return False
    if skip_conflicting and q.writer_label != q.gold_label:
        return False

    ca = correct_answer(q)
    nd = best_distractor(q)
    if nd is None:
        return False
    if ca == nd:
        return False
    if incompatible(ca) or incompatible(nd):
        return False
    return True


# ── Deduplication (mirrors deduplicate_stories) ───────────────────────────────
def deduplicate(rows: list[dict], max_per_story: int) -> list[dict]:
    counts: dict[str, int] = {}
    out = []
    for row in rows:
        title = row["story_title"]
        counts[title] = counts.get(title, 0)
        if counts[title] < max_per_story:
            counts[title] += 1
            out.append(row)
    return out


# ── Public entry point ─────────────────────────────────────────────────────────
def load_questions(
    splits: list[str] | None = None,
    sources: list[str] | None = None,
    difficulty: int = 1,
    max_answerability: float = 1.0,
    min_untimed_accuracy: float = 1.0,
    max_speed_accuracy: float = 0.5,
    min_context_required: float = 1.5,
    skip_conflicting: bool = True,
    ignore_nyu: bool = False,
    max_from_same_story: int = 1,
    limit: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Download, filter, deduplicate and return `limit` questions as plain dicts
    with the fields expected by the debate pipeline:
        id, question, correct_answer, negative_answer, story, story_title,
        question_set_id
    """
    if splits is None:
        splits = ["train", "dev"]
    if sources is None:
        sources = ["Gutenberg"]
    questions_to_avoid = [(q.strip(), s.strip()) for q, s in EXCLUDED_QUESTIONS]
    stories_to_avoid = [s.strip() for s in EXCLUDED_STORIES]

    rows: list[dict] = []

    for split in splits:
        articles = fetch_split(split)
        for article in articles:
            for q in article.questions:
                if not passes_filter(
                    q,
                    article,
                    sources=sources,
                    difficulty=difficulty,
                    max_answerability=max_answerability,
                    min_untimed_accuracy=min_untimed_accuracy,
                    max_speed_accuracy=max_speed_accuracy,
                    min_context_required=min_context_required,
                    skip_conflicting=skip_conflicting,
                    ignore_nyu=ignore_nyu,
                    questions_to_avoid=questions_to_avoid,
                    stories_to_avoid=stories_to_avoid,
                ):
                    continue

                rows.append(
                    {
                        "id": len(rows),
                        "question": q.question,
                        "correct_answer": correct_answer(q),
                        "negative_answer": best_distractor(q),
                        "story": article.article,
                        "story_title": article.title,
                        "question_set_id": article.set_unique_id,
                        "split": split,
                    }
                )

    # Use a local Random instance so we don't mutate the process-wide RNG
    # as a side effect of loading questions.
    rng = random.Random(seed)
    rng.shuffle(rows)

    rows = deduplicate(rows, max_from_same_story)

    for i, row in enumerate(rows):
        row["id"] = i

    rows = rows[:limit]
    print(f"  Loaded {len(rows)} questions after filtering.")
    return rows
