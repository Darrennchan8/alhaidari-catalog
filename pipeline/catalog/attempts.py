"""Append-only log of chunk translation attempts, for failure statistics, the review queue and
avoiding wasted retries on chunks that fail no matter which model translates them."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from . import config

Outcome = Literal["retry", "accepted", "accepted_with_problem", "rejected"]
Check = Literal["coverage", "words", "timing", "overrun", "shorter", "other"]

_lock = threading.Lock()


class Attempt(BaseModel):
    video: str
    chunk: int
    start: float
    end: float
    model: str
    at: datetime
    kind: Literal["new", "upgrade"]
    outcome: Outcome
    check: Check | None = None  # which check failed, if any
    problem: str | None = None


def check_kind(problem: str | None) -> Check | None:
    """Classify a problem message from the chunk checks."""
    if not problem:
        return None
    p = problem.lower()
    if "no duration" in p:
        return "timing"
    if "overran" in p:
        return "overrun"
    if "much shorter" in p:
        return "shorter"
    if "covers only" in p or "no segments" in p:
        return "coverage"
    if "words for" in p:
        return "words"
    return "other"


def record(
    video: str,
    chunk: int,
    start: float,
    end: float,
    model: str,
    kind: Literal["new", "upgrade"],
    outcome: Outcome,
    problem: str | None = None,
) -> Attempt:
    a = Attempt(
        video=video,
        chunk=chunk,
        start=round(start, 2),
        end=round(end, 2),
        model=model,
        at=datetime.now(UTC),
        kind=kind,
        outcome=outcome,
        check=check_kind(problem),
        problem=problem,
    )
    with _lock:
        config.ATTEMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with config.ATTEMPTS_FILE.open("a", encoding="utf-8") as f:
            f.write(a.model_dump_json() + "\n")
    return a


def load() -> list[Attempt]:
    if not config.ATTEMPTS_FILE.exists():
        return []
    with config.ATTEMPTS_FILE.open(encoding="utf-8") as f:
        return [Attempt.model_validate_json(line) for line in f if line.strip()]


def for_chunk(attempts: Iterable[Attempt], video: str, start: float) -> list[Attempt]:
    """Attempts on one chunk (chunks are identified by video and start time)."""
    return [a for a in attempts if a.video == video and abs(a.start - start) < 0.5]


def failed_models(history: Iterable[Attempt]) -> set[str]:
    """Models that produced a result failing the checks for this chunk."""
    return {a.model for a in history if a.check is not None and a.check != "shorter"}
