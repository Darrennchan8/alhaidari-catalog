"""Review queue: transcript chunks a person should look at, and their decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel

from . import attempts, config, quality, store
from .models import ChunkInfo, Transcript

# A chunk that fails checks with this many different models is probably an audio problem
# (recitation, music, silence, crosstalk, non-Arabic speech) rather than a model problem.
FAILING_MODELS_FOR_REVIEW = 2


class Decision(BaseModel):
    status: str  # "ok" (checked, nothing to fix) or "fix" (needs a manual correction)
    model: str  # the chunk's model when reviewed; a later re-translation reopens the review
    note: str = ""
    at: datetime


@dataclass
class Flag:
    video: str
    chunk: int
    info: ChunkInfo
    reasons: list[str] = field(default_factory=list)
    decision: Decision | None = None

    @property
    def key(self) -> str:
        return chunk_key(self.video, self.info.start)

    @property
    def url(self) -> str:
        return f"https://youtu.be/{self.video}?t={int(self.info.start)}"


def chunk_key(video: str, start: float) -> str:
    return f"{video}@{int(start)}"


def load_decisions() -> dict[str, Decision]:
    if not config.REVIEW_FILE.exists():
        return {}
    raw: dict = store.read_json(config.REVIEW_FILE)  # type: ignore[assignment]
    return {k: Decision.model_validate(v) for k, v in raw.items()}


def save_decision(video: str, start: float, model: str, status: str, note: str = "") -> None:
    decisions = load_decisions()
    decisions[chunk_key(video, start)] = Decision(
        status=status, model=model, note=note, at=datetime.now(UTC)
    )
    store.write_json(
        config.REVIEW_FILE, {k: v.model_dump(mode="json") for k, v in sorted(decisions.items())}
    )


def chunk_reasons(video: str, info: ChunkInfo, history: list[attempts.Attempt]) -> list[str]:
    """Why this chunk needs a person's attention (empty if it doesn't)."""
    reasons: list[str] = []
    failed = attempts.failed_models(history)
    if info.problem and quality.is_best(info.model):
        reasons.append(f"best model still fails checks: {info.problem}")
    elif info.problem and len(failed) >= FAILING_MODELS_FOR_REVIEW:
        reasons.append(f"fails checks with {len(failed)} models ({', '.join(sorted(failed))})")
    rejected = [
        a
        for a in history
        if a.outcome == "rejected" and a.check == "shorter" and a.at > info.translated_at
    ]
    if rejected:
        reasons.append(f"better model's translation rejected as shorter: {rejected[-1].problem}")
    return reasons


def flags(
    transcripts: dict[str, Transcript],
    history: list[attempts.Attempt] | None = None,
    decisions: dict[str, Decision] | None = None,
    include_resolved: bool = False,
) -> list[Flag]:
    """Chunks needing review. A decision hides a flag until the chunk is re-translated."""
    history = attempts.load() if history is None else history
    decisions = load_decisions() if decisions is None else decisions
    out: list[Flag] = []
    for vid, t in sorted(transcripts.items()):
        for i, info in enumerate(t.chunk_info):
            reasons = chunk_reasons(vid, info, attempts.for_chunk(history, vid, info.start))
            if not reasons:
                continue
            d = decisions.get(chunk_key(vid, info.start))
            if d and d.model != info.model:
                d = None  # re-translated since the review
            if d and not include_resolved:
                continue
            out.append(Flag(video=vid, chunk=i, info=info, reasons=reasons, decision=d))
    return out


def reader_notice(info: ChunkInfo, decision: Decision | None) -> str | None:
    """A short note shown to readers above a transcript section with known problems."""
    if not info.problem or (decision and decision.status == "ok" and decision.model == info.model):
        return None
    kind = attempts.check_kind(info.problem)
    if kind in ("timing", "overrun"):
        return "Timestamps in this section may be inaccurate."
    if kind in ("coverage", "words"):
        return "Parts of this section may be missing from the translation."
    return "This section may contain translation errors."
