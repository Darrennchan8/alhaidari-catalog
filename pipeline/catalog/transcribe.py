"""Stage: audio → timestamped English transcript, via Gemini, chunk by chunk."""

from __future__ import annotations

import contextlib
import itertools
import logging
import re
import shutil
from datetime import UTC, datetime
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field

from . import attempts, audio, config, prompts, quality, review, store, ytdlp
from .gemini import Gemini, QuotaExhausted
from .models import ChunkInfo, Segment, Transcript, VideoMeta

log = logging.getLogger(__name__)


class _OutSegment(BaseModel):
    start: str = Field(description="MM:SS from the start of this clip")
    end: str = Field(description="MM:SS from the start of this clip")
    speaker: str
    kind: Literal["speech", "quran", "hadith", "poetry", "dua", "other"]
    text: str = Field(description="Faithful English translation")
    terms: list[str]


class _ChunkOut(BaseModel):
    segments: list[_OutSegment]


_TS = re.compile(r"^\s*(?:(\d+):)?(\d{1,3}):(\d{1,2}(?:\.\d+)?)\s*$")


def parse_ts(value: str) -> float:
    m = _TS.match(value)
    if not m:
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"bad timestamp {value!r}") from e
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + float(s)


def fmt_ts(seconds: float) -> str:
    s = round(seconds)
    h, rem = divmod(s, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60:02d}:{rem % 60:02d}"


# Models sometimes run their clock fast and emit timestamps past the end of the clip.
OVERRUN_TOLERANCE = 1.03


def timing_overrun(out: _ChunkOut, chunk: audio.Chunk) -> float:
    """How far the model's clock overran the clip: latest timestamp / clip length, or 1.0 when
    there is no systematic overrun. At least 3 segments must start past the end; a single bad
    timestamp is a glitch and is simply clamped."""
    length = chunk.end - chunk.start
    if length <= 0:
        return 1.0
    stamps: list[float] = []
    starts_beyond = 0
    for o in out.segments:
        with contextlib.suppress(ValueError):
            a = parse_ts(o.start)
            stamps.append(a)
            starts_beyond += a > length
        with contextlib.suppress(ValueError):
            stamps.append(parse_ts(o.end))
    if starts_beyond < 3:
        return 1.0
    return max(stamps) / length


def offset_segments(out: _ChunkOut, chunk: audio.Chunk) -> list[Segment]:
    """Convert clip-relative timestamps to absolute ones, clamped to the chunk and monotonic.
    If the model's timestamps overran the clip, they are first rescaled to fit it."""
    length = chunk.end - chunk.start
    overrun = timing_overrun(out, chunk)
    scale = 1 / overrun if overrun > OVERRUN_TOLERANCE else 1.0
    segs: list[Segment] = []
    prev_end = 0.0
    for o in out.segments:
        text = o.text.strip()
        if not text:
            continue
        try:
            a, b = parse_ts(o.start) * scale, parse_ts(o.end) * scale
        except ValueError:
            a, b = prev_end, prev_end
        a = min(max(a, prev_end), length)
        b = min(max(b, a), length)
        segs.append(
            Segment(
                start=round(chunk.start + a, 2),
                end=round(chunk.start + b, 2),
                speaker=o.speaker.strip() or None,
                kind=o.kind,
                text=text,
                terms=[t.strip() for t in o.terms if t.strip()][:4],
            )
        )
        prev_end = b
    # Close small gaps so the transcript tiles the timeline; the last segment ends the chunk.
    for cur, nxt in itertools.pairwise(segs):
        if nxt.start - cur.end < 5:
            cur.end = nxt.start
    if segs and length - (segs[-1].end - chunk.start) < 30:
        segs[-1].end = round(chunk.end, 2)
    return segs


def check_coverage(segs: list[Segment], chunk: audio.Chunk) -> str | None:
    """Return a problem description if the chunk transcript looks truncated."""
    length = chunk.end - chunk.start
    if not segs:
        return "no segments"
    collapsed = sum(1 for s in segs if s.end - s.start < 0.5)
    if collapsed >= 3:
        return f"{collapsed} segments have no duration (timestamps out of order or past the clip)"
    covered = segs[-1].end - chunk.start
    if length > 120 and covered < length * 0.85:
        return f"covers only {covered:.0f}s of {length:.0f}s"
    words = sum(len(s.text.split()) for s in segs)
    if length > 120 and words < length * 0.6:  # ~36 words/min floor; lectures run ~110
        return f"only {words} words for {length:.0f}s"
    return None


def _words(segs: list[Segment]) -> int:
    return sum(len(s.text.split()) for s in segs)


# An upgrade is rejected if it is this much shorter than what it replaces: a sudden drop in
# length usually means the model summarised or truncated instead of translating.
MIN_UPGRADE_WORD_RATIO = 0.75


def accept_upgrade(
    new: list[Segment], new_problem: str | None, old: list[Segment], old_problem: str | None
) -> str | None:
    """Return why a re-translated chunk should NOT replace the old one, or None to accept."""
    if new_problem and not old_problem:
        return f"new translation failed checks ({new_problem})"
    old_words = _words(old)
    if old_words and _words(new) < old_words * MIN_UPGRADE_WORD_RATIO:
        return f"new translation is much shorter ({_words(new)} vs {old_words} words)"
    return None


def _translate_chunk(
    gem: Gemini,
    ch: audio.Chunk,
    chain: list[str],
    meta: VideoMeta,
    parts: int,
    before: list[Segment],
    kind: Literal["new", "upgrade"] = "new",
    tries: int = 3,
) -> tuple[list[Segment], str, str | None]:
    """Translate one chunk with the best available model in `chain`, retrying up to `tries`
    times while the result fails the checks. Returns (segments, model used, problem or None);
    the caller records the final attempt's outcome. Raises QuotaExhausted if no model in
    chain is usable."""
    context = ""
    if before:
        tail = " ".join(s.text for s in before[-3:])[-600:]
        context = f'The previous part ended with: "…{tail}" '
    user = prompts.TRANSCRIBE_USER.format(
        part=ch.index + 1,
        parts=parts,
        title=meta.title_ar,
        start=fmt_ts(ch.start),
        end=fmt_ts(ch.end),
        context=context,
    )
    contents = [types.Part.from_bytes(data=ch.path.read_bytes(), mime_type="audio/ogg"), user]
    system = prompts.TRANSCRIBE_SYSTEM.format(glossary=prompts.glossary_text())
    segs: list[Segment] = []
    problem: str | None = "not attempted"
    model = chain[0]
    for attempt in range(1, tries + 1):
        # wait=False: if every allowed model is cooling down, raise Overloaded so the caller
        # (a per-model driver) can hand the video back instead of blocking on it.
        out = gem.generate(chain, contents, _ChunkOut, system=system, wait=False)
        model = gem.last_model or chain[0]
        segs = offset_segments(out, ch)
        problem = check_coverage(segs, ch)
        overrun = timing_overrun(out, ch)
        if not problem and overrun > OVERRUN_TOLERANCE:
            problem = f"timestamps overran the clip by {100 * (overrun - 1):.0f}% (rescaled)"
        if not problem or attempt == tries:
            break
        attempts.record(meta.id, ch.index, ch.start, ch.end, model, kind, "retry", problem)
        log.warning("%s chunk %d (%s): %s; retrying", meta.id, ch.index, model, problem)
    return segs, model, problem


def _allowed(chain: list[str], models: list[str] | None) -> list[str]:
    return chain if models is None else [m for m in chain if m in models]


def upgradable_chunks(
    gem: Gemini, existing: Transcript, models: list[str] | None = None
) -> list[int]:
    """Indexes of chunks that a better-ranked model with quota left (restricted to `models`
    if given) could re-translate."""
    infos = existing.chunk_info or [
        ChunkInfo(
            start=0, end=existing.duration, model=existing.model, translated_at=existing.created_at
        )
    ]
    return [
        i
        for i, c in enumerate(infos)
        if gem.available(_allowed(quality.better_models(c.model), models))
    ]


def transcribe_video(
    gem: Gemini,
    meta: VideoMeta,
    existing: Transcript | None = None,
    keep_audio: bool = False,
    models: list[str] | None = None,
) -> Transcript | None:
    """Create a transcript, or upgrade the chunks of `existing` that were translated by
    lower-ranked models. Returns the written transcript, or None if nothing changed.

    New transcripts raise QuotaExhausted when no model is left (chunk results are cached, so a
    later run resumes). Upgrades keep the old chunk whenever no better model is available or
    the new translation fails the acceptance checks. `models` restricts which models may be
    used (a per-model driver passes just its own). Raises Overloaded if they are all cooling
    down; translated chunks stay cached, so whoever picks the video up next reuses them."""
    if existing is not None and not upgradable_chunks(gem, existing, models):
        return None
    if existing is None and not _allowed(config.MODEL_RANKING, models):
        raise ValueError(f"none of {models} is in CATALOG_MODEL_RANKING")
    src = ytdlp.download_audio(meta.id, config.AUDIO_DIR)
    work = config.CHUNK_DIR / meta.id
    legacy = existing is not None and not existing.chunk_info
    cuts = (
        [existing.chunk_info[0].start] + [c.end for c in existing.chunk_info]
        if existing is not None and not legacy
        else None
    )
    chunks = audio.make_chunks(src, work, config.CHUNK_SECONDS, cuts)
    now = datetime.now(UTC)
    history = attempts.load()

    segments: list[Segment] = []
    infos: list[ChunkInfo] = []
    changed = existing is None
    for ch in chunks:
        if existing is None:
            old_info, old_segs = None, []
        elif legacy:
            old_info = ChunkInfo(
                start=ch.start, end=ch.end, model=existing.model, translated_at=existing.created_at
            )
            old_segs = [x for x in existing.segments if ch.start <= x.start < ch.end]
        else:
            old_info = existing.chunk_info[ch.index]
            old_segs = existing.segments_by_chunk()[ch.index]
        chain = _allowed(quality.better_models(old_info.model if old_info else None), models)

        cache = work / f"{ch.index:03d}.json"
        cached = store.read_json(cache) if cache.exists() else None
        result: tuple[list[Segment], str, str | None] | None = None
        kind: Literal["new", "upgrade"] = "new" if old_info is None else "upgrade"
        fresh = False  # translated in this run (vs. reused from the chunk cache)
        if isinstance(cached, dict) and quality.rank(cached["model"]) > quality.rank(
            old_info.model if old_info else None
        ):
            result = (
                [Segment.model_validate(x) for x in cached["segments"]],
                cached["model"],
                cached.get("problem"),
            )
        elif chain and (existing is None or gem.available(chain)):
            # A chunk that already failed with several models is likely an audio problem
            # (recitation, music, silence...): don't spend extra requests retrying it.
            failed = attempts.failed_models(attempts.for_chunk(history, meta.id, ch.start))
            tries = 1 if len(failed) >= review.FAILING_MODELS_FOR_REVIEW else 3
            try:
                result = _translate_chunk(
                    gem, ch, chain, meta, len(chunks), segments, kind=kind, tries=tries
                )
                fresh = True
            except QuotaExhausted:
                if existing is None:
                    raise
            if result:
                segs, model, problem = result
                store.write_json(
                    cache,
                    {
                        "model": model,
                        "problem": problem,
                        "segments": [x.model_dump() for x in segs],
                    },
                )

        if result and old_info is not None:
            reason = accept_upgrade(result[0], result[2], old_segs, old_info.problem)
            if reason:
                log.warning(
                    "%s chunk %d: keeping %s: %s", meta.id, ch.index, old_info.model, reason
                )
                if fresh:
                    shorter = "shorter" in reason
                    attempts.record(
                        meta.id,
                        ch.index,
                        ch.start,
                        ch.end,
                        result[1],
                        kind,
                        "rejected",
                        reason if shorter else result[2],
                    )
                result = None
        if result and fresh:
            attempts.record(
                meta.id,
                ch.index,
                ch.start,
                ch.end,
                result[1],
                kind,
                "accepted_with_problem" if result[2] else "accepted",
                result[2],
            )

        if result:
            segs, model, problem = result
            if problem:
                log.warning("%s chunk %d: accepted with problem: %s", meta.id, ch.index, problem)
            info = ChunkInfo(
                start=ch.start,
                end=ch.end,
                model=model,
                translated_at=now,
                problem=problem,
                segment_count=len(segs),
            )
            changed = True
            log.info(
                "%s: chunk %d/%d %s by %s (%d segments)",
                meta.id,
                ch.index + 1,
                len(chunks),
                "upgraded" if old_info else "translated",
                model,
                len(segs),
            )
        else:
            assert old_info is not None  # new transcripts always produce a result or raise
            segs = old_segs
            info = old_info.model_copy(
                update={"start": ch.start, "end": ch.end, "segment_count": len(old_segs)}
            )
        segments.extend(segs)
        infos.append(info)

    shutil.rmtree(work, ignore_errors=True)
    if not keep_audio:
        src.unlink(missing_ok=True)
    if not changed:
        return None
    transcript = Transcript(
        id=meta.id,
        model=", ".join(dict.fromkeys(c.model for c in infos)),
        created_at=now,
        duration=chunks[-1].end if chunks else 0,
        chunks=len(chunks),
        chunk_info=infos,
        segments=segments,
    )
    store.write_json(config.TRANSCRIPT_DIR / f"{meta.id}.json", transcript)
    return transcript
