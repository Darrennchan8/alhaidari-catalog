"""Stage: audio → timestamped English transcript, via Gemini, chunk by chunk."""

from __future__ import annotations

import itertools
import logging
import re
import shutil
from datetime import UTC, datetime
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field

from . import audio, config, prompts, store, ytdlp
from .gemini import Gemini
from .models import Segment, Transcript, VideoMeta

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


def offset_segments(out: _ChunkOut, chunk: audio.Chunk) -> list[Segment]:
    """Convert clip-relative timestamps to absolute ones, clamped to the chunk and monotonic."""
    length = chunk.end - chunk.start
    segs: list[Segment] = []
    prev_end = 0.0
    for o in out.segments:
        text = o.text.strip()
        if not text:
            continue
        try:
            a, b = parse_ts(o.start), parse_ts(o.end)
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
    covered = segs[-1].end - chunk.start
    if length > 120 and covered < length * 0.85:
        return f"covers only {covered:.0f}s of {length:.0f}s"
    words = sum(len(s.text.split()) for s in segs)
    if length > 120 and words < length * 0.6:  # ~36 words/min floor; lectures run ~110
        return f"only {words} words for {length:.0f}s"
    return None


def transcribe_video(gem: Gemini, meta: VideoMeta, keep_audio: bool = False) -> Transcript:
    src = ytdlp.download_audio(meta.id, config.AUDIO_DIR)
    work = config.CHUNK_DIR / meta.id
    chunks = audio.make_chunks(src, work, config.CHUNK_SECONDS)
    system = prompts.TRANSCRIBE_SYSTEM.format(glossary=prompts.glossary_text())
    segments: list[Segment] = []
    models: list[str] = []
    for ch in chunks:
        cache = work / f"{ch.index:03d}.json"
        if cache.exists():
            cached = store.read_json(cache)
            if isinstance(cached, list):  # older cache format: bare segment list
                cached = {"model": config.TRANSCRIBE_MODEL, "segments": cached}
            segs = [Segment.model_validate(s) for s in cached["segments"]]
            models.append(cached["model"])
        else:
            context = ""
            if segments:
                tail = " ".join(s.text for s in segments[-3:])[-600:]
                context = f'The previous part ended with: "…{tail}" '
            user = prompts.TRANSCRIBE_USER.format(
                part=ch.index + 1,
                parts=len(chunks),
                title=meta.title_ar,
                start=fmt_ts(ch.start),
                end=fmt_ts(ch.end),
                context=context,
            )
            contents = [
                types.Part.from_bytes(data=ch.path.read_bytes(), mime_type="audio/ogg"),
                user,
            ]
            segs, problem = [], "not attempted"
            for _ in range(3):
                out = gem.generate(config.TRANSCRIBE_MODEL, contents, _ChunkOut, system=system)
                segs = offset_segments(out, ch)
                problem = check_coverage(segs, ch)
                if not problem:
                    break
                log.warning("%s chunk %d: %s; retrying", meta.id, ch.index, problem)
            if problem:
                log.warning("%s chunk %d: accepting with problem: %s", meta.id, ch.index, problem)
            used = gem.last_model or config.TRANSCRIBE_MODEL
            models.append(used)
            store.write_json(cache, {"model": used, "segments": [s.model_dump() for s in segs]})
        segments.extend(segs)
        log.info(
            "%s: chunk %d/%d done (%d segments)", meta.id, ch.index + 1, len(chunks), len(segs)
        )

    transcript = Transcript(
        id=meta.id,
        model=", ".join(dict.fromkeys(models)) or config.TRANSCRIBE_MODEL,
        created_at=datetime.now(UTC),
        duration=chunks[-1].end if chunks else 0,
        chunks=len(chunks),
        segments=segments,
    )
    store.write_json(config.TRANSCRIPT_DIR / f"{meta.id}.json", transcript)
    shutil.rmtree(work, ignore_errors=True)
    if not keep_audio:
        src.unlink(missing_ok=True)
    return transcript
