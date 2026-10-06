"""Stage: English title/description/summary/topics/references for each video, and English
playlist titles."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from . import config, prompts, store
from .gemini import Gemini
from .models import Channel, Enrichment, QuranRef, References, Transcript, VideoMeta
from .transcribe import fmt_ts

Format = Literal[
    "lecture",
    "lesson",
    "interview",
    "dialogue",
    "q&a",
    "sermon",
    "speech",
    "documentary",
    "clip",
    "other",
]


class _Quran(BaseModel):
    surah: int
    ayah_start: int | None
    ayah_end: int | None
    note: str


class _Refs(BaseModel):
    quran: list[_Quran]
    people: list[str]
    works: list[str]


class _EnrichOut(BaseModel):
    title_en: str
    description_en: str
    summary: str
    key_points: list[str]
    topics: list[str]
    series_en: str | None
    episode: int | None
    format: Format
    references: _Refs


# A transcript this long (characters) is sampled rather than sent whole; Flash models handle
# ~100k tokens comfortably, so this only triggers for multi-hour recordings.
MAX_TRANSCRIPT_CHARS = 300_000


def transcript_body(t: Transcript) -> str:
    lines = [f"[{fmt_ts(s.start)}] {s.speaker or ''}: {s.text}" for s in t.segments]
    text = "\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        half = MAX_TRANSCRIPT_CHARS // 2
        text = text[:half] + "\n[… middle omitted …]\n" + text[-half:]
    return "English transcript (machine translated from the Arabic audio):\n" + text


def enrich_video(
    gem: Gemini, meta: VideoMeta, playlists: list[str], transcript: Transcript | None
) -> Enrichment:
    has_t = transcript is not None and bool(transcript.segments)
    user = prompts.ENRICH_USER.format(
        title_ar=meta.title_ar,
        description_ar=meta.description_ar.strip() or "(none)",
        playlists=", ".join(playlists) or "(none)",
        upload_date=meta.upload_date or "unknown",
        duration=fmt_ts(meta.duration or 0),
        body=transcript_body(transcript) if has_t else "(No transcript available yet.)",
        summary_rule=prompts.ENRICH_SUMMARY_TRANSCRIPT if has_t else prompts.ENRICH_SUMMARY_META,
        key_points_rule=prompts.ENRICH_KP_TRANSCRIPT if has_t else prompts.ENRICH_KP_META,
    )
    out = gem.generate(
        config.ENRICH_MODEL,
        [user],
        _EnrichOut,
        system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
    )
    e = Enrichment(
        id=meta.id,
        model=gem.last_model or config.ENRICH_MODEL,
        created_at=datetime.now(UTC),
        source="transcript" if has_t else "metadata",
        title_en=out.title_en.strip(),
        description_en=out.description_en.strip(),
        summary=out.summary.strip(),
        key_points=[k.strip() for k in out.key_points if k.strip()],
        topics=sorted({t.strip() for t in out.topics if t.strip()}),
        series_en=(out.series_en or "").strip() or None,
        episode=out.episode,
        format=out.format,
        references=References(
            quran=[QuranRef(**q.model_dump()) for q in out.references.quran if 1 <= q.surah <= 114],
            people=out.references.people,
            works=out.references.works,
        ),
    )
    store.write_json(config.ENRICH_DIR / f"{meta.id}.json", e)
    return e


class _PlaylistItem(BaseModel):
    id: str
    title_en: str
    description_en: str


class _PlaylistsOut(BaseModel):
    items: list[_PlaylistItem]


def translate_playlists(gem: Gemini, channel: Channel) -> dict[str, dict[str, str]]:
    existing: dict = (
        store.read_json(config.PLAYLISTS_EN_FILE) if config.PLAYLISTS_EN_FILE.exists() else {}
    )
    todo = [p for p in channel.playlists if p.id not in existing]
    if todo:
        items = "\n".join(
            f"- id={p.id}\n  title: {p.title_ar}\n  description: {p.description_ar.strip() or '(none)'}"
            for p in todo
        )
        out = gem.generate(
            config.ENRICH_MODEL,
            [prompts.PLAYLISTS_PROMPT.format(items=items)],
            _PlaylistsOut,
            system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
            fallbacks=config.ENRICH_FALLBACK_MODELS,
        )
        wanted = {p.id for p in todo}
        for it in out.items:
            if it.id in wanted:
                existing[it.id] = {"title_en": it.title_en, "description_en": it.description_en}
        store.write_json(config.PLAYLISTS_EN_FILE, existing)
    return existing


class _BatchItem(_EnrichOut):
    id: str


class _BatchOut(BaseModel):
    items: list[_BatchItem]


BATCH_PROMPT = """\
Catalog each of these videos from its title and description only (no transcript is \
available yet). Return one item per video id, in the same order.

{items}

For each produce:
- title_en: faithful, natural English rendering of the Arabic title (keep episode numbers \
as "Episode N" / "Lesson N"; drop a trailing honorific byline).
- description_en: English translation of the description (empty if none; keep URLs).
- summary: one or two sentences on what the video is about, inferred ONLY from the title, \
description and series context. Do not invent arguments or conclusions.
- key_points: empty list.
- topics: 2–5 specific subject topics in English.
- series_en / episode / format: as evident from the title and playlist, else null / "other".
- references: only what the title/description explicitly names (else empty lists).
"""


def enrich_metadata_batch(
    gem: Gemini, batch: list[tuple[VideoMeta, list[str]]]
) -> list[Enrichment]:
    items = "\n".join(
        f"- id={m.id}\n  title: {m.title_ar}\n  playlists: {', '.join(pls) or '(none)'}\n"
        f"  date: {m.upload_date or '?'}; duration: {fmt_ts(m.duration or 0)}\n"
        f"  description: {(m.description_ar.strip() or '(none)')[:1500]}"
        for m, pls in batch
    )
    out = gem.generate(
        config.ENRICH_MODEL,
        [BATCH_PROMPT.format(items=items)],
        _BatchOut,
        system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
        fallbacks=config.ENRICH_FALLBACK_MODELS,
    )
    wanted = {m.id for m, _ in batch}
    results: list[Enrichment] = []
    for it in out.items:
        if it.id not in wanted:
            continue
        e = Enrichment(
            id=it.id,
            model=gem.last_model or config.ENRICH_MODEL,
            created_at=datetime.now(UTC),
            source="metadata",
            title_en=it.title_en.strip(),
            description_en=it.description_en.strip(),
            summary=it.summary.strip(),
            key_points=[],
            topics=sorted({t.strip() for t in it.topics if t.strip()}),
            series_en=(it.series_en or "").strip() or None,
            episode=it.episode,
            format=it.format,
            references=References(
                quran=[
                    QuranRef(**q.model_dump()) for q in it.references.quran if 1 <= q.surah <= 114
                ],
                people=it.references.people,
                works=it.references.works,
            ),
        )
        store.write_json(config.ENRICH_DIR / f"{it.id}.json", e)
        results.append(e)
    return results
