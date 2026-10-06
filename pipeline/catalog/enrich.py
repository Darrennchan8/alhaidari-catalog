"""Stage: English title/description/summary/topics/references for each video, and English
playlist titles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from . import config, prompts, quality, store
from .gemini import Gemini
from .models import Channel, Enrichment, QuranRef, References, Transcript, VideoMeta
from .transcribe import fmt_ts

log = logging.getLogger(__name__)

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


def _to_enrichment(out: _EnrichOut, video_id: str, model: str, source: str) -> Enrichment:
    return Enrichment(
        id=video_id,
        model=model,
        created_at=datetime.now(UTC),
        source=source,
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


def problems(e: Enrichment) -> str | None:
    """Basic sanity checks; a failing result never replaces an existing one."""
    if not e.title_en or any("\u0600" <= c <= "\u06ff" for c in e.title_en):
        return "missing or untranslated title"
    if not e.summary:
        return "missing summary"
    if e.source == "transcript" and not e.key_points:
        return "missing key points"
    return None


def _save(e: Enrichment, current: Enrichment | None) -> Enrichment | None:
    problem = problems(e)
    if problem and current is not None:
        log.warning(
            "%s: keeping %s enrichment; new one from %s has %s",
            e.id,
            current.model,
            e.model,
            problem,
        )
        return None
    store.write_json(config.ENRICH_DIR / f"{e.id}.json", e)
    return e


@dataclass
class EnrichJob:
    meta: VideoMeta
    playlists: list[str]
    transcript: Transcript | None
    current: Enrichment | None
    models: list[str]  # models allowed to (re)do this job, best first
    priority: int  # lower runs first
    reason: str


def plan_job(
    meta: VideoMeta, playlists: list[str], t: Transcript | None, e: Enrichment | None
) -> EnrichJob | None:
    """Decide whether a video's enrichment needs (re)doing, with which models, and how urgently:
    missing first, then stale (the transcript is newer), then upgrades from the lowest rank up."""
    has_t = t is not None and bool(t.segments)
    if e is None:
        return EnrichJob(meta, playlists, t, e, list(config.MODEL_RANKING), 0, "missing")
    if has_t and (e.source != "transcript" or e.created_at < t.created_at):
        return EnrichJob(meta, playlists, t, e, list(config.MODEL_RANKING), 1, "stale")
    better = quality.better_models(e.model)
    if better:
        return EnrichJob(meta, playlists, t, e, better, 2 + quality.rank(e.model), "upgrade")
    return None


def enrich_video(gem: Gemini, job: EnrichJob) -> Enrichment | None:
    meta, transcript = job.meta, job.transcript
    has_t = transcript is not None and bool(transcript.segments)
    user = prompts.ENRICH_USER.format(
        title_ar=meta.title_ar,
        description_ar=meta.description_ar.strip() or "(none)",
        playlists=", ".join(job.playlists) or "(none)",
        upload_date=meta.upload_date or "unknown",
        duration=fmt_ts(meta.duration or 0),
        body=transcript_body(transcript) if has_t else "(No transcript available yet.)",
        summary_rule=prompts.ENRICH_SUMMARY_TRANSCRIPT if has_t else prompts.ENRICH_SUMMARY_META,
        key_points_rule=prompts.ENRICH_KP_TRANSCRIPT if has_t else prompts.ENRICH_KP_META,
    )
    out = gem.generate(
        job.models,
        [user],
        _EnrichOut,
        system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
    )
    e = _to_enrichment(
        out, meta.id, gem.last_model or job.models[0], "transcript" if has_t else "metadata"
    )
    return _save(e, job.current)


class _PlaylistItem(BaseModel):
    id: str
    title_en: str
    description_en: str


class _PlaylistsOut(BaseModel):
    items: list[_PlaylistItem]


def translate_playlists(gem: Gemini, channel: Channel) -> dict[str, dict[str, str]]:
    """English playlist titles; missing ones are added and lower-ranked ones upgraded."""
    existing: dict = (
        store.read_json(config.PLAYLISTS_EN_FILE) if config.PLAYLISTS_EN_FILE.exists() else {}
    )
    todo = [
        p
        for p in channel.playlists
        if p.id not in existing or quality.better_models(existing[p.id].get("model", "unknown"))
    ]
    worst = min(
        (existing[p.id].get("model", "unknown") if p.id in existing else None for p in todo),
        key=quality.rank,
        default="",
    )
    models = gem.available(quality.better_models(worst)) if todo else []
    if not models:
        return existing
    items = "\n".join(
        f"- id={p.id}\n  title: {p.title_ar}\n  description: {p.description_ar.strip() or '(none)'}"
        for p in todo
    )
    out = gem.generate(
        models,
        [prompts.PLAYLISTS_PROMPT.format(items=items)],
        _PlaylistsOut,
        system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
    )
    model = gem.last_model or models[0]
    wanted = {p.id for p in todo}
    for it in out.items:
        old = existing.get(it.id)
        if (
            it.id in wanted
            and it.title_en.strip()
            and (old is None or quality.rank(model) > quality.rank(old.get("model", "unknown")))
        ):
            existing[it.id] = {
                "title_en": it.title_en.strip(),
                "description_en": it.description_en.strip(),
                "model": model,
            }
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


def enrich_metadata_batch(gem: Gemini, jobs: list[EnrichJob]) -> list[Enrichment]:
    """Catalog several untranscribed videos from their metadata in one request. All jobs must
    share the same allowed `models`."""
    items = "\n".join(
        f"- id={j.meta.id}\n  title: {j.meta.title_ar}\n  playlists: {', '.join(j.playlists) or '(none)'}\n"
        f"  date: {j.meta.upload_date or '?'}; duration: {fmt_ts(j.meta.duration or 0)}\n"
        f"  description: {(j.meta.description_ar.strip() or '(none)')[:1500]}"
        for j in jobs
    )
    out = gem.generate(
        jobs[0].models,
        [BATCH_PROMPT.format(items=items)],
        _BatchOut,
        system=prompts.ENRICH_SYSTEM.format(glossary=prompts.glossary_text()),
    )
    model = gem.last_model or jobs[0].models[0]
    by_id = {j.meta.id: j for j in jobs}
    results: list[Enrichment] = []
    for it in out.items:
        job = by_id.pop(it.id, None)
        if job is None:
            continue
        e = _to_enrichment(it, it.id, model, "metadata")
        e.key_points = []
        if _save(e, job.current):
            results.append(e)
    return results
