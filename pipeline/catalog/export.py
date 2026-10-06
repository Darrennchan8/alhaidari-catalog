"""Stage: merge channel, metadata, enrichments, transcripts and topics into the JSON the
static site reads (data/site/). Offline and deterministic."""

from __future__ import annotations

import re
import shutil
import unicodedata
from collections import Counter
from datetime import UTC, datetime

from . import config, store
from .models import (
    Channel,
    Enrichment,
    References,
    SitePlaylist,
    SiteVideo,
    SiteVideoSummary,
    Taxonomy,
    Transcript,
    VideoMeta,
)
from .taxonomy import load_assignments, load_overrides

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EPISODE = re.compile(r"(\d+)\D*$")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80] or "untitled"


def episode_from_title(title_ar: str) -> int | None:
    m = _EPISODE.search(title_ar.translate(_ARABIC_DIGITS))
    return int(m.group(1)) if m and len(m.group(1)) <= 4 else None


def export_site() -> dict[str, int]:
    ch = store.read_model(config.CHANNEL_FILE, Channel)
    pl_en: dict = (
        store.read_json(config.PLAYLISTS_EN_FILE) if config.PLAYLISTS_EN_FILE.exists() else {}
    )
    tax = store.maybe_model(config.TAXONOMY_FILE, Taxonomy)
    assigned = {k: a.topics for k, a in load_assignments().items()}
    assigned.update(load_overrides())
    valid_topics = (
        {f"{c.slug}/{s.slug}" for c in tax.categories for s in c.subtopics} if tax else set()
    )

    out = config.SITE_DATA
    if out.exists():
        shutil.rmtree(out)
    (out / "videos").mkdir(parents=True)

    # Playlists, with stable unique slugs.
    playlists: list[SitePlaylist] = []
    used: set[str] = set()
    for p in ch.playlists:
        en = pl_en.get(p.id, {})
        title_en = en.get("title_en") or ""
        slug = slugify(title_en) if title_en else p.id.lower()
        if slug in used:
            slug = f"{slug}-{p.id[-6:].lower()}"
        used.add(slug)
        playlists.append(
            SitePlaylist(
                id=p.id,
                slug=slug,
                title_en=title_en,
                title_ar=p.title_ar,
                description_en=en.get("description_en", ""),
                video_ids=[v for v in p.video_ids if any(s.id == v for s in ch.videos)],
            )
        )
    pl_slug = {p.id: p.slug for p in playlists}

    summaries: list[SiteVideoSummary] = []
    for stub in ch.videos:
        meta = store.maybe_model(config.META_DIR / f"{stub.id}.json", VideoMeta)
        enr = store.maybe_model(config.ENRICH_DIR / f"{stub.id}.json", Enrichment)
        tr = store.maybe_model(config.TRANSCRIPT_DIR / f"{stub.id}.json", Transcript)
        topics = [t for t in assigned.get(stub.id, []) if t in valid_topics]
        title_ar = meta.title_ar if meta else stub.title_ar
        video = SiteVideo(
            id=stub.id,
            title_en=enr.title_en if enr else "",
            title_ar=title_ar,
            upload_date=meta.upload_date if meta else None,
            duration=(meta.duration if meta else None) or stub.duration,
            view_count=(meta.view_count if meta else None) or stub.view_count,
            topics=topics,
            playlists=[pl_slug[r.id] for r in stub.playlists if r.id in pl_slug],
            series=enr.series_en if enr else None,
            episode=(enr.episode if enr and enr.episode else None) or episode_from_title(title_ar),
            format=enr.format if enr else None,
            has_transcript=bool(tr and tr.segments),
            summary=enr.summary if enr else "",
            description_ar=meta.description_ar if meta else "",
            description_en=enr.description_en if enr else "",
            key_points=enr.key_points if enr else [],
            references=enr.references if enr else References(),
            chapters=meta.chapters if meta else [],
            segments=tr.segments if tr else [],
            transcript_model=tr.model if tr else None,
        )
        store.write_json(out / "videos" / f"{stub.id}.json", video)
        summaries.append(SiteVideoSummary.model_validate(video.model_dump()))

    summaries.sort(key=lambda v: (v.upload_date or "", v.id), reverse=True)
    store.write_json(out / "index.json", [s.model_dump() for s in summaries])
    store.write_json(out / "playlists.json", [p.model_dump() for p in playlists])

    counts = Counter(t for s in summaries for t in s.topics)
    topics_out = []
    if tax:
        for c in tax.categories:
            subs = [
                {
                    **s.model_dump(),
                    "key": f"{c.slug}/{s.slug}",
                    "count": counts[f"{c.slug}/{s.slug}"],
                }
                for s in c.subtopics
            ]
            topics_out.append(
                {
                    "slug": c.slug,
                    "name": c.name,
                    "description": c.description,
                    "count": len(
                        {
                            s.id
                            for s in summaries
                            if any(t.startswith(c.slug + "/") for t in s.topics)
                        }
                    ),
                    "subtopics": subs,
                }
            )
    store.write_json(out / "topics.json", topics_out)

    stats = {
        "videos": len(summaries),
        "transcribed": sum(s.has_transcript for s in summaries),
        "translated": sum(bool(s.title_en) for s in summaries),
        "with_topics": sum(bool(s.topics) for s in summaries),
        "playlists": len(playlists),
        "hours": round(sum(s.duration or 0 for s in summaries) / 3600),
        "transcribed_hours": round(
            sum(s.duration or 0 for s in summaries if s.has_transcript) / 3600
        ),
    }
    store.write_json(
        out / "stats.json",
        {
            **stats,
            "channel_url": ch.url,
            "synced_at": ch.synced_at,
            "exported_at": datetime.now(UTC),
        },
    )
    return stats
