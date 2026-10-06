"""Pydantic models for everything persisted under data/catalog and data/site."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SegmentKind = Literal["speech", "quran", "hadith", "poetry", "dua", "other"]


class PlaylistRef(BaseModel):
    id: str
    position: int | None = None


class VideoStub(BaseModel):
    """A video as listed on the channel (cheap, from a flat listing)."""

    id: str
    title_ar: str
    duration: int | None = None
    view_count: int | None = None
    live: bool = False
    playlists: list[PlaylistRef] = []


class Playlist(BaseModel):
    id: str
    title_ar: str
    description_ar: str = ""
    video_ids: list[str] = []


class Channel(BaseModel):
    url: str
    synced_at: datetime
    videos: list[VideoStub]
    playlists: list[Playlist]


class Chapter(BaseModel):
    start: float
    end: float
    title: str


class VideoMeta(BaseModel):
    """Full per-video metadata from yt-dlp (no translation)."""

    id: str
    title_ar: str
    description_ar: str = ""
    upload_date: str | None = None  # YYYY-MM-DD
    duration: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    tags: list[str] = []
    chapters: list[Chapter] = []
    has_ar_captions: bool = False
    fetched_at: datetime


class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    kind: SegmentKind = "speech"
    text: str
    # Key Arabic terms (as spoken) for terms rendered in English in this segment.
    terms: list[str] = []


class Transcript(BaseModel):
    id: str
    model: str
    created_at: datetime
    duration: float
    chunks: int
    segments: list[Segment]


class QuranRef(BaseModel):
    surah: int
    ayah_start: int | None = None
    ayah_end: int | None = None
    note: str = ""


class References(BaseModel):
    quran: list[QuranRef] = []
    people: list[str] = []
    works: list[str] = []


class Enrichment(BaseModel):
    id: str
    model: str
    created_at: datetime
    source: Literal["transcript", "metadata"]
    title_en: str
    description_en: str = ""
    summary: str = ""
    key_points: list[str] = []
    topics: list[str] = []  # free-form, consolidated later by the taxonomy stage
    series_en: str | None = None  # e.g. "Dialogue on Religion and Secularism"
    episode: int | None = None
    format: str | None = None  # lecture, interview, lesson, q&a, sermon ...
    references: References = References()


class Subtopic(BaseModel):
    slug: str
    name: str
    description: str = ""


class Category(BaseModel):
    slug: str
    name: str
    description: str = ""
    subtopics: list[Subtopic] = []


class Taxonomy(BaseModel):
    model: str
    created_at: datetime
    categories: list[Category]
    # free-form topic (lowercased) -> list of "category/subtopic" keys
    mapping: dict[str, list[str]] = {}


# ---- Site export --------------------------------------------------------------------------


class SiteVideoSummary(BaseModel):
    id: str
    title_en: str
    title_ar: str
    upload_date: str | None
    duration: int | None
    view_count: int | None
    topics: list[str]
    playlists: list[str]
    series: str | None
    episode: int | None
    format: str | None
    has_transcript: bool
    summary: str


class SiteVideo(SiteVideoSummary):
    description_ar: str
    description_en: str
    key_points: list[str]
    references: References
    chapters: list[Chapter]
    segments: list[Segment] = Field(default_factory=list)
    transcript_model: str | None = None


class SitePlaylist(BaseModel):
    id: str
    slug: str
    title_en: str
    title_ar: str
    description_en: str = ""
    video_ids: list[str]
