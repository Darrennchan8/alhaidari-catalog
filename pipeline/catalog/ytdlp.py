"""Thin wrapper around the yt-dlp CLI (subprocess keeps yt-dlp's own config/plugins working)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import config
from .models import Channel, Chapter, Playlist, PlaylistRef, VideoMeta, VideoStub

log = logging.getLogger(__name__)


class YtDlpError(RuntimeError):
    pass


def _base_args() -> list[str]:
    args = [sys.executable, "-m", "yt_dlp", "--js-runtimes", "node", "--no-warnings"]
    if config.YTDLP_COOKIES:
        args += ["--cookies", config.YTDLP_COOKIES]
    return args


def _run(args: list[str], retries: int = 3, timeout: int = 1800) -> str:
    for attempt in range(1, retries + 1):
        proc = subprocess.run(_base_args() + args, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout
        err = proc.stderr.strip().splitlines()[-1:] or ["unknown error"]
        # Permanently unavailable videos should not be retried.
        if any(s in err[0] for s in ("Private video", "Video unavailable", "removed")):
            raise YtDlpError(err[0])
        log.warning("yt-dlp failed (attempt %d/%d): %s", attempt, retries, err[0])
        time.sleep(10 * attempt)
    raise YtDlpError(err[0])


def _flat(url: str) -> dict:
    return json.loads(_run(["--flat-playlist", "-J", url]))


def fetch_channel() -> Channel:
    """List every upload, live stream and playlist (with membership) on the channel."""
    stubs: dict[str, VideoStub] = {}
    for tab in ("videos", "streams"):
        try:
            listing = _flat(f"{config.CHANNEL_URL}/{tab}")
        except YtDlpError as e:  # channel may have no streams tab
            log.info("skipping %s tab: %s", tab, e)
            continue
        for e in listing.get("entries") or []:
            if not e or not e.get("id") or e["id"] in stubs:
                continue
            stubs[e["id"]] = VideoStub(
                id=e["id"],
                title_ar=e.get("title") or "",
                duration=int(e["duration"]) if e.get("duration") else None,
                view_count=e.get("view_count"),
                live=tab == "streams",
            )
        log.info("%s tab: %d videos total so far", tab, len(stubs))

    playlists: list[Playlist] = []
    for p in _flat(f"{config.CHANNEL_URL}/playlists").get("entries") or []:
        if not p or not p.get("id"):
            continue
        time.sleep(config.YTDLP_SLEEP)
        detail = _flat(f"https://www.youtube.com/playlist?list={p['id']}")
        ids = [e["id"] for e in detail.get("entries") or [] if e and e.get("id")]
        playlists.append(
            Playlist(
                id=p["id"],
                title_ar=detail.get("title") or p.get("title") or "",
                description_ar=detail.get("description") or "",
                video_ids=ids,
            )
        )
        for pos, vid in enumerate(ids, 1):
            if vid in stubs:
                stubs[vid].playlists.append(PlaylistRef(id=p["id"], position=pos))
        log.info("playlist %s: %d videos", p["id"], len(ids))

    return Channel(
        url=config.CHANNEL_URL,
        synced_at=datetime.now(UTC),
        videos=list(stubs.values()),
        playlists=playlists,
    )


def fetch_meta(video_id: str) -> VideoMeta:
    d = json.loads(_run(["-J", "--skip-download", f"https://www.youtube.com/watch?v={video_id}"]))
    ud = d.get("upload_date")
    return VideoMeta(
        id=video_id,
        title_ar=d.get("title") or "",
        description_ar=d.get("description") or "",
        upload_date=f"{ud[:4]}-{ud[4:6]}-{ud[6:]}" if ud else None,
        duration=int(d["duration"]) if d.get("duration") else None,
        view_count=d.get("view_count"),
        like_count=d.get("like_count"),
        tags=d.get("tags") or [],
        chapters=[
            Chapter(start=c["start_time"], end=c["end_time"], title=c.get("title") or "")
            for c in d.get("chapters") or []
        ],
        has_ar_captions=any(
            k.startswith("ar")
            for k in (d.get("automatic_captions") or {}) | (d.get("subtitles") or {})
        ),
        fetched_at=datetime.now(UTC),
    )


def download_audio(video_id: str, dest_dir: Path) -> Path:
    """Download the smallest reasonable audio-only stream; returns the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest_dir.glob(f"{video_id}.*") if not p.name.endswith(".part")]
    if existing:
        return existing[0]
    _run(
        [
            "-f",
            "bestaudio[abr<=80]/bestaudio/best",
            "--no-progress",
            "--no-playlist",
            "-o",
            str(dest_dir / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
    )
    found = [p for p in dest_dir.glob(f"{video_id}.*") if not p.name.endswith(".part")]
    if not found:
        raise YtDlpError(f"audio for {video_id} not found after download")
    return found[0]
