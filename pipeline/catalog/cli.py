"""`catalog` command line: run pipeline stages, each idempotent and resumable."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import config, store, ytdlp
from .models import Channel, Enrichment, Transcript, VideoMeta

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()
log = logging.getLogger("catalog")


class Order(StrEnum):
    playlist = "playlist"  # playlist videos first, in playlist order
    views = "views"
    newest = "newest"
    oldest = "oldest"
    shortest = "shortest"


@app.callback()
def _setup(verbose: bool = typer.Option(False, "-v", "--verbose")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _channel() -> Channel:
    if not config.CHANNEL_FILE.exists():
        raise typer.BadParameter("no channel.json yet; run `catalog sync` first")
    return store.read_model(config.CHANNEL_FILE, Channel)


def _meta(video_id: str) -> VideoMeta | None:
    return store.maybe_model(config.META_DIR / f"{video_id}.json", VideoMeta)


def _ordered_ids(ch: Channel, order: Order) -> list[str]:
    vids = ch.videos
    if order is Order.playlist:
        seen: dict[str, None] = {}
        for p in ch.playlists:
            for vid in p.video_ids:
                seen.setdefault(vid, None)
        known = {v.id for v in vids}
        rest = sorted((v for v in vids if v.id not in seen), key=lambda v: -(v.view_count or 0))
        return [i for i in seen if i in known] + [v.id for v in rest]
    if order is Order.views:
        return [v.id for v in sorted(vids, key=lambda v: -(v.view_count or 0))]
    if order is Order.shortest:
        return [v.id for v in sorted(vids, key=lambda v: v.duration or 0)]
    # Channel listing is newest-first.
    ids = [v.id for v in vids]
    return ids if order is Order.newest else ids[::-1]


@app.command()
def sync() -> None:
    """List all videos and playlists on the channel → data/catalog/channel.json."""
    ch = ytdlp.fetch_channel()
    store.write_json(config.CHANNEL_FILE, ch)
    console.print(f"[green]{len(ch.videos)} videos, {len(ch.playlists)} playlists")


@app.command()
def meta(
    workers: int = typer.Option(2, help="parallel yt-dlp processes"),
    refresh: bool = typer.Option(False, help="refetch existing metadata"),
    limit: int = typer.Option(0, help="stop after N videos (0 = all)"),
) -> None:
    """Fetch full per-video metadata (description, date, chapters) → data/catalog/meta/."""
    import time

    ids = [v.id for v in _channel().videos]
    todo = [i for i in ids if refresh or not (config.META_DIR / f"{i}.json").exists()]
    if limit:
        todo = todo[:limit]
    console.print(f"{len(todo)} videos to fetch")

    def one(vid: str) -> str:
        m = ytdlp.fetch_meta(vid)
        store.write_json(config.META_DIR / f"{vid}.json", m)
        time.sleep(config.YTDLP_SLEEP)
        return vid

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, v): v for v in todo}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
                if done % 50 == 0:
                    log.info("meta %d/%d", done, len(todo))
            except Exception as e:
                log.error("meta %s failed: %s", futs[f], e)
    console.print(f"[green]fetched {done}/{len(todo)}")


@app.command()
def transcribe(
    ids: list[str] = typer.Argument(None, help="specific video ids (default: by --order)"),
    order: Order = typer.Option(Order.playlist),
    limit: int = typer.Option(0, help="max videos to transcribe this run (0 = no limit)"),
    workers: int = typer.Option(2, help="videos processed in parallel"),
    max_hours: float = typer.Option(0, help="skip videos longer than this (0 = no limit)"),
    keep_audio: bool = typer.Option(False),
    enrich_after: bool = typer.Option(True, help="re-run enrichment from the transcript"),
) -> None:
    """Download audio and translate it into a timestamped English transcript."""
    from .enrich import enrich_video
    from .gemini import Gemini, QuotaExhausted
    from .transcribe import transcribe_video

    ch = _channel()
    pl_titles = {p.id: p.title_ar for p in ch.playlists}
    stubs = {v.id: v for v in ch.videos}
    queue = ids or _ordered_ids(ch, order)
    queue = [i for i in queue if not (config.TRANSCRIPT_DIR / f"{i}.json").exists()]
    if max_hours:
        queue = [i for i in queue if (stubs[i].duration or 0) <= max_hours * 3600]
    if limit:
        queue = queue[:limit]
    console.print(f"{len(queue)} videos queued")
    gem = Gemini()
    stop = False

    def one(vid: str) -> str:
        if stop:
            return f"{vid}: skipped"
        m = _meta(vid)
        if m is None:
            m = ytdlp.fetch_meta(vid)
            store.write_json(config.META_DIR / f"{vid}.json", m)
        t = transcribe_video(gem, m, keep_audio=keep_audio)
        if enrich_after:
            enrich_video(gem, m, [pl_titles[p.id] for p in stubs[vid].playlists], t)
        return f"{vid}: {len(t.segments)} segments"

    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, v): v for v in queue}
        for f in as_completed(futs):
            try:
                log.info("done %s", f.result())
            except QuotaExhausted as e:
                stop = True
                log.error("daily quota exhausted; stopping (re-run later to resume): %s", e)
            except Exception:
                log.exception("transcribe %s failed", futs[f])
    console.print(f"usage: {gem.usage}")


@app.command()
def enrich(
    transcribed_only: bool = typer.Option(False, help="only (re)enrich transcribed videos"),
    refresh: bool = typer.Option(False, help="redo videos that already have an enrichment"),
    workers: int = typer.Option(2),
    limit: int = typer.Option(0),
    batch_size: int = typer.Option(30, help="metadata-only videos per request"),
) -> None:
    """English titles, summaries, topics and references for each video. Videos without a
    transcript are catalogued from title/description only, and upgraded once transcribed."""
    from .enrich import enrich_metadata_batch, enrich_video, translate_playlists
    from .gemini import Gemini, QuotaExhausted

    ch = _channel()
    gem = Gemini()
    translate_playlists(gem, ch)
    pl_titles = {p.id: p.title_ar for p in ch.playlists}
    todo: list[tuple[VideoMeta, list[str], Transcript | None]] = []
    for v in ch.videos:
        m = _meta(v.id)
        if m is None:
            continue
        t = store.maybe_model(config.TRANSCRIPT_DIR / f"{v.id}.json", Transcript)
        e = store.maybe_model(config.ENRICH_DIR / f"{v.id}.json", Enrichment)
        if transcribed_only and t is None:
            continue
        stale = e is None or refresh or (t is not None and e.source != "transcript")
        if stale:
            todo.append((m, [pl_titles[p.id] for p in v.playlists], t))
    if limit:
        todo = todo[:limit]
    # Videos without transcripts are catalogued in batches from their metadata (cheap);
    # transcribed videos get a dedicated call with the full transcript.
    with_t = [a for a in todo if a[2] is not None]
    without = [(m, pls) for m, pls, t in todo if t is None]
    jobs: list = [(enrich_video, a) for a in with_t] + [
        (enrich_metadata_batch, (without[i : i + batch_size],))
        for i in range(0, len(without), batch_size)
    ]
    console.print(f"{len(with_t)} transcribed + {len(without)} metadata-only videos to enrich")
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fn, gem, *args): n for n, (fn, args) in enumerate(jobs)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
                if n % 10 == 0:
                    log.info("enrich jobs %d/%d", n, len(jobs))
            except QuotaExhausted as e:
                log.error("daily quota exhausted: %s", e)
                pool.shutdown(cancel_futures=True)
                break
            except Exception as e:
                log.error("enrich job %s failed: %s", futs[f], e)
    console.print(f"usage: {gem.usage}")


@app.command()
def taxonomy(
    rebuild: bool = typer.Option(False, help="redesign the taxonomy and reassign everything"),
) -> None:
    """Consolidate topics into a browsable taxonomy and assign videos to it."""
    from .gemini import Gemini
    from .models import Taxonomy
    from .taxonomy import assign_topics, build_taxonomy, load_enrichments

    gem = Gemini()
    enrichments = load_enrichments()
    tax = store.maybe_model(config.TAXONOMY_FILE, Taxonomy)
    if tax is None or rebuild:
        tax = build_taxonomy(gem, enrichments)
        console.print(f"[green]taxonomy: {len(tax.categories)} categories")
    assigned = assign_topics(gem, tax, enrichments, rebuild=rebuild)
    console.print(f"[green]{len(assigned)} videos have topics; usage {gem.usage}")


@app.command()
def export() -> None:
    """Merge everything into site-ready JSON → data/site/ (offline, no API calls)."""
    from .export import export_site

    stats = export_site()
    console.print(f"[green]exported {stats}")


@app.command()
def run(
    resync: bool = typer.Option(True, help="re-list the channel first to pick up new uploads"),
    transcribe_limit: int = typer.Option(0, help="max videos to transcribe (0 = until quota)"),
    order: Order = typer.Option(Order.playlist),
    workers: int = typer.Option(2),
) -> None:
    """Run every stage in order (suitable for a daily cron job); stops early on quota."""
    from .gemini import QuotaExhausted

    if resync:
        sync()
    meta(workers=workers, refresh=False, limit=0)
    try:
        enrich(transcribed_only=False, refresh=False, workers=workers, limit=0, batch_size=30)
        transcribe(
            ids=None,
            order=order,
            limit=transcribe_limit,
            workers=workers,
            max_hours=0,
            keep_audio=False,
            enrich_after=True,
        )
        if config.TAXONOMY_FILE.exists() or len(list(config.ENRICH_DIR.glob("*.json"))) > 100:
            taxonomy(rebuild=False)
    except QuotaExhausted as e:
        log.warning("stopping early: %s", e)
    export()
    status()


@app.command()
def status() -> None:
    """Show pipeline coverage."""
    ch = _channel()
    n = len(ch.videos)
    hours = sum(v.duration or 0 for v in ch.videos) / 3600

    def count(d) -> int:
        return len(list(d.glob("*.json"))) if d.exists() else 0

    t_ids = (
        {p.stem for p in config.TRANSCRIPT_DIR.glob("*.json")}
        if config.TRANSCRIPT_DIR.exists()
        else set()
    )
    t_hours = sum(v.duration or 0 for v in ch.videos if v.id in t_ids) / 3600
    table = Table(title=f"{config.CHANNEL_URL} — {n} videos, {hours:.0f} h")
    table.add_column("stage")
    table.add_column("done", justify="right")
    table.add_column("%", justify="right")
    for name, done in [
        ("metadata", count(config.META_DIR)),
        ("enriched", count(config.ENRICH_DIR)),
        ("transcribed", len(t_ids)),
    ]:
        table.add_row(name, str(done), f"{100 * done / max(n, 1):.1f}")
    table.add_row("transcribed hours", f"{t_hours:.0f}", f"{100 * t_hours / max(hours, 1):.1f}")
    table.add_row("playlists", str(len(ch.playlists)), "")
    console.print(table)


if __name__ == "__main__":
    app()
