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


def _transcripts() -> dict[str, Transcript]:
    if not config.TRANSCRIPT_DIR.exists():
        return {}
    return {
        p.stem: store.read_model(p, Transcript)
        for p in sorted(config.TRANSCRIPT_DIR.glob("*.json"))
    }


@app.command()
def transcribe(
    ids: list[str] = typer.Argument(None, help="specific video ids (default: by --order)"),
    order: Order = typer.Option(Order.playlist),
    limit: int = typer.Option(0, help="max videos to process this run (0 = no limit)"),
    workers: int = typer.Option(2, help="videos processed in parallel"),
    max_hours: float = typer.Option(0, help="skip videos longer than this (0 = no limit)"),
    upgrade: bool = typer.Option(
        True, help="after new videos, re-translate chunks done by lower-ranked models"
    ),
    keep_audio: bool = typer.Option(False),
    enrich_after: bool = typer.Option(True, help="re-run enrichment from the new transcript"),
) -> None:
    """Translate audio into timestamped English transcripts: untranscribed videos first (in
    --order), then upgrades of the lowest-ranked chunks, as far as model quota allows."""
    from . import quality
    from .enrich import enrich_video, plan_job
    from .gemini import Gemini, QuotaExhausted
    from .transcribe import transcribe_video

    ch = _channel()
    pl_titles = {p.id: p.title_ar for p in ch.playlists}
    stubs = {v.id: v for v in ch.videos}
    existing = _transcripts()
    ordered = ids or _ordered_ids(ch, order)
    position = {vid: n for n, vid in enumerate(ordered)}
    new = [i for i in ordered if i not in existing]
    if max_hours:
        new = [i for i in new if (stubs[i].duration or 0) <= max_hours * 3600]
    upgrades = (
        sorted(
            (i for i in ordered if i in existing and not quality.is_best(existing[i].model)),
            key=lambda i: (quality.rank(existing[i].model), position[i]),
        )
        if upgrade
        else []
    )
    queue = new + upgrades
    if limit:
        queue = queue[:limit]
    console.print(f"{len(new)} new and {len(upgrades)} upgradable transcripts; {len(queue)} queued")
    gem = Gemini()
    stop_new = False

    def one(vid: str) -> str:
        old = existing.get(vid)
        if old is None and stop_new:
            return f"{vid}: skipped (no quota)"
        m = _meta(vid)
        if m is None:
            m = ytdlp.fetch_meta(vid)
            store.write_json(config.META_DIR / f"{vid}.json", m)
        t = transcribe_video(gem, m, existing=old, keep_audio=keep_audio)
        if t is None:
            return f"{vid}: unchanged"
        if enrich_after:
            e = store.maybe_model(config.ENRICH_DIR / f"{vid}.json", Enrichment)
            job = plan_job(m, [pl_titles[p.id] for p in stubs[vid].playlists], t, e)
            if job and gem.available(job.models):
                try:
                    enrich_video(gem, job)
                except QuotaExhausted:
                    log.warning("%s: no quota to re-enrich; `catalog enrich` will catch up", vid)
        return f"{vid}: {t.model} ({len(t.segments)} segments)"

    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, v): v for v in queue}
        for f in as_completed(futs):
            try:
                log.info("done %s", f.result())
            except QuotaExhausted as e:
                stop_new = True
                log.error("no model has quota for new transcripts; continuing upgrades only: %s", e)
            except Exception:
                log.exception("transcribe %s failed", futs[f])
    console.print(f"usage: {gem.usage}")


@app.command()
def enrich(
    transcribed_only: bool = typer.Option(False, help="only (re)enrich transcribed videos"),
    refresh: bool = typer.Option(False, help="redo every video, with the best available model"),
    workers: int = typer.Option(2),
    limit: int = typer.Option(0, help="max videos to process (0 = until quota)"),
    batch_size: int = typer.Option(30, help="metadata-only videos per request"),
    only: list[str] = typer.Option(
        None, help="restrict to job kinds: missing, stale, upgrade (repeatable)"
    ),
) -> None:
    """English titles, summaries, topics and references. Works in priority order: missing,
    then stale (transcript newer than the summary), then upgrades from the lowest-ranked model
    up, stopping when no allowed model has quota."""
    from .enrich import (
        EnrichJob,
        enrich_metadata_batch,
        enrich_video,
        plan_job,
        translate_playlists,
    )
    from .gemini import Gemini, QuotaExhausted

    ch = _channel()
    gem = Gemini()
    try:
        translate_playlists(gem, ch)
    except QuotaExhausted:
        log.warning("no quota to translate playlists")
    pl_titles = {p.id: p.title_ar for p in ch.playlists}
    jobs: list[EnrichJob] = []
    for v in ch.videos:
        m = _meta(v.id)
        if m is None:
            continue
        t = store.maybe_model(config.TRANSCRIPT_DIR / f"{v.id}.json", Transcript)
        if transcribed_only and t is None:
            continue
        e = None if refresh else store.maybe_model(config.ENRICH_DIR / f"{v.id}.json", Enrichment)
        job = plan_job(m, [pl_titles[p.id] for p in v.playlists], t, e)
        if job and (not only or job.reason in only):
            if refresh:
                job.current = store.maybe_model(config.ENRICH_DIR / f"{v.id}.json", Enrichment)
            jobs.append(job)
    jobs.sort(key=lambda j: j.priority)
    if limit:
        jobs = jobs[:limit]

    # Transcribed videos get a call each; the rest go in batches of jobs sharing allowed models.
    units: list[tuple[int, list[EnrichJob]]] = []
    batches: dict[tuple[int, tuple[str, ...]], list[EnrichJob]] = {}
    for job in jobs:
        if job.transcript is not None and job.transcript.segments:
            units.append((job.priority, [job]))
        else:
            batches.setdefault((job.priority, tuple(job.models)), []).append(job)
    for (prio, _), group in batches.items():
        units += [(prio, group[i : i + batch_size]) for i in range(0, len(group), batch_size)]
    units.sort(key=lambda u: u[0])
    reasons = {r: sum(j.reason == r for j in jobs) for r in ("missing", "stale", "upgrade")}
    console.print(f"{len(jobs)} videos to enrich {reasons} in {len(units)} requests")

    def run_unit(unit: list[EnrichJob]) -> int:
        if not gem.available(unit[0].models):
            return 0
        try:
            if len(unit) == 1 and unit[0].transcript is not None and unit[0].transcript.segments:
                return int(enrich_video(gem, unit[0]) is not None)
            return len(enrich_metadata_batch(gem, unit))
        except QuotaExhausted:
            return 0

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(run_unit, u): n for n, (_, u) in enumerate(units)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                done += f.result()
            except Exception as e:
                log.error("enrich request %s failed: %s", futs[f], e)
            if n % 10 == 0:
                log.info("enrich requests %d/%d (%d videos updated)", n, len(units), done)
            if not gem.available(config.MODEL_RANKING):
                log.warning("every ranked model is out of quota; stopping")
                pool.shutdown(cancel_futures=True)
                break
    console.print(f"[green]{done} videos updated; usage: {gem.usage}")


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
    """Run every stage in order (suitable for a daily cron job). Each stage fills gaps first,
    then upgrades lower-ranked results, until model quota runs out.

    Transcription goes first: a transcript re-hydrates the video's summary (done right after
    each transcript), so quota spent upgrading a summary from title/description would be wasted
    once that video is transcribed. Enrichment then uses whatever quota is left."""
    if resync:
        sync()
    meta(workers=workers, refresh=False, limit=0)
    # New uploads get English titles first; it's cheap (one request per 30 videos).
    enrich(
        transcribed_only=False,
        refresh=False,
        workers=workers,
        limit=0,
        batch_size=30,
        only=["missing"],
    )
    transcribe(
        ids=None,
        order=order,
        limit=transcribe_limit,
        workers=workers,
        max_hours=0,
        upgrade=True,
        keep_audio=False,
        enrich_after=True,
    )
    enrich(
        transcribed_only=False, refresh=False, workers=workers, limit=0, batch_size=30, only=None
    )
    if config.TAXONOMY_FILE.exists() or len(list(config.ENRICH_DIR.glob("*.json"))) > 100:
        taxonomy(rebuild=False)
    export()
    status()


@app.command()
def review(
    key: str = typer.Argument(None, help="chunk to decide on, as VIDEO@START (from the listing)"),
    ok: bool = typer.Option(False, "--ok", help="checked: nothing to fix (hides the flag)"),
    fix: bool = typer.Option(False, "--fix", help="checked: needs a manual correction"),
    note: str = typer.Option("", help="note to store with the decision"),
    show_all: bool = typer.Option(False, "--all", help="include chunks already decided"),
) -> None:
    """List transcript chunks that need a person's attention, or record a decision on one.
    A decision lasts until the chunk is re-translated by another model."""
    from . import review as rv

    transcripts = _transcripts()
    if key:
        if ok == fix:
            raise typer.BadParameter("pass exactly one of --ok / --fix")
        vid = key.partition("@")[0]
        t = transcripts.get(vid)
        info = next(
            (c for c in (t.chunk_info if t else []) if rv.chunk_key(vid, c.start) == key), None
        )
        if info is None:
            raise typer.BadParameter(f"no transcript chunk {key!r}")
        rv.save_decision(vid, info.start, info.model, "ok" if ok else "fix", note)
        console.print(f"[green]{key}: marked {'ok' if ok else 'fix'}")
        return

    flags = rv.flags(transcripts, include_resolved=show_all)
    if not flags:
        console.print("[green]nothing to review")
        return
    table = Table(title=f"{len(flags)} chunk(s) to review")
    for col in ("chunk", "model", "why", "listen", "decision"):
        table.add_column(col, overflow="fold")
    for f in flags:
        decided = f"{f.decision.status}: {f.decision.note}" if f.decision else ""
        table.add_row(f.key, f.info.model, "; ".join(f.reasons), f.url, decided)
    console.print(table)


@app.command()
def status() -> None:
    """Show coverage, and how much of each result type each model produced (best first)."""
    from collections import Counter

    from . import quality
    from .taxonomy import load_assignments

    ch = _channel()
    n = len(ch.videos)
    durations = {v.id: v.duration or 0 for v in ch.videos}
    hours = sum(durations.values()) / 3600
    transcripts = _transcripts()
    enrichments = (
        [store.read_model(p, Enrichment) for p in sorted(config.ENRICH_DIR.glob("*.json"))]
        if config.ENRICH_DIR.exists()
        else []
    )
    assigned = load_assignments()

    t_hours = sum(durations.get(i, 0) for i in transcripts) / 3600
    table = Table(title=f"{config.CHANNEL_URL} — {n} videos, {hours:.0f} h")
    table.add_column("stage")
    table.add_column("done", justify="right")
    table.add_column("%", justify="right")
    meta_n = len(list(config.META_DIR.glob("*.json"))) if config.META_DIR.exists() else 0
    for name, done in [
        ("metadata", meta_n),
        ("enriched", len(enrichments)),
        ("  from transcript", sum(e.source == "transcript" for e in enrichments)),
        ("transcribed", len(transcripts)),
        ("topics assigned", len(assigned)),
    ]:
        table.add_row(name, str(done), f"{100 * done / max(n, 1):.1f}")
    table.add_row("transcribed hours", f"{t_hours:.0f}", f"{100 * t_hours / max(hours, 1):.1f}")
    console.print(table)

    chunk_hours: Counter[str] = Counter()
    for t in transcripts.values():
        infos = t.chunk_info or []
        if infos:
            for c in infos:
                chunk_hours[c.model] += (c.end - c.start) / 3600
        else:
            chunk_hours[t.model] += t.duration / 3600
    columns = {
        "summaries (videos)": Counter(e.model for e in enrichments),
        "topics (videos)": Counter(a.model for a in assigned.values()),
        "transcripts (hours)": chunk_hours,
    }
    models = sorted({m for c in columns.values() for m in c}, key=lambda m: -quality.rank(m))
    q = Table(title="Results by model (ranked best → worst; unranked last)")
    q.add_column("model")
    for name in columns:
        q.add_column(name, justify="right")
    for m in models:
        label = m if quality.rank(m) > quality.UNRANKED else f"{m} (unranked)"
        q.add_row(
            label,
            *[
                (f"{c[m]:.1f}" if name.endswith("(hours)") else f"{c[m]:.0f}") if c[m] else ""
                for name, c in columns.items()
            ],
        )
    console.print(q)

    from . import attempts as att
    from . import review as rv

    log_entries = att.load()
    if log_entries:
        checks = ["coverage", "words", "timing", "overrun", "shorter"]
        f = Table(title="Chunk translation attempts (each row: one model response)")
        f.add_column("model")
        f.add_column("responses", justify="right")
        f.add_column("failed", justify="right")
        for c in checks:
            f.add_column(c, justify="right")
        by_model: dict[str, list] = {}
        for a in log_entries:
            by_model.setdefault(a.model, []).append(a)
        for m in sorted(by_model, key=lambda m: -quality.rank(m)):
            rows = by_model[m]
            failed = sum(a.check is not None for a in rows)
            f.add_row(
                m,
                str(len(rows)),
                f"{100 * failed / len(rows):.0f}%",
                *[str(sum(a.check == c for a in rows) or "") for c in checks],
            )
        console.print(f)
    open_flags = rv.flags(transcripts, history=log_entries)
    console.print(f"chunks awaiting review: {len(open_flags)} (see `catalog review`)")


if __name__ == "__main__":
    app()
