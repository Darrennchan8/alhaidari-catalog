import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from catalog import audio, config, quality, store, transcribe
from catalog.enrich import plan_job
from catalog.gemini import QuotaExhausted
from catalog.models import ChunkInfo, Enrichment, Segment, Transcript, VideoMeta
from catalog.taxonomy import Assignment, plan_assignments

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_rank_orders_models_and_handles_unknown_and_mixed() -> None:
    assert quality.rank(None) == quality.ABSENT
    assert quality.rank("best") > quality.rank("good") > quality.rank("lite") > quality.UNRANKED
    assert quality.rank("unknown (best or good)") == quality.UNRANKED
    assert quality.rank("best, lite") == quality.rank("lite")  # mixed ranks as its worst


def test_better_models() -> None:
    assert quality.better_models(None) == ["best", "good", "ok", "lite"]
    assert quality.better_models("ok") == ["best", "good"]
    assert quality.better_models("best") == []
    assert quality.better_models("something-else") == ["best", "good", "ok", "lite"]
    assert quality.is_best("best") and not quality.is_best("good")


# ---- enrichment planning -----------------------------------------------------------------


def _meta(vid: str = "v") -> VideoMeta:
    return VideoMeta(id=vid, title_ar="عنوان", fetched_at=T0)


def _enr(model: str, source: str = "metadata", at: datetime = T0) -> Enrichment:
    return Enrichment(id="v", model=model, created_at=at, source=source, title_en="T", summary="S")


def _tr(model: str = "best", at: datetime = T0) -> Transcript:
    return Transcript(
        id="v",
        model=model,
        created_at=at,
        duration=10,
        chunks=1,
        segments=[Segment(start=0, end=10, text="x")],
    )


def test_plan_job_priorities() -> None:
    missing = plan_job(_meta(), [], None, None)
    assert missing and missing.reason == "missing" and missing.priority == 0
    assert missing.models == ["best", "good", "ok", "lite"]

    # Transcript exists but the summary came from metadata → stale, any model may redo it.
    stale = plan_job(_meta(), [], _tr(), _enr("best"))
    assert stale and stale.reason == "stale"
    # Transcript newer than a transcript-based summary → stale too.
    newer = plan_job(_meta(), [], _tr(at=T0 + timedelta(hours=1)), _enr("best", "transcript"))
    assert newer and newer.reason == "stale"

    lite = plan_job(_meta(), [], None, _enr("lite"))
    good = plan_job(_meta(), [], None, _enr("good"))
    assert lite and good and lite.priority < good.priority  # lowest quality first
    assert good.models == ["best"]  # only strictly better models may replace it

    assert plan_job(_meta(), [], None, _enr("best")) is None
    assert plan_job(_meta(), [], _tr(), _enr("best", "transcript", T0 + timedelta(1))) is None


def test_plan_assignments_orders_missing_stale_then_upgrades() -> None:
    def e(vid: str, at: datetime = T0) -> Enrichment:
        return Enrichment(id=vid, model="best", created_at=at, source="metadata", title_en="T")

    assigned = {
        "stale": Assignment(topics=["a/b"], model="best", assigned_at=T0),
        "lite": Assignment(topics=["a/b"], model="lite", assigned_at=T0 + timedelta(1)),
        "done": Assignment(topics=["a/b"], model="best", assigned_at=T0 + timedelta(1)),
    }
    plan = plan_assignments(
        [e("lite"), e("done"), e("stale", T0 + timedelta(hours=1)), e("missing")], assigned
    )
    assert [x.id for x, _ in plan] == ["missing", "stale", "lite"]
    assert plan[-1][1] == ["best", "good", "ok"]


# ---- transcript chunk checks and upgrades -------------------------------------------------


def _seg(start: str, end: str, words: int = 40) -> transcribe._OutSegment:
    return transcribe._OutSegment(
        start=start, end=end, speaker="al-Haydari", kind="speech", text="w " * words, terms=[]
    )


def test_timestamp_overrun_is_rescaled() -> None:
    chunk = audio.Chunk(index=0, start=100.0, end=700.0, path=Path("x"))  # 10-minute clip
    # The model's clock ran 40% fast: it claims the clip lasts 14 minutes.
    out = transcribe._ChunkOut(segments=[_seg(f"{m:02d}:00", f"{m + 1:02d}:00") for m in range(14)])
    assert transcribe.timing_overrun(out, chunk) == pytest.approx(1.4)
    segs = transcribe.offset_segments(out, chunk)
    assert segs[-1].start == pytest.approx(100 + 13 * 60 / 1.4, abs=0.01)
    assert all(s.end - s.start > 1 for s in segs)  # nothing collapsed onto the end


def test_single_bad_timestamp_is_clamped_not_rescaled() -> None:
    chunk = audio.Chunk(index=0, start=0.0, end=600.0, path=Path("x"))
    out = transcribe._ChunkOut(segments=[_seg("00:00", "05:00"), _seg("05:00", "99:00")])
    assert transcribe.timing_overrun(out, chunk) == 1.0
    assert transcribe.offset_segments(out, chunk)[0].end == 300.0


def test_check_coverage_flags_collapsed_segments() -> None:
    chunk = audio.Chunk(index=0, start=0.0, end=600.0, path=Path("x"))
    segs = [Segment(start=m * 60, end=(m + 1) * 60, text="w " * 200) for m in range(10)]
    segs += [Segment(start=600, end=600, text="tail") for _ in range(3)]
    assert "no duration" in (transcribe.check_coverage(segs, chunk) or "")


@pytest.mark.parametrize(
    ("new_words", "new_problem", "old_problem", "accepted"),
    [
        (100, None, None, True),
        (70, None, None, False),  # much shorter → likely summarised
        (100, "covers only 300s", None, False),  # new fails checks, old didn't
        (100, "covers only 300s", "covers only 200s", True),  # both flawed → prefer better model
    ],
)
def test_accept_upgrade(new_words, new_problem, old_problem, accepted) -> None:
    old = [Segment(start=0, end=10, text="w " * 100)]
    new = [Segment(start=0, end=10, text="w " * new_words)]
    assert (transcribe.accept_upgrade(new, new_problem, old, old_problem) is None) == accepted


def test_segments_by_chunk_uses_counts() -> None:
    # Two segments collapsed onto the boundary at 10s belong to chunk 0, not chunk 1.
    segs = [Segment(start=0, end=10, text="a")] + [Segment(start=10, end=10, text="b")] * 2
    segs += [Segment(start=10, end=20, text="c")]
    t = Transcript(
        id="v",
        model="good",
        created_at=T0,
        duration=20,
        chunks=2,
        segments=segs,
        chunk_info=[
            ChunkInfo(start=0, end=10, model="good", translated_at=T0, segment_count=3),
            ChunkInfo(start=10, end=20, model="best", translated_at=T0, segment_count=1),
        ],
    )
    assert [len(c) for c in t.segments_by_chunk()] == [3, 1]


class FakeGemini:
    """Answers every chunk with a fixed translation, as the best model in the allowed list."""

    def __init__(self, exhausted: set[str] = frozenset(), words: int = 60) -> None:
        self.exhausted = set(exhausted)
        self.words = words
        self.calls: list[list[str]] = []
        self.last_model: str | None = None

    def available(self, models: list[str]) -> list[str]:
        return [m for m in models if m not in self.exhausted]

    def generate(self, models, contents, schema, **_):
        usable = self.available(models)
        if not usable:
            raise QuotaExhausted("all exhausted")
        self.calls.append(list(models))
        self.last_model = usable[0]
        return transcribe._ChunkOut(
            segments=[_seg(f"{m:02d}:00", f"{m + 1:02d}:00", self.words) for m in range(10)]
        )


@pytest.fixture
def fake_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(config, "CHUNK_DIR", tmp_path / "chunks")
    monkeypatch.setattr(config, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    src = tmp_path / "audio" / "v.webm"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"")
    monkeypatch.setattr(transcribe.ytdlp, "download_audio", lambda vid, d: src)

    def chunks(src_, out_dir, target, cuts=None):
        cuts = cuts or [0.0, 600.0, 1200.0]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for i, (a, b) in enumerate(itertools.pairwise(cuts)):
            p = out_dir / f"{i:03d}.ogg"
            p.write_bytes(b"ogg")
            result.append(audio.Chunk(index=i, start=a, end=b, path=p))
        return result

    monkeypatch.setattr(transcribe.audio, "make_chunks", chunks)
    return tmp_path


def _existing(models: list[str], words: int = 60, problems=(None, None)) -> Transcript:
    segs, infos = [], []
    for i, (m, problem) in enumerate(zip(models, problems, strict=True)):
        start = i * 600.0
        chunk = [
            Segment(start=start + k * 60, end=start + (k + 1) * 60, text=f"old{i} " * words)
            for k in range(10)
        ]
        segs += chunk
        infos.append(
            ChunkInfo(
                start=start,
                end=start + 600,
                model=m,
                translated_at=T0,
                problem=problem,
                segment_count=len(chunk),
            )
        )
    return Transcript(
        id="v",
        model=", ".join(models),
        created_at=T0,
        duration=1200,
        chunks=2,
        chunk_info=infos,
        segments=segs,
    )


def test_new_transcript_records_model_per_chunk(fake_audio) -> None:
    gem = FakeGemini(exhausted={"best"})
    t = transcribe.transcribe_video(gem, _meta(), existing=None)
    assert t is not None and t.model == "good"
    assert [c.model for c in t.chunk_info] == ["good", "good"]
    assert [c.segment_count for c in t.chunk_info] == [10, 10]
    assert store.read_model(config.TRANSCRIPT_DIR / "v.json", Transcript).model == "good"


def test_new_transcript_raises_when_no_model_has_quota(fake_audio) -> None:
    with pytest.raises(QuotaExhausted):
        transcribe.transcribe_video(FakeGemini(exhausted={"best", "good", "ok", "lite"}), _meta())


def test_upgrade_retranslates_only_lower_ranked_chunks(fake_audio) -> None:
    gem = FakeGemini()
    t = transcribe.transcribe_video(gem, _meta(), existing=_existing(["best", "lite"]))
    assert t is not None
    assert len(gem.calls) == 1 and gem.calls[0] == ["best", "good", "ok"]  # chunk 2 only
    assert [c.model for c in t.chunk_info] == ["best", "best"]
    assert t.segments[0].text.startswith("old0")  # chunk 1 kept as is
    assert not t.segments[10].text.startswith("old1")  # chunk 2 replaced
    assert t.created_at > T0  # newer → enrichment becomes stale


def test_upgrade_skipped_without_download_when_no_better_model(fake_audio, monkeypatch) -> None:
    monkeypatch.setattr(transcribe.ytdlp, "download_audio", lambda *a: pytest.fail("downloaded"))
    gem = FakeGemini(exhausted={"best", "good"})
    assert transcribe.transcribe_video(gem, _meta(), existing=_existing(["best", "ok"])) is None


def test_upgrade_rejected_when_much_shorter(fake_audio) -> None:
    gem = FakeGemini(words=60)  # passes coverage checks, but 40% shorter than before
    old = _existing(["good", "good"], words=100)
    assert transcribe.transcribe_video(gem, _meta(), existing=old) is None
    assert len(gem.calls) == 2


@pytest.mark.parametrize(
    ("now_utc", "reset_utc"),
    [
        # 01:59 EDT (05:59 UTC) → midnight Pacific is 07:00 UTC the same day.
        (datetime(2026, 10, 6, 5, 59, tzinfo=UTC), datetime(2026, 10, 6, 7, 0, tzinfo=UTC)),
        # 10:00 EDT → next midnight Pacific, 07:00 UTC tomorrow.
        (datetime(2026, 10, 6, 14, 0, tzinfo=UTC), datetime(2026, 10, 7, 7, 0, tzinfo=UTC)),
        # Winter (PST, UTC-8).
        (datetime(2026, 12, 1, 12, 0, tzinfo=UTC), datetime(2026, 12, 2, 8, 0, tzinfo=UTC)),
    ],
)
def test_next_quota_reset_is_midnight_pacific(now_utc, reset_utc) -> None:
    from catalog.gemini import next_quota_reset

    assert next_quota_reset(now_utc.timestamp()) == reset_utc.timestamp()
