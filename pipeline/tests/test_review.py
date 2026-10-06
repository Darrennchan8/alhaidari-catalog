from datetime import UTC, datetime, timedelta

from catalog import attempts, review, transcribe
from catalog.models import ChunkInfo, Segment, Transcript

from .test_upgrades import FakeGemini, _existing, _meta, fake_audio  # noqa: F401

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _attempt(model: str, problem: str | None, outcome="accepted_with_problem", at=T0, start=0.0):
    return attempts.Attempt(
        video="v",
        chunk=0,
        start=start,
        end=600,
        model=model,
        at=at,
        kind="new",
        outcome=outcome,
        check=attempts.check_kind(problem),
        problem=problem,
    )


def test_check_kind() -> None:
    assert attempts.check_kind(None) is None
    assert attempts.check_kind("covers only 300s of 600s") == "coverage"
    assert attempts.check_kind("only 100 words for 600s") == "words"
    assert attempts.check_kind("16 segments have no duration (…)") == "timing"
    assert attempts.check_kind("timestamps overran the clip by 20% (rescaled)") == "overrun"
    assert attempts.check_kind("new translation is much shorter (1 vs 9 words)") == "shorter"


def test_record_and_load_roundtrip() -> None:
    attempts.record("v", 0, 0, 600, "good", "new", "retry", "only 10 words for 600s")
    attempts.record("v", 1, 600, 1200, "good", "new", "accepted")
    log = attempts.load()
    assert [a.outcome for a in log] == ["retry", "accepted"]
    assert len(attempts.for_chunk(log, "v", 600.2)) == 1
    assert attempts.failed_models(log) == {"good"}


def _info(model: str, problem: str | None = "only 100 words for 600s") -> ChunkInfo:
    return ChunkInfo(start=0, end=600, model=model, translated_at=T0, problem=problem)


def test_reasons_best_model_still_failing() -> None:
    assert review.chunk_reasons("v", _info("best"), [])
    assert not review.chunk_reasons("v", _info("good"), [])  # an upgrade may still fix it
    assert not review.chunk_reasons("v", _info("best", problem=None), [])


def test_reasons_failing_across_models() -> None:
    history = [_attempt("ok", "only 50 words for 600s"), _attempt("good", "only 60 words for 600s")]
    reasons = review.chunk_reasons("v", _info("good"), history)
    assert reasons and "2 models" in reasons[0]


def test_reasons_rejected_shorter_upgrade_only_after_current_translation() -> None:
    later = _attempt(
        "best", "new translation is much shorter (5 vs 9 words)", "rejected", T0 + timedelta(1)
    )
    earlier = later.model_copy(update={"at": T0 - timedelta(1)})
    assert review.chunk_reasons("v", _info("good", problem=None), [later])
    assert not review.chunk_reasons("v", _info("good", problem=None), [earlier])


def test_decision_hides_flag_until_retranslated() -> None:
    t = Transcript(
        id="v",
        model="best",
        created_at=T0,
        duration=600,
        chunks=1,
        chunk_info=[_info("best")],
        segments=[Segment(start=0, end=600, text="x")],
    )
    assert len(review.flags({"v": t}, history=[], decisions={})) == 1

    review.save_decision("v", 0, "best", "ok", "Qur'an recitation")
    decisions = review.load_decisions()
    assert review.flags({"v": t}, history=[], decisions=decisions) == []
    assert review.reader_notice(t.chunk_info[0], decisions["v@0"]) is None

    # A decision made when the chunk came from another model no longer applies.
    stale = {"v@0": decisions["v@0"].model_copy(update={"model": "good"})}
    assert len(review.flags({"v": t}, history=[], decisions=stale)) == 1


def test_reader_notice_messages() -> None:
    assert "Timestamps" in review.reader_notice(_info("x", "3 segments have no duration"), None)
    assert "missing" in review.reader_notice(_info("x", "covers only 1s of 600s"), None)
    assert review.reader_notice(_info("x", None), None) is None


def test_transcribe_logs_attempts_and_skips_retries_on_known_bad_chunks(fake_audio) -> None:  # noqa: F811
    # Chunk 0 has already failed with two models: one try only. Chunk 1: normal 3 tries.
    for model in ("ok", "lite"):
        attempts.record(
            "v", 0, 0, 600, model, "new", "accepted_with_problem", "only 9 words for 600s"
        )
    gem = FakeGemini(words=5)  # every translation fails the word-count check
    transcribe.transcribe_video(gem, _meta(), existing=None)
    assert len(gem.calls) == 1 + 3
    log = attempts.load()[2:]
    assert [(a.chunk, a.outcome) for a in log] == [
        (0, "accepted_with_problem"),
        (1, "retry"),
        (1, "retry"),
        (1, "accepted_with_problem"),
    ]


def test_rejected_upgrade_is_logged(fake_audio) -> None:  # noqa: F811
    gem = FakeGemini(words=60)
    transcribe.transcribe_video(gem, _meta(), existing=_existing(["good", "best"], words=100))
    (a,) = attempts.load()
    assert (a.chunk, a.outcome, a.check) == (0, "rejected", "shorter")
