import itertools
from pathlib import Path

import pytest

from catalog.audio import Chunk, plan_cuts
from catalog.transcribe import (
    _ChunkOut,
    _OutSegment,
    check_coverage,
    fmt_ts,
    offset_segments,
    parse_ts,
)


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("00:00", 0),
        ("01:05", 65),
        ("12:30", 750),
        ("1:02:03", 3723),
        ("75:10", 4510),
        ("12.5", 12.5),
    ],
)
def test_parse_ts(raw: str, seconds: float) -> None:
    assert parse_ts(raw) == seconds


def test_parse_ts_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_ts("soon")


def test_fmt_ts() -> None:
    assert fmt_ts(65) == "01:05"
    assert fmt_ts(3723) == "1:02:03"


def test_plan_cuts_snaps_to_nearest_silence() -> None:
    cuts = plan_cuts(2000, silences=[300, 690, 735, 1500], target=720)
    assert cuts[0] == 0 and cuts[-1] == 2000
    assert cuts[1] == 735  # closest silence to 720
    assert len(cuts) == 4  # 0, ~735, ~1455 (no silence near) , end


def test_plan_cuts_short_audio_single_chunk() -> None:
    assert plan_cuts(800, silences=[], target=720) == [0.0, 800]


def _seg(start: str, end: str, text: str = "word " * 20) -> _OutSegment:
    return _OutSegment(
        start=start, end=end, speaker="al-Haydari", kind="speech", text=text, terms=[]
    )


def test_offset_segments_makes_absolute_monotonic_and_clamped() -> None:
    chunk = Chunk(index=1, start=700.0, end=1400.0, path=Path("x"))
    out = _ChunkOut(
        segments=[
            _seg("00:00", "00:30"),
            _seg("00:20", "01:00"),  # overlaps previous → pushed forward
            _seg("01:02", "99:00"),  # beyond chunk → clamped
            _seg("bad", "bad", text="  "),  # empty text dropped
        ]
    )
    segs = offset_segments(out, chunk)
    assert [s.start for s in segs] == [700.0, 730.0, 762.0]
    assert segs[0].end == 730.0  # small gap closed
    assert segs[-1].end == 1400.0
    assert all(a.start <= b.start for a, b in itertools.pairwise(segs))


def test_check_coverage_flags_truncation() -> None:
    chunk = Chunk(index=0, start=0.0, end=720.0, path=Path("x"))
    short = offset_segments(_ChunkOut(segments=[_seg("00:00", "05:00")]), chunk)
    assert "covers only" in (check_coverage(short, chunk) or "")
    full = offset_segments(
        _ChunkOut(segments=[_seg(f"{m:02d}:00", f"{m + 1:02d}:00", "w " * 150) for m in range(12)]),
        chunk,
    )
    assert check_coverage(full, chunk) is None
