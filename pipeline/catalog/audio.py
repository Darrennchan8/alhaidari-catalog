"""Audio preparation: normalise to small mono Opus and cut into chunks at silences."""

from __future__ import annotations

import itertools
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float
    path: Path


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise RuntimeError(f"cannot read duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def detect_silences(path: Path, noise_db: int = -32, min_len: float = 0.4) -> list[float]:
    """Return midpoints of silent stretches (seconds)."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_len}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?\d+(?:\.\d+)?)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: (\d+(?:\.\d+)?)", out)]
    return [(s + e) / 2 for s, e in zip(starts, ends, strict=False)]


def plan_cuts(
    duration: float, silences: list[float], target: float, window: float = 90.0
) -> list[float]:
    """Choose cut points near every `target` seconds, snapping to the closest silence within
    `window` seconds. Returns boundaries including 0 and `duration`."""
    cuts = [0.0]
    while duration - cuts[-1] > target * 1.25:
        ideal = cuts[-1] + target
        near = [s for s in silences if abs(s - ideal) <= window and s > cuts[-1] + target / 2]
        cuts.append(min(near, key=lambda s: abs(s - ideal)) if near else ideal)
    cuts.append(duration)
    return cuts


def make_chunks(src: Path, out_dir: Path, target_seconds: int) -> list[Chunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(src)
    cuts = plan_cuts(duration, detect_silences(src), float(target_seconds))
    chunks: list[Chunk] = []
    for i, (a, b) in enumerate(itertools.pairwise(cuts)):
        dest = out_dir / f"{i:03d}.ogg"
        if not dest.exists():
            tmp = dest.with_suffix(".tmp.ogg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{a:.3f}",
                    "-to",
                    f"{b:.3f}",
                    "-i",
                    str(src),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "24k",
                    str(tmp),
                ],
                check=True,
            )
            tmp.rename(dest)
        chunks.append(Chunk(index=i, start=a, end=b, path=dest))
    return chunks
