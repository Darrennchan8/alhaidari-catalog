"""Model quality ranking, used to decide which results are worth upgrading and with what."""

from __future__ import annotations

from . import config

ABSENT = -1  # no result at all
UNRANKED = 0  # a model that isn't in MODEL_RANKING (or an unknown/legacy record)


def rank(model: str | None) -> int:
    """Higher is better. A comma-separated list (a mixed-model transcript) ranks as its worst."""
    if model is None:
        return ABSENT
    parts = [m.strip() for m in model.split(",") if m.strip()]
    if len(parts) > 1:
        return min(rank(m) for m in parts)
    ranking = config.MODEL_RANKING
    return len(ranking) - ranking.index(model) if model in ranking else UNRANKED


def best_rank() -> int:
    return len(config.MODEL_RANKING)


def better_models(model: str | None) -> list[str]:
    """Ranked models strictly better than `model`, best first. Empty when nothing would upgrade it."""
    current = rank(model)
    return [m for m in config.MODEL_RANKING if rank(m) > current]


def is_best(model: str | None) -> bool:
    return rank(model) >= best_rank()
