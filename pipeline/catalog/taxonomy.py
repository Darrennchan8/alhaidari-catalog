"""Stage: consolidate free-form topics into a two-level taxonomy, then assign each video
1–4 canonical topics."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

import yaml
from pydantic import BaseModel

from . import config, prompts, quality, store
from .gemini import Gemini, Overloaded, QuotaExhausted
from .models import Category, Enrichment, Taxonomy

log = logging.getLogger(__name__)

MAX_LABELS = 600
ASSIGN_BATCH = 80


class _Sub(BaseModel):
    slug: str
    name: str
    description: str


class _Cat(BaseModel):
    slug: str
    name: str
    description: str
    subtopics: list[_Sub]


class _TaxOut(BaseModel):
    categories: list[_Cat]


class _Assign(BaseModel):
    id: str
    topics: list[str]


class _AssignOut(BaseModel):
    items: list[_Assign]


def topic_keys(tax: Taxonomy) -> list[str]:
    return [f"{c.slug}/{s.slug}" for c in tax.categories for s in c.subtopics]


def load_enrichments() -> list[Enrichment]:
    return [
        Enrichment.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(config.ENRICH_DIR.glob("*.json"))
    ]


def build_taxonomy(gem: Gemini, enrichments: list[Enrichment]) -> Taxonomy:
    counts = Counter(t.lower() for e in enrichments for t in e.topics)
    labels = "\n".join(f"{n}\t{t}" for t, n in counts.most_common(MAX_LABELS))
    out = gem.generate(
        [config.TAXONOMY_MODEL, *config.MODEL_RANKING],
        [prompts.TAXONOMY_PROMPT.format(n=len(enrichments), labels=labels)],
        _TaxOut,
        thinking_level="high",
    )
    tax = Taxonomy(
        model=gem.last_model or config.TAXONOMY_MODEL,
        created_at=datetime.now(UTC),
        categories=[Category.model_validate(c.model_dump()) for c in out.categories],
    )
    store.write_json(config.TAXONOMY_FILE, tax)
    return tax


ASSIGN_PROMPT = """\
Assign each video 1–4 topics from this taxonomy (use the exact keys; choose the most \
specific fitting subtopics; the first is the primary topic).

Taxonomy keys:
{keys}

Videos:
{videos}
"""


class Assignment(BaseModel):
    topics: list[str]
    model: str
    assigned_at: datetime


def load_assignments() -> dict[str, Assignment]:
    if not config.ASSIGNMENTS_FILE.exists():
        return {}
    raw: dict = store.read_json(config.ASSIGNMENTS_FILE)  # type: ignore[assignment]
    return {k: Assignment.model_validate(v) for k, v in raw.items()}


def _save_assignments(assigned: dict[str, Assignment]) -> None:
    store.write_json(
        config.ASSIGNMENTS_FILE,
        {k: v.model_dump(mode="json") for k, v in sorted(assigned.items())},
    )


def plan_assignments(
    enrichments: list[Enrichment], assigned: dict[str, Assignment]
) -> list[tuple[Enrichment, list[str]]]:
    """Videos needing topics, with the models allowed for each: missing first, then stale (the
    enrichment changed since), then upgrades of assignments made by lower-ranked models."""
    jobs: list[tuple[int, Enrichment, list[str]]] = []
    for e in enrichments:
        a = assigned.get(e.id)
        if a is None:
            jobs.append((0, e, list(config.MODEL_RANKING)))
        elif e.created_at > a.assigned_at:
            jobs.append((1, e, list(config.MODEL_RANKING)))
        elif better := quality.better_models(a.model):
            jobs.append((2 + quality.rank(a.model), e, better))
    jobs.sort(key=lambda j: j[0])
    return [(e, models) for _, e, models in jobs]


def assign_topics(
    gem: Gemini, tax: Taxonomy, enrichments: list[Enrichment], rebuild: bool = False
) -> dict[str, Assignment]:
    valid = set(topic_keys(tax))
    assigned = {} if rebuild else load_assignments()
    # Drop topic keys that are no longer in the taxonomy; reassign videos left with none.
    for k, a in list(assigned.items()):
        a.topics = [t for t in a.topics if t in valid]
        if not a.topics:
            del assigned[k]
    keys_text = "\n".join(
        f"{c.slug}/{s.slug} — {c.name} › {s.name}" for c in tax.categories for s in c.subtopics
    )
    # Batch jobs that share the same allowed models.
    groups: dict[tuple[str, ...], list[Enrichment]] = {}
    for e, models in plan_assignments(enrichments, assigned):
        groups.setdefault(tuple(models), []).append(e)
    total = sum(len(v) for v in groups.values())
    done = 0
    for models, todo in groups.items():
        for i in range(0, len(todo), ASSIGN_BATCH):
            usable = gem.available(list(models))
            if not usable:
                log.info("no quota left for %d assignment(s) needing %s", len(todo) - i, models[0])
                break
            batch = todo[i : i + ASSIGN_BATCH]
            videos = "\n".join(
                f"- id={e.id} | {e.title_en} | topics: {', '.join(e.topics)} | {e.summary[:240]}"
                for e in batch
            )
            try:
                out = gem.generate(
                    usable, [ASSIGN_PROMPT.format(keys=keys_text, videos=videos)], _AssignOut
                )
            except (QuotaExhausted, Overloaded) as e:
                log.warning("topic assignment paused: %s", e)
                break
            model = gem.last_model or usable[0]
            now = datetime.now(UTC)
            ids = {e.id for e in batch}
            for it in out.items:
                picked = [t for t in dict.fromkeys(it.topics) if t in valid][:4]
                if it.id in ids and picked:
                    assigned[it.id] = Assignment(topics=picked, model=model, assigned_at=now)
            _save_assignments(assigned)
            done += len(batch)
            log.info("assigned topics %d/%d", done, total)
    return assigned


def load_overrides() -> dict[str, list[str]]:
    if not config.OVERRIDES_FILE.exists():
        return {}
    data = yaml.safe_load(config.OVERRIDES_FILE.read_text(encoding="utf-8")) or {}
    return {str(k): list(v) for k, v in (data.get("topics") or {}).items()}
