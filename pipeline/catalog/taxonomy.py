"""Stage: consolidate free-form topics into a two-level taxonomy, then assign each video
1–4 canonical topics."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

import yaml
from pydantic import BaseModel

from . import config, prompts, store
from .gemini import Gemini
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
        config.TAXONOMY_MODEL,
        [prompts.TAXONOMY_PROMPT.format(n=len(enrichments), labels=labels)],
        _TaxOut,
        thinking_level="high",
        fallbacks=config.ENRICH_FALLBACK_MODELS,
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


def assign_topics(
    gem: Gemini, tax: Taxonomy, enrichments: list[Enrichment], rebuild: bool = False
) -> dict[str, list[str]]:
    valid = set(topic_keys(tax))
    existing: dict[str, list[str]] = (
        {}
        if rebuild or not config.ASSIGNMENTS_FILE.exists()
        else store.read_json(config.ASSIGNMENTS_FILE)  # type: ignore[assignment]
    )
    # Drop assignments that point at keys no longer in the taxonomy.
    existing = {k: [t for t in v if t in valid] for k, v in existing.items()}
    existing = {k: v for k, v in existing.items() if v}
    todo = [e for e in enrichments if e.id not in existing]
    keys_text = "\n".join(
        f"{c.slug}/{s.slug} — {c.name} › {s.name}" for c in tax.categories for s in c.subtopics
    )
    for i in range(0, len(todo), ASSIGN_BATCH):
        batch = todo[i : i + ASSIGN_BATCH]
        videos = "\n".join(
            f"- id={e.id} | {e.title_en} | topics: {', '.join(e.topics)} | {e.summary[:240]}"
            for e in batch
        )
        out = gem.generate(
            config.ENRICH_MODEL,
            [ASSIGN_PROMPT.format(keys=keys_text, videos=videos)],
            _AssignOut,
            fallbacks=config.ENRICH_FALLBACK_MODELS,
        )
        ids = {e.id for e in batch}
        for it in out.items:
            picked = [t for t in dict.fromkeys(it.topics) if t in valid][:4]
            if it.id in ids and picked:
                existing[it.id] = picked
        store.write_json(config.ASSIGNMENTS_FILE, existing)
        log.info("assigned topics %d/%d", min(i + ASSIGN_BATCH, len(todo)), len(todo))
    return existing


def load_overrides() -> dict[str, list[str]]:
    if not config.OVERRIDES_FILE.exists():
        return {}
    data = yaml.safe_load(config.OVERRIDES_FILE.read_text(encoding="utf-8")) or {}
    return {str(k): list(v) for k, v in (data.get("topics") or {}).items()}
