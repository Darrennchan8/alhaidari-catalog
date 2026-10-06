import json
from datetime import UTC, datetime

import pytest

from catalog import config, export, store
from catalog.models import (
    Category,
    Channel,
    Enrichment,
    Playlist,
    PlaylistRef,
    Segment,
    Subtopic,
    Taxonomy,
    Transcript,
    VideoMeta,
    VideoStub,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("title", "episode"),
    [
        ("مفاتيح عملية الاستنباط الفقهي 287", 287),
        ("حديث الثقلين سنده ودلالته ق (25)", 25),
        ("بحوث حول الإمام المهدي المنتظر (عج) الحلقة (٦)", 6),
        ("من أين تستمد الشعائر الدينية مشروعيتها؟", None),
    ],
)
def test_episode_from_title(title: str, episode: int | None) -> None:
    assert export.episode_from_title(title) == episode


def test_slugify() -> None:
    assert export.slugify("Dialogue on Religion & Secularism") == "dialogue-on-religion-secularism"
    assert export.slugify("Mulla Ṣadrā’s Philosophy") == "mulla-sadras-philosophy"
    assert export.slugify("؟؟") == "untitled"


@pytest.fixture
def catalog_dir(tmp_path, monkeypatch):
    cat = tmp_path / "catalog"
    for name, value in {
        "CATALOG": cat,
        "SITE_DATA": tmp_path / "site",
        "CHANNEL_FILE": cat / "channel.json",
        "META_DIR": cat / "meta",
        "TRANSCRIPT_DIR": cat / "transcripts",
        "ENRICH_DIR": cat / "enrich",
        "TAXONOMY_FILE": cat / "taxonomy.json",
        "ASSIGNMENTS_FILE": cat / "topic_assignments.json",
        "PLAYLISTS_EN_FILE": cat / "playlists_en.json",
        "OVERRIDES_FILE": tmp_path / "overrides.yaml",
    }.items():
        monkeypatch.setattr(config, name, value)
    return tmp_path


def test_export_merges_all_sources(catalog_dir) -> None:
    store.write_json(
        config.CHANNEL_FILE,
        Channel(
            url="https://www.youtube.com/@x",
            synced_at=NOW,
            videos=[
                VideoStub(
                    id="a",
                    title_ar="عنوان 1",
                    duration=600,
                    playlists=[PlaylistRef(id="PL1", position=1)],
                ),
                VideoStub(id="b", title_ar="عنوان 2", duration=1200),
            ],
            playlists=[Playlist(id="PL1", title_ar="سلسلة", video_ids=["a", "gone"])],
        ),
    )
    store.write_json(
        config.PLAYLISTS_EN_FILE, {"PL1": {"title_en": "A Series", "description_en": ""}}
    )
    store.write_json(
        config.META_DIR / "a.json",
        VideoMeta(
            id="a", title_ar="عنوان 1", upload_date="2020-01-02", duration=600, fetched_at=NOW
        ),
    )
    store.write_json(
        config.ENRICH_DIR / "a.json",
        Enrichment(
            id="a",
            model="m",
            created_at=NOW,
            source="transcript",
            title_en="Title One",
            summary="S",
            topics=["x"],
        ),
    )
    store.write_json(
        config.TRANSCRIPT_DIR / "a.json",
        Transcript(
            id="a",
            model="m",
            created_at=NOW,
            duration=600,
            chunks=1,
            segments=[Segment(start=0, end=10, text="Hello")],
        ),
    )
    store.write_json(
        config.TAXONOMY_FILE,
        Taxonomy(
            model="m",
            created_at=NOW,
            categories=[
                Category(
                    slug="phil", name="Philosophy", subtopics=[Subtopic(slug="being", name="Being")]
                )
            ],
        ),
    )
    store.write_json(
        config.ASSIGNMENTS_FILE,
        {
            "a": {
                "topics": ["phil/being", "phil/removed"],
                "model": "m",
                "assigned_at": NOW.isoformat(),
            }
        },
    )
    config.OVERRIDES_FILE.write_text("topics:\n  b: [phil/being]\n")

    stats = export.export_site()

    assert stats["videos"] == 2 and stats["transcribed"] == 1 and stats["translated"] == 1
    site = config.SITE_DATA
    index = {v["id"]: v for v in json.loads((site / "index.json").read_text())}
    assert index["a"]["title_en"] == "Title One"
    assert index["a"]["topics"] == ["phil/being"]  # unknown key dropped
    assert index["a"]["playlists"] == ["a-series"]
    assert index["a"]["episode"] == 1
    assert index["b"]["title_en"] == ""  # untranslated → UI falls back to Arabic
    assert index["b"]["topics"] == ["phil/being"]  # from overrides
    video_a = json.loads((site / "videos" / "a.json").read_text())
    assert video_a["segments"][0]["text"] == "Hello"
    playlists = json.loads((site / "playlists.json").read_text())
    assert playlists[0]["video_ids"] == ["a"]  # missing video filtered
    topics = json.loads((site / "topics.json").read_text())
    assert topics[0]["count"] == 2 and topics[0]["subtopics"][0]["count"] == 2
