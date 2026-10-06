import pytest

from catalog import config

RANKING = ["best", "good", "ok", "lite"]


@pytest.fixture(autouse=True)
def ranking(monkeypatch):
    """A fixed model ranking for every test, independent of env configuration."""
    monkeypatch.setattr(config, "MODEL_RANKING", list(RANKING))
    return RANKING


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Keep tests from appending to the real attempt log or review decisions."""
    monkeypatch.setattr(config, "ATTEMPTS_FILE", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(config, "REVIEW_FILE", tmp_path / "review.json")
    from catalog import gemini

    monkeypatch.setattr(gemini, "REQUESTS_FILE", tmp_path / "requests.jsonl")
    monkeypatch.setattr(gemini, "QUOTA_FILE", tmp_path / "quota.json")
