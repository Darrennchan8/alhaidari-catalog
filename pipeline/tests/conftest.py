import pytest

from catalog import config

RANKING = ["best", "good", "ok", "lite"]


@pytest.fixture(autouse=True)
def ranking(monkeypatch):
    """A fixed model ranking for every test, independent of env configuration."""
    monkeypatch.setattr(config, "MODEL_RANKING", list(RANKING))
    return RANKING
