import threading
import time
from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from catalog import gemini
from catalog.drivers import WorkItem, WorkQueue, run_drivers
from catalog.gemini import Gemini, Overloaded, QuotaExhausted


class Out(BaseModel):
    text: str


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0  # a fixed, arbitrary instant

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _overloaded():
    return errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )


def _daily_quota():
    msg = "quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    return errors.ClientError(
        429, {"error": {"code": 429, "message": msg, "status": "RESOURCE_EXHAUSTED"}}
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Gemini client on a fake clock whose API answers from a per-model script."""
    monkeypatch.setattr(gemini.config, "GEMINI_MIN_INTERVAL", 0)
    clock = FakeClock()
    gem = Gemini(api_key="test", clock=clock.now, sleep=clock.sleep)
    script: dict[str, list] = {}  # model -> outcomes to replay ("ok" or an exception)
    calls: list[str] = []

    def generate_content(model, contents, config):
        calls.append(model)
        outcome = script.get(model, ["ok"]).pop(0) if script.get(model) else "ok"
        if outcome != "ok":
            raise outcome
        return SimpleNamespace(text='{"text": "hi"}', candidates=[], usage_metadata=None)

    gem.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return SimpleNamespace(gem=gem, clock=clock, script=script, calls=calls)


def test_overloaded_model_cools_down_and_others_take_over(client) -> None:
    gem, script = client.gem, client.script
    script["best"] = [_overloaded(), _overloaded()]
    assert gem.generate(["best", "good"], [], Out).text == "hi"
    assert client.calls == ["best", "good"]
    assert gem.last_model == "good"
    # "best" is cooling for 5 min, so the next call goes straight to "good".
    assert gem.ready(["best", "good"]) == ["good"]
    gem.generate(["best", "good"], [], Out)
    assert client.calls[-1] == "good"
    # After the cooldown, "best" is probed; a second 503 doubles its cooldown to 10 min.
    client.clock.sleep(301)
    gem.generate(["best", "good"], [], Out)
    assert client.calls[-2:] == ["best", "good"]
    assert gem.ready_at(["best"]) == pytest.approx(client.clock.now() + 600)
    # It never backs off for more than 30 min.
    gem._state("best").overloads = 10
    gem._overloaded("best")
    assert gem.ready_at(["best"]) == pytest.approx(client.clock.now() + 1800)


def test_success_resets_backoff(client) -> None:
    client.script["best"] = [_overloaded()]
    gem = client.gem
    gem.generate(["best", "good"], [], Out)
    client.clock.sleep(301)
    gem.generate(["best"], [], Out)
    assert gem._state("best").overloads == 0 and gem.ready(["best"]) == ["best"]


def test_pinned_call_raises_overloaded_instead_of_waiting(client) -> None:
    client.script["best"] = [_overloaded()]
    with pytest.raises(Overloaded) as exc:
        client.gem.generate(["best"], [], Out, wait=False)
    assert exc.value.retry_at == pytest.approx(client.clock.now() + 300)


def test_pooled_call_waits_for_cooldown(client) -> None:
    client.script["best"] = [_overloaded()]
    start = client.clock.now()
    assert client.gem.generate(["best"], [], Out).text == "hi"
    assert client.clock.now() - start == pytest.approx(300)


def test_daily_quota_marks_model_until_midnight_pacific(client) -> None:
    client.script["best"] = [_daily_quota()]
    client.gem.generate(["best", "good"], [], Out)
    assert client.gem.available(["best", "good"]) == ["good"]
    st = client.gem._state("best")
    assert st.exhausted_until == gemini.next_quota_reset(client.clock.now())
    client.script["good"] = [_daily_quota()]
    with pytest.raises(QuotaExhausted):
        client.gem.generate(["best", "good"], [], Out)


# ---- work queue -------------------------------------------------------------------------


def _item(key: str, prio: int, eligible=lambda m: True) -> WorkItem:
    return WorkItem((prio,), key, eligible)


def test_queue_claims_highest_priority_eligible_item() -> None:
    q = WorkQueue([_item("low", 2), _item("only-best", 0, lambda m: m == "best"), _item("high", 1)])
    high, _ = q.claim("good")  # "only-best" ranks first but good can't improve it
    assert high.key == "high"
    only_best, _ = q.claim("best")
    assert only_best.key == "only-best"
    q.finish(high)
    low, _ = q.claim("good")
    assert low.key == "low"
    q.finish(low)
    q.release(only_best)  # best got overloaded and handed it back…
    assert q.claim("good") == (None, False)  # …but good still can't do it
    assert q.claim("best")[0].key == "only-best"


def test_queue_reports_claimed_items_that_may_come_back() -> None:
    q = WorkQueue([_item("a", 0)])
    item, _ = q.claim("best")
    assert q.claim("good") == (None, True)
    q.finish(item)
    assert q.claim("good") == (None, False)
    assert q.remaining() == 0


# ---- drivers ----------------------------------------------------------------------------


class FakeModels:
    """Model availability for driver tests: `cooling[m]` counts down on each readiness check."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cooling: dict[str, int] = {}
        self.exhausted: set[str] = set()

    def available(self, models):
        return [m for m in models if m not in self.exhausted]

    def ready(self, models):
        with self.lock:
            out = []
            for m in self.available(models):
                if self.cooling.get(m, 0) > 0:
                    self.cooling[m] -= 1
                else:
                    out.append(m)
            return out

    def ready_at(self, models):
        return time.time()


def test_drivers_hand_back_work_when_overloaded_and_resume_after() -> None:
    gem = FakeModels()
    q = WorkQueue([_item(f"v{i}", i) for i in range(6)])
    by: dict[str, str] = {}
    overloaded_once = threading.Event()

    def work(item: WorkItem, model: str) -> str:
        if model == "best" and not overloaded_once.is_set():
            overloaded_once.set()
            gem.cooling["best"] = 20  # cools down for a while
            raise Overloaded("busy", time.time())
        time.sleep(0.01)
        by[item.key] = model
        return model

    done = run_drivers(gem, ["best", "good"], q, work, sleep=lambda s: time.sleep(0.001))
    assert sorted(by) == [f"v{i}" for i in range(6)]  # every item done exactly once
    assert sum(done.values()) == 6
    assert by["v0"] == "good" or done["best"] > 0  # v0 handed back, then picked up
    assert done["best"] > 0  # best resumed after its cooldown


def test_driver_stops_when_its_model_is_out_of_quota() -> None:
    gem = FakeModels()
    q = WorkQueue([_item(f"v{i}", i) for i in range(3)])

    def work(item: WorkItem, model: str) -> str:
        if model == "best":
            gem.exhausted.add("best")
            raise QuotaExhausted("done for today")
        return model

    done = run_drivers(gem, ["best", "good"], q, work, sleep=lambda s: time.sleep(0.001))
    assert done == {"good": 3}


def test_drivers_respect_limit() -> None:
    q = WorkQueue([_item(f"v{i}", i) for i in range(10)])
    done = run_drivers(FakeModels(), ["best"], q, lambda i, m: m, limit=3)
    assert sum(done.values()) == 3 and q.remaining() == 7


def test_every_request_outcome_is_logged(client) -> None:
    client.script["best"] = [_overloaded(), _daily_quota()]
    client.gem.generate(["best", "good"], [], Out)  # best: 503 → good: ok
    client.clock.sleep(301)
    client.gem.generate(["best", "good"], [], Out)  # best: quota → good: ok
    rows = gemini.load_requests()
    assert [(r["model"], r["outcome"]) for r in rows] == [
        ("best", "http_503"),
        ("good", "ok"),
        ("best", "quota"),
        ("good", "ok"),
    ]
    assert gemini.quota_day(rows[0]["at"]) == "2027-01-15"
