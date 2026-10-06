"""One driver (worker thread) per model, all pulling from a shared priority queue.

Each driver only claims work its model can improve: anything missing, or results produced by
a lower-ranked model. When its model is overloaded it releases the claimed item back to the
queue (so another driver can take it) and sleeps until the model's cooldown ends. When its
model's daily quota is used up, the driver stops. So every model's quota is used in parallel,
and the best model always takes the highest-priority work it can get while it's healthy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .gemini import Gemini, Overloaded, QuotaExhausted

log = logging.getLogger(__name__)

# How long an idle driver waits before checking the queue again, and the longest single
# sleep while its model cools down (so it notices a stop request reasonably quickly).
IDLE_POLL = 5.0
MAX_SLEEP = 30.0


@dataclass(order=True)
class WorkItem:
    priority: tuple
    key: str = field(compare=False)
    # Whether a given model would improve this item (e.g. ranks above its current result).
    eligible: Callable[[str], bool] = field(compare=False, repr=False)
    payload: Any = field(default=None, compare=False, repr=False)


class WorkQueue:
    """Thread-safe priority queue where an item is claimed by one driver at a time."""

    def __init__(self, items: list[WorkItem]) -> None:
        self._items = sorted(items)
        self._claimed: set[str] = set()
        self._finished: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, model: str) -> tuple[WorkItem | None, bool]:
        """Claim the highest-priority unclaimed item `model` can improve. Returns
        (item, more_possible): `more_possible` is True when nothing is free right now but an
        item this model could do is claimed by another driver and may be released."""
        with self._lock:
            waiting = False
            for item in self._items:
                if item.key in self._finished or not item.eligible(model):
                    continue
                if item.key in self._claimed:
                    waiting = True
                    continue
                self._claimed.add(item.key)
                return item, True
            return None, waiting

    def release(self, item: WorkItem) -> None:
        """Give an item back unfinished (e.g. the model became overloaded)."""
        with self._lock:
            self._claimed.discard(item.key)

    def finish(self, item: WorkItem) -> None:
        with self._lock:
            self._claimed.discard(item.key)
            self._finished.add(item.key)

    def remaining(self) -> int:
        with self._lock:
            return sum(1 for i in self._items if i.key not in self._finished)


def run_drivers(
    gem: Gemini,
    models: list[str],
    queue: WorkQueue,
    work: Callable[[WorkItem, str], str | None],
    limit: int = 0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] | None = None,
) -> Counter[str]:
    """Run one driver thread per model until the queue is drained, every model is out of
    quota, or `limit` items are done. `work(item, model)` does one item using only `model`
    and returns a short description for the log. Returns items completed per model."""
    done: Counter[str] = Counter()
    lock = threading.Lock()
    stop = threading.Event()
    pause_for = sleep or stop.wait  # stop.wait wakes early when the run is stopped

    def driver(model: str) -> None:
        while not stop.is_set():
            if not gem.available([model]):
                log.info("[%s] out of daily quota; driver stopping", model)
                return
            if not gem.ready([model]):
                pause_for(min(max(gem.ready_at([model]) - clock(), 0.1), MAX_SLEEP))
                continue
            item, more = queue.claim(model)
            if item is None:
                if not more:
                    log.info("[%s] no more work this model can improve; driver stopping", model)
                    return
                pause_for(IDLE_POLL)
                continue
            try:
                note = work(item, model)
            except Overloaded:
                queue.release(item)  # cooldown is set; another driver may take the item
                continue
            except QuotaExhausted:
                queue.release(item)
                continue  # the availability check at the top stops this driver
            except Exception:
                log.exception("[%s] %s failed", model, item.key)
                queue.finish(item)  # don't retry within this run
                continue
            queue.finish(item)
            with lock:
                done[model] += 1
                total = sum(done.values())
            log.info(
                "[%s] done %s%s (%d this run)", model, item.key, f": {note}" if note else "", total
            )
            if limit and total >= limit:
                stop.set()

    threads = [threading.Thread(target=driver, args=(m,), name=f"driver-{m}") for m in models]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        log.warning("stopping after the current items finish…")
        for t in threads:
            t.join()
    return done
