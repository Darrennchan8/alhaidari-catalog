"""Gemini API client: per-model quota, backoff and pacing; retries; structured output.

Each model has its own state, shared by every thread using the client:

- daily quota: a model that returns a per-day 429 is skipped until the quota resets
  (midnight Pacific), and that is persisted in data/raw/quota.json across runs;
- overload backoff: each consecutive 503 puts the model in a cooldown that doubles
  (5, 10, 20 min, then every 30 min). Every caller skips a cooling model; the first request
  after the cooldown is the probe, and a success resets it. The backoff is deliberately slow:
  503s appear to count against the free tier's daily request quota;
- pacing: requests to the same model are spaced at least CATALOG_GEMINI_MIN_INTERVAL apart
  (free-tier per-minute limits are per model).

Every request's outcome is appended to data/raw/requests.jsonl; `catalog status` summarises
it per model and (Pacific) quota day.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import TypeVar
from zoneinfo import ZoneInfo

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from . import config

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)

COOLDOWN_BASE = float(os.getenv("CATALOG_COOLDOWN_BASE", "300"))
COOLDOWN_MAX = float(os.getenv("CATALOG_COOLDOWN_MAX", "1800"))
# A pooled call (one that may use several models) waits at most this long for a cooling
# model before giving up with Overloaded.
MAX_COOLDOWN_WAIT = float(os.getenv("CATALOG_MAX_COOLDOWN_WAIT", "3600"))


class QuotaExhausted(RuntimeError):
    """Every allowed model is out of daily quota; resume after the reset."""


class Overloaded(RuntimeError):
    """Every allowed model with quota is cooling down after 503s."""

    def __init__(self, message: str, retry_at: float) -> None:
        super().__init__(message)
        self.retry_at = retry_at


@dataclass
class ModelState:
    exhausted_until: float = 0.0  # daily quota used up until this time
    cooldown_until: float = 0.0  # overloaded: skip until this time
    overloads: int = 0  # consecutive 503s
    last_request: float = 0.0


class Gemini:
    def __init__(
        self,
        api_key: str | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set (put it in .env at the repo root)")
        # Generous timeout: a 12-minute audio chunk can take a few minutes to translate.
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=900_000))
        self._now = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._local = threading.local()
        self._states: dict[str, ModelState] = {}
        for model, until in _load_quota().items():
            self._state(model).exhausted_until = until
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    # ---- model state -------------------------------------------------------------------------

    def _state(self, model: str) -> ModelState:
        return self._states.setdefault(model, ModelState())

    def available(self, models: list[str]) -> list[str]:
        """Models (in order) that still have daily quota, cooling or not."""
        now = self._now()
        with self._lock:
            return [m for m in models if self._state(m).exhausted_until <= now]

    def ready(self, models: list[str]) -> list[str]:
        """Models (in order) with quota that are not cooling down."""
        now = self._now()
        with self._lock:
            return [
                m
                for m in models
                if self._state(m).exhausted_until <= now and self._state(m).cooldown_until <= now
            ]

    def ready_at(self, models: list[str]) -> float:
        """When the earliest of `models` with quota stops cooling down (now if one is ready)."""
        now = self._now()
        with self._lock:
            times = [
                max(now, self._state(m).cooldown_until)
                for m in models
                if self._state(m).exhausted_until <= now
            ]
        return min(times, default=now)

    def _overloaded(self, model: str) -> None:
        with self._lock:
            st = self._state(model)
            st.overloads += 1
            wait = min(COOLDOWN_BASE * 2 ** (st.overloads - 1), COOLDOWN_MAX)
            st.cooldown_until = self._now() + wait
            n = st.overloads
        log.warning("%s overloaded (%d in a row); cooling down %.0fs", model, n, wait)

    def _succeeded(self, model: str) -> None:
        with self._lock:
            st = self._state(model)
            recovered = st.overloads > 0
            st.overloads = 0
            st.cooldown_until = 0.0
        if recovered:
            log.info("%s is responding again", model)

    def _mark_exhausted(self, model: str) -> None:
        until = next_quota_reset(self._now())
        with self._lock:
            self._state(model).exhausted_until = until
            _save_quota(
                {m: s.exhausted_until for m, s in self._states.items() if s.exhausted_until}
            )
        log.warning(
            "%s daily quota exhausted until %s",
            model,
            time.strftime("%Y-%m-%d %H:%M", time.localtime(until)),
        )

    def _pace(self, model: str) -> None:
        """Space requests to the same model at least GEMINI_MIN_INTERVAL apart."""
        with self._lock:
            st = self._state(model)
            start = max(self._now(), st.last_request + config.GEMINI_MIN_INTERVAL)
            st.last_request = start
        wait = start - self._now()
        if wait > 0:
            self._sleep(wait)

    # ---- requests ----------------------------------------------------------------------------

    def _pick(self, chain: list[str], wait: bool, waited: float) -> tuple[str, float]:
        """Choose the best ready model, waiting for a cooldown if allowed.
        Returns (model, total seconds waited so far)."""
        while True:
            ready = self.ready(chain)
            if ready:
                return ready[0], waited
            if not self.available(chain):
                raise QuotaExhausted("daily quota used up for: " + ", ".join(chain))
            retry_at = self.ready_at(chain)
            pause = retry_at - self._now()
            if not wait or waited + pause > MAX_COOLDOWN_WAIT:
                raise Overloaded("cooling down: " + ", ".join(self.available(chain)), retry_at)
            self._sleep(max(pause, 0.1))
            waited += pause

    def generate(
        self,
        models: list[str],
        contents: list,
        schema: type[M],
        system: str | None = None,
        temperature: float = 0.2,
        thinking_level: str | None = None,
        max_attempts: int = 6,
        wait: bool = True,
    ) -> M:
        """Call the best ready model in `models` (best first). A model that is overloaded
        goes into cooldown and the call moves on to the next ready one.

        Raises QuotaExhausted when every model is out of quota, and Overloaded when the rest
        are all cooling down and `wait` is False (or waiting would exceed MAX_COOLDOWN_WAIT).
        Other failures (5xx, malformed output) are retried up to `max_attempts` times."""
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=65536,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        level = thinking_level or os.getenv("CATALOG_THINKING_LEVEL", "low")
        if level != "default":
            cfg.thinking_config = types.ThinkingConfig(thinking_level=level)
        chain = list(dict.fromkeys(models))
        if not chain:
            raise ValueError("no models to call")
        failures = 0
        waited = 0.0
        delay = 20.0
        while True:
            current, waited = self._pick(chain, wait, waited)
            self._pace(current)
            try:
                resp = self.client.models.generate_content(
                    model=current, contents=contents, config=cfg
                )
                self._account(resp)
                if not resp.text:
                    reason = resp.candidates[0].finish_reason if resp.candidates else "none"
                    raise ValueError(f"empty response (finish_reason={reason})")
                result = schema.model_validate_json(resp.text)
                self._log_request(current, "ok")
                self._succeeded(current)
                self._local.model = current
                return result
            except errors.ClientError as e:
                self._log_request(
                    current,
                    "quota" if e.code == 429 and _is_daily_quota(e) else f"http_{e.code}",
                )
                if e.code == 429 and _is_daily_quota(e):
                    self._mark_exhausted(current)
                    continue
                if e.code == 429:
                    self._overloaded(current)  # per-minute limit: back off this model
                    continue
                if e.code in (400, 403, 404) and "thinking" in str(e).lower():
                    cfg.thinking_config = None  # model doesn't support thinking_level
                    continue
                if e.code == 404:
                    # Retired/unknown model: skip it for the rest of this process.
                    log.warning("%s unavailable (404)", current)
                    with self._lock:
                        self._state(current).exhausted_until = float("inf")
                    continue
                if e.code in (400, 401, 403):
                    raise
                log.warning("%s client error %s: %s", current, e.code, e)
            except errors.ServerError as e:
                self._log_request(current, f"http_{e.code}")
                if e.code == 503:
                    self._overloaded(current)
                    continue
                log.warning("%s server error %s: %s", current, e.code, e)
            except (ValidationError, ValueError) as e:
                self._log_request(current, "bad_response")
                log.warning("%s bad response: %s", current, str(e)[:300])
            failures += 1
            if failures >= max_attempts:
                raise RuntimeError(f"Gemini call failed {failures} times")
            self._sleep(delay)
            delay = min(delay * 2, 600)

    def _log_request(self, model: str, outcome: str) -> None:
        line = json.dumps({"at": self._now(), "model": model, "outcome": outcome})
        with self._lock:
            REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with REQUESTS_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @property
    def last_model(self) -> str | None:
        """The model that produced the most recent successful response in this thread."""
        return getattr(self._local, "model", None)

    def _account(self, resp: types.GenerateContentResponse) -> None:
        with self._lock:
            self.usage["requests"] += 1
            um = resp.usage_metadata
            if um:
                self.usage["input_tokens"] += um.prompt_token_count or 0
                self.usage["output_tokens"] += (um.candidates_token_count or 0) + (
                    um.thoughts_token_count or 0
                )


QUOTA_FILE = config.RAW / "quota.json"
REQUESTS_FILE = config.RAW / "requests.jsonl"
QUOTA_TZ = ZoneInfo("America/Los_Angeles")


def next_quota_reset(now: float) -> float:
    """Daily request quotas reset at midnight Pacific time. (The 429's retryDelay is not
    reliable for this: observed values pointed at midnight UTC, ~17 hours too late.)"""
    local = datetime.fromtimestamp(now, QUOTA_TZ)
    midnight = datetime.combine(local.date() + timedelta(days=1), dt_time(), tzinfo=QUOTA_TZ)
    return midnight.timestamp()


def _is_daily_quota(err: errors.ClientError) -> bool:
    msg = str(err)
    return "PerDay" in msg or "per day" in msg.lower()


def _load_quota() -> dict[str, float]:
    try:
        data = json.loads(QUOTA_FILE.read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {m: t for m, t in data.items() if t > now}


def _save_quota(exhausted: dict[str, float]) -> None:
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    QUOTA_FILE.write_text(
        json.dumps({m: t for m, t in exhausted.items() if t != float("inf") and t > now})
    )


def quota_day(ts: float) -> str:
    """The Pacific calendar day a request counts against (YYYY-MM-DD)."""
    return datetime.fromtimestamp(ts, QUOTA_TZ).date().isoformat()


def load_requests() -> list[dict]:
    if not REQUESTS_FILE.exists():
        return []
    with REQUESTS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
