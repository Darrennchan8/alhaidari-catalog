"""Gemini API client: rate limiting, retries and structured (pydantic) output."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from . import config

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)


class QuotaExhausted(RuntimeError):
    """Daily quota is used up; the caller should stop and resume later."""


class Gemini:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set (put it in .env at the repo root)")
        # Generous timeout: a 12-minute audio chunk can take a few minutes to translate.
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=900_000))
        self._lock = threading.Lock()
        self._local = threading.local()
        self._last = 0.0
        self._exhausted: dict[str, float] = _load_quota()
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + config.GEMINI_MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def generate(
        self,
        model: str,
        contents: list,
        schema: type[M],
        system: str | None = None,
        temperature: float = 0.2,
        thinking_level: str | None = None,
        max_attempts: int = 8,
        fallbacks: list[str] | None = None,
    ) -> M:
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
        # Fall back along the chain when a model is overloaded (503) or out of daily quota
        # (free-tier quotas are per model).
        chain = list(
            dict.fromkeys([model, *(config.FALLBACK_MODELS if fallbacks is None else fallbacks)])
        )
        overloaded: dict[str, int] = {}
        delay = 20.0
        for attempt in range(1, max_attempts + 1):
            current = self._pick(chain, overloaded)
            self._throttle()
            try:
                resp = self.client.models.generate_content(
                    model=current, contents=contents, config=cfg
                )
                self._account(resp)
                if not resp.text:
                    reason = resp.candidates[0].finish_reason if resp.candidates else "none"
                    raise ValueError(f"empty response (finish_reason={reason})")
                result = schema.model_validate_json(resp.text)
                self._local.model = current
                return result
            except errors.ClientError as e:
                if e.code == 429:
                    if _is_daily_quota(e):
                        self._mark_exhausted(current, e)
                        continue
                    log.warning("%s rate limited; sleeping %.0fs", current, delay)
                elif e.code in (400, 403, 404) and "thinking" in str(e).lower():
                    cfg.thinking_config = None  # model doesn't support thinking_level
                    continue
                elif e.code == 404 and len(chain) > 1:
                    # Retired/unknown model: skip it for the rest of this process.
                    log.warning("%s unavailable (404); falling back", current)
                    with self._lock:
                        self._exhausted[current] = float("inf")
                    continue
                elif e.code in (400, 401, 403, 404):
                    raise
                else:
                    log.warning("client error %s: %s", e.code, e)
            except errors.ServerError as e:
                if e.code == 503:
                    overloaded[current] = overloaded.get(current, 0) + 1
                    log.warning("%s overloaded (attempt %d)", current, attempt)
                    if overloaded[current] % 2 == 0:
                        continue  # move on to the next model right away
                else:
                    log.warning("server error %s (attempt %d): %s", e.code, attempt, e)
            except (ValidationError, ValueError) as e:
                log.warning("bad response (attempt %d): %s", attempt, str(e)[:300])
            if attempt == max_attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 600)
        raise RuntimeError(f"Gemini call failed after {max_attempts} attempts")

    @property
    def last_model(self) -> str | None:
        """The model that produced the most recent successful response in this thread."""
        return getattr(self._local, "model", None)

    def _pick(self, chain: list[str], overloaded: dict[str, int]) -> str:
        now = time.time()
        with self._lock:
            available = [m for m in chain if self._exhausted.get(m, 0) <= now]
        if not available:
            raise QuotaExhausted(
                "daily quota used up for every model in the chain: " + ", ".join(chain)
            )
        # Prefer the earliest model in the chain that has been overloaded least in this call.
        return min(available, key=lambda m: (overloaded.get(m, 0) // 2, available.index(m)))

    def _mark_exhausted(self, model: str, err: errors.ClientError) -> None:
        retry = re.search(r"retryDelay'?:\s*'(\d+)s", str(err))
        until = time.time() + (int(retry.group(1)) if retry else 6 * 3600)
        with self._lock:
            self._exhausted[model] = until
            _save_quota(self._exhausted)
        log.warning(
            "%s daily quota exhausted until %s; falling back",
            model,
            time.strftime("%Y-%m-%d %H:%M", time.localtime(until)),
        )

    def _account(self, resp: types.GenerateContentResponse) -> None:
        self.usage["requests"] += 1
        um = resp.usage_metadata
        if um:
            self.usage["input_tokens"] += um.prompt_token_count or 0
            self.usage["output_tokens"] += (um.candidates_token_count or 0) + (
                um.thoughts_token_count or 0
            )


QUOTA_FILE = config.RAW / "quota.json"


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
    QUOTA_FILE.write_text(json.dumps({m: t for m, t in exhausted.items() if t != float("inf")}))
