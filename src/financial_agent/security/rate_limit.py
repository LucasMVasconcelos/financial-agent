"""Per-user rate limiting.

Applied to the Telegram webhook keyed by the *authenticated* `user_id`
(never by anything the client claims outside that). Two backends implement
the same `RateLimiter` protocol:

  * `InMemoryRateLimiter` — fixed-window counter in a process-local dict.
    Fine for local dev/single-instance deployments; resets on restart and
    does not coordinate across replicas.
  * `RedisRateLimiter` — same fixed-window algorithm backed by Redis
    `INCR` + `EXPIRE`, so it works correctly across multiple API replicas.

Selection is controlled by `Settings.use_redis`.
"""

from __future__ import annotations

import time
from typing import Protocol

from financial_agent.domain.errors import RateLimitedAppError


class RateLimiter(Protocol):
    async def check(self, key: str) -> None:
        """Raise `RateLimitedAppError` if `key` has exceeded its quota."""
        ...


class InMemoryRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    async def check(self, key: str) -> None:
        now = time.monotonic()
        window_start = now - self._window_seconds
        hits = [t for t in self._hits.get(key, []) if t > window_start]
        if len(hits) >= self._max_requests:
            raise RateLimitedAppError(f"Rate limit exceeded for '{key}'.")
        hits.append(now)
        self._hits[key] = hits


class RedisRateLimiter:
    def __init__(self, *, redis_client: object, max_requests: int, window_seconds: int) -> None:
        self._redis = redis_client
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    async def check(self, key: str) -> None:
        bucket = f"ratelimit:{key}:{int(time.time()) // self._window_seconds}"
        current = await self._redis.incr(bucket)  # type: ignore[attr-defined]
        if current == 1:
            await self._redis.expire(bucket, self._window_seconds)  # type: ignore[attr-defined]
        if current > self._max_requests:
            raise RateLimitedAppError(f"Rate limit exceeded for '{key}'.")
