from __future__ import annotations

import pytest

from financial_agent.domain.errors import RateLimitedAppError
from financial_agent.security.rate_limit import InMemoryRateLimiter


class TestInMemoryRateLimiter:
    async def test_allows_requests_within_quota(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            await limiter.check("user:123")  # should not raise

    async def test_raises_rate_limited_once_quota_exceeded(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

        await limiter.check("user:123")
        await limiter.check("user:123")

        with pytest.raises(RateLimitedAppError):
            await limiter.check("user:123")

    async def test_quota_is_independent_per_key(self) -> None:
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)

        await limiter.check("user:123")
        await limiter.check("user:456")  # different key, should not raise
