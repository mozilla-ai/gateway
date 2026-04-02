"""Unit tests for rate limiter core behavior and config validation."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.rate_limit import RateLimiter, RateLimitInfo


def test_rate_limiter_allows_under_limit() -> None:
    limiter = RateLimiter(rpm=5)
    for _ in range(5):
        limiter.check("user-1")


def test_rate_limiter_returns_rate_limit_info() -> None:
    limiter = RateLimiter(rpm=5)
    info = limiter.check("user-1")
    assert isinstance(info, RateLimitInfo)
    assert info.limit == 5
    assert info.remaining == 4

    info2 = limiter.check("user-1")
    assert info2.remaining == 3


def test_rate_limiter_reset_is_wall_clock() -> None:
    limiter = RateLimiter(rpm=5)

    with patch("gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0
        info = limiter.check("user-1")

    assert info.reset == pytest.approx(1700000060.0, abs=1.0)


def test_rate_limiter_rejects_over_limit() -> None:
    limiter = RateLimiter(rpm=3)
    for _ in range(3):
        limiter.check("user-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user-1")
    assert exc_info.value.status_code == 429


def test_rate_limiter_429_includes_retry_after() -> None:
    limiter = RateLimiter(rpm=1)
    limiter.check("user-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user-1")
    assert exc_info.value.headers is not None
    assert "Retry-After" in exc_info.value.headers
    retry_after = int(exc_info.value.headers["Retry-After"])
    assert 1 <= retry_after <= 60


def test_rate_limiter_per_user_isolation() -> None:
    limiter = RateLimiter(rpm=2)
    limiter.check("user-1")
    limiter.check("user-1")
    limiter.check("user-2")
    limiter.check("user-2")

    with pytest.raises(HTTPException):
        limiter.check("user-1")
    with pytest.raises(HTTPException):
        limiter.check("user-2")


def test_rate_limiter_window_expiry() -> None:
    limiter = RateLimiter(rpm=2)

    with patch("gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0
        limiter.check("user-1")
        limiter.check("user-1")

        with pytest.raises(HTTPException):
            limiter.check("user-1")

        mock_time.monotonic.return_value = 1061.0
        mock_time.time.return_value = 1700000061.0
        info = limiter.check("user-1")
        assert info.remaining == 1


def test_rate_limiter_cleanup() -> None:
    limiter = RateLimiter(rpm=100)
    limiter._CLEANUP_INTERVAL = 3

    with patch("gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0

        limiter.check("stale-user")
        limiter.check("stale-user")

        mock_time.monotonic.return_value = 1061.0
        mock_time.time.return_value = 1700000061.0
        limiter.check("active-user")

        assert "stale-user" not in limiter._requests
        assert "active-user" in limiter._requests


def test_config_rejects_zero_rate_limit() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(rate_limit_rpm=0)


def test_config_rejects_negative_rate_limit() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(rate_limit_rpm=-1)


def test_config_accepts_positive_rate_limit() -> None:
    config = GatewayConfig(rate_limit_rpm=1)
    assert config.rate_limit_rpm == 1


def test_config_accepts_none_rate_limit() -> None:
    config = GatewayConfig(rate_limit_rpm=None)
    assert config.rate_limit_rpm is None
