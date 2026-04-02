"""Unit tests for module-level state reset helpers."""

import pytest

from gateway.api.deps import get_config, reset_config, set_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db


def test_reset_config_clears_state() -> None:
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
    )
    set_config(config)
    assert get_config() is config

    reset_config()

    with pytest.raises(RuntimeError, match="Config not initialized"):
        get_config()


def test_reset_db_allows_reinit() -> None:
    from core import database

    reset_db()

    assert database._engine is None
    assert database._SessionLocal is None
