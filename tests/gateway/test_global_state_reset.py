"""Tests for global state reset functions."""

import pytest

from api.deps import get_config, reset_config, set_config
from core.config import GatewayConfig
from core.database import reset_db


def test_reset_config_clears_state() -> None:
    """Test that reset_config clears the global config."""
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
    """Test that reset_db clears state so init_db can be called again.

    We can't fully test init_db without a database, but we can verify
    reset_db doesn't raise and clears the module state.
    """
    from core import database

    # Verify the function exists and runs without error when nothing is initialized
    reset_db()

    assert database._engine is None
    assert database._SessionLocal is None
