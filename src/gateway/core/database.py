"""Async database initialization helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from gateway.core.config import GatewayConfig

_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None
_SYNC_DATABASE_URL: str | None = None


def _to_async_url(database_url: str) -> tuple[str, dict[str, Any]]:
    """Convert a sync SQLAlchemy URL into its async equivalent."""

    url: URL = make_url(database_url)
    connect_args: dict[str, Any] = {}
    drivername = url.drivername

    if drivername.startswith("sqlite"):
        async_url = url.set(drivername="sqlite+aiosqlite")
        connect_args["check_same_thread"] = False
        return async_url.render_as_string(hide_password=False), connect_args

    if drivername in {"postgresql", "postgresql+psycopg2"}:
        query = dict(url.query)
        sslmode = query.pop("sslmode", None)
        async_url = url.set(drivername="postgresql+asyncpg", query=query)
        if sslmode:
            connect_args["ssl"] = sslmode
        return async_url.render_as_string(hide_password=False), connect_args

    if drivername == "postgresql+asyncpg":
        return database_url, connect_args

    return database_url, connect_args


def _run_migrations(database_url: str) -> None:
    alembic_cfg = Config()
    alembic_dir = Path(__file__).resolve().parents[3] / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.attributes["configure_logger"] = False
    command.upgrade(alembic_cfg, "head")


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN001, ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db(config: GatewayConfig) -> None:
    """Initialize async database engine and optionally run migrations."""

    global _engine, _SessionLocal, _SYNC_DATABASE_URL  # noqa: PLW0603

    database_url = config.database_url
    async_url, connect_args = _to_async_url(database_url)

    is_sqlite = async_url.startswith("sqlite+aiosqlite")

    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True, "connect_args": connect_args}
    if is_sqlite:
        engine_kwargs["poolclass"] = NullPool
    else:
        engine_kwargs["pool_size"] = config.db_pool_size
        engine_kwargs["max_overflow"] = config.db_max_overflow
        engine_kwargs["pool_timeout"] = config.db_pool_timeout
        if config.db_pool_recycle >= 0:
            engine_kwargs["pool_recycle"] = config.db_pool_recycle

    _engine = create_async_engine(async_url, **engine_kwargs)
    _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)
    _SYNC_DATABASE_URL = database_url

    if is_sqlite:
        _enable_sqlite_foreign_keys(_engine)

    if config.auto_migrate:
        _run_migrations(database_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession."""

    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _SessionLocal() as session:
        yield session


@asynccontextmanager
async def create_session() -> AsyncIterator[AsyncSession]:
    """Async context manager for creating sessions outside request scope."""

    if _SessionLocal is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _SessionLocal() as session:
        yield session


def reset_db() -> None:
    """Dispose the active engine so it can be re-initialized (testing helper)."""

    global _engine, _SessionLocal, _SYNC_DATABASE_URL  # noqa: PLW0603

    engine = _engine
    _engine = None
    _SessionLocal = None
    _SYNC_DATABASE_URL = None

    if engine is None:
        return

    dispose_coro = engine.dispose()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(dispose_coro)
    else:
        loop.create_task(dispose_coro)


__all__ = [
    "create_session",
    "get_db",
    "init_db",
    "reset_db",
]
