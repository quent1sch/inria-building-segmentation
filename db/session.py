"""
db/session.py

Async SQLAlchemy engine and session factory.

Uses async engine throughout so database I/O doesn't block the FastAPI
event loop. SQLite uses aiosqlite driver, PostgreSQL uses asyncpg.

Both are configured from DATABASE_URL env var:
  sqlite+aiosqlite:///jobs.db (local default)
  postgresql+asyncpg://user:pass@host/db  (cloud)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings
from db.models import Base

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        # SQLite needs check_same_thread=False for async use
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            connect_args=connect_args,
        )
    return _engine


def _get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


async def init_db() -> None:
    """Create all tables. Called once at application startup."""
    async with _get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose engine. Called at application shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for a database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(...)

    Commits on clean exit, rolls back on exception.
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency version of get_session().

    Usage in route:
        async def my_route(db: AsyncSession = Depends(db_session_dependency)):
            ...
    """
    async with get_session() as session:
        yield session