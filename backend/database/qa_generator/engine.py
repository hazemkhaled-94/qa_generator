"""Connection to the qa_generator database.

The only module that builds an engine or knows where the database lives.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def _pool(name: str, fallback: int) -> int:
    """Reads a pool size, falling back so a worker needs no new setting."""
    value = os.environ.get(name, "").strip()
    return int(value) if value else fallback


@lru_cache(maxsize=1)
def engine() -> Engine:
    """Builds the engine, once per process."""
    return create_engine(
        os.environ["DATABASE_URL"],
        # pool_pre_ping: a connection dropped by a restarted database is
        # replaced rather than surfacing as an error in the caller.
        pool_pre_ping=True,
        pool_size=_pool("DATABASE_POOL_SIZE", 5),
        max_overflow=_pool("DATABASE_POOL_OVERFLOW", 5),
        pool_recycle=1800,
    )


@lru_cache(maxsize=1)
def sessions() -> sessionmaker:
    """Builds the session factory, once per process."""
    # expire_on_commit off: callers read values out of a closing session.
    return sessionmaker(engine(), expire_on_commit=False)
