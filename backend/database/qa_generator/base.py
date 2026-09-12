"""Declarative base for the qa_generator database."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for every table in the qa_generator database.

    Carries ``metadata``, which holds every model imported so far.
    """
