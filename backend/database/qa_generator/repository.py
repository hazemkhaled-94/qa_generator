"""The base every repository shares."""

from __future__ import annotations

from database.qa_generator.engine import sessions


class Repository:
    """Holds the shared session factory, and nothing else."""

    def __init__(self) -> None:
        """Binds to the session factory."""
        self._session = sessions()
