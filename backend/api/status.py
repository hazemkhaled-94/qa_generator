"""System status, aggregated for the frontend."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sqlalchemy import inspect

from blob_store.s3 import (
    ArchiveBucket,
    Bucket,
    DocumentsBucket,
    ExportBucket,
    ModelsBucket,
    ParsedBucket,
    bucket_names,
)
from database.qa_generator import Base, sessions

log = logging.getLogger(__name__)

#: The buckets the application requires, as classes rather than names so the
#: check runs against the objects the services actually write to.
_REQUIRED_BUCKETS: tuple[type[Bucket], ...] = (
    DocumentsBucket,
    ParsedBucket,
    ModelsBucket,
    ExportBucket,
    ArchiveBucket,
)


@dataclass(frozen=True)
class Component:
    """The state of one thing the backend depends on."""

    ok: bool
    detail: str
    metrics: Mapping[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Counter:
    """A service's own numbers, for the status panel.

    A callable supplied by the composition root, not a query written here,
    so this module learns no table names.
    """

    detail: str
    counts: Callable[[], Mapping[str, int | None]]


class StatusService:
    """Reports whether the backend is healthy and what its services hold."""

    def __init__(self, counters: dict[str, Counter]) -> None:
        """Initialises the service."""
        self._counters = counters
        self._session = sessions()

    def snapshot(self) -> dict[str, Component]:
        """Collects the current state of every component.

        Never raises: an unreachable component is reported as unhealthy with
        the reason in its detail. The service counters are omitted when the
        database itself is down.
        """
        database = self._database()
        components = {"database": database, "object_store": self._object_store()}
        if database.ok:
            components |= {
                name: self._counter(name, counter)
                for name, counter in self._counters.items()
            }
        return components

    def _database(self) -> Component:
        """Checks the database and whether the schema is applied.

        Compares the live tables against every model the project declares,
        so no table is named here.
        """
        try:
            with self._session() as session:
                live = set(inspect(session.connection()).get_table_names())
        except Exception as exc:  # noqa: BLE001
            log.error("database unreachable: %s: %s", type(exc).__name__, exc)
            return Component(False, f"Unreachable: {type(exc).__name__}: {exc}")

        missing = set(Base.metadata.tables) - live
        if missing:
            return Component(
                False, f"Connected, but missing: {', '.join(sorted(missing))}"
            )
        return Component(True, "Connected, schema applied.")

    @staticmethod
    def _counter(name: str, counter: Counter) -> Component:
        """Asks one service for its numbers."""
        try:
            return Component(True, counter.detail, counter.counts())
        except Exception as exc:  # noqa: BLE001
            log.error("cannot count %s: %s: %s", name, type(exc).__name__, exc)
            return Component(False, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _object_store() -> Component:
        """Checks the object store and measures each required bucket."""
        try:
            present = set(bucket_names())
        except Exception as exc:  # noqa: BLE001
            log.error("object store unreachable: %s: %s", type(exc).__name__, exc)
            return Component(False, f"Unreachable: {type(exc).__name__}: {exc}")

        missing = {bucket.name for bucket in _REQUIRED_BUCKETS} - present
        if missing:
            return Component(False, f"missing: {', '.join(sorted(missing))}")

        metrics: dict[str, int | None] = {}
        for bucket in _REQUIRED_BUCKETS:
            try:
                metrics[bucket.name] = bucket().count()
            except Exception as exc:  # noqa: BLE001
                log.error("cannot measure bucket %s: %s", bucket.name, exc)
                metrics[bucket.name] = None
        return Component(True, ", ".join(sorted(present)), metrics)
