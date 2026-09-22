"""The archive command line.

The second deletion, and the only one with nothing behind it. What this
covers is the guard: a purge names what it purges, because nothing named
is the shape of a typo rather than of a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from archive.run import _sized
from archive.store import Held, Purged


@dataclass
class Store:
    """An archive that records what was purged."""

    held: list[Held]
    objects_held: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        """Starts with nothing purged."""
        self.purges: list[dict] = []

    def rows(self) -> list[Held]:
        """What is held, by table."""
        return self.held

    def objects(self) -> tuple[int, int]:
        """What the bucket holds."""
        return self.objects_held

    def purge(self, *, table=None, older_than_days=None) -> Purged:
        """Records one purge."""
        self.purges.append({"table": table, "older_than_days": older_than_days})
        return Purged(rows=4, objects=2)


def held(table: str = "questions", rows: int = 3) -> Held:
    """One table's entry in the archive."""
    when = datetime(2026, 9, 1, tzinfo=UTC)
    return Held(table_name=table, rows=rows, bytes=2048, oldest=when, newest=when)


class Driver:
    """The command line with its store replaced."""

    def __init__(self, monkeypatch, store: Store) -> None:
        """Points the command at a recording store."""
        from archive import run as module

        self.store = store
        monkeypatch.setattr(module.telemetry, "configure", lambda *_, **__: None)
        monkeypatch.setattr(module.telemetry, "trace_engine", lambda *_: None)
        monkeypatch.setattr(module, "engine", lambda: None)
        monkeypatch.setattr(module, "Archive", lambda: store)
        self._main = module.main

    def run(self, *argv: str) -> int:
        """Runs the command line."""
        return self._main(list(argv))


@pytest.fixture
def cli(monkeypatch) -> Driver:
    """The command line over an archive holding one table."""
    return Driver(monkeypatch, Store([held()], objects_held=(2, 4096)))


@pytest.fixture
def empty(monkeypatch) -> Driver:
    """The command line over an archive holding nothing."""
    return Driver(monkeypatch, Store([]))


def test_status_reports_what_is_held_without_purging(cli) -> None:
    """The cheap flag, and the one that is safe to run."""
    assert cli.run("--status") == 0
    assert cli.store.purges == []


def test_status_over_an_empty_archive_is_not_an_error(empty) -> None:
    """Nothing held is the ordinary state after a purge."""
    assert empty.run("--status") == 0


def test_a_purge_naming_nothing_is_refused(cli) -> None:
    """This is the deletion the archive was protecting against."""
    assert cli.run("--purge") == 2
    assert cli.store.purges == []


def test_a_purge_of_one_table_leaves_the_objects_alone(cli) -> None:
    """An archived upload belongs to no table."""
    assert cli.run("--purge", "--table", "questions") == 0
    assert cli.store.purges == [{"table": "questions", "older_than_days": None}]


def test_a_purge_by_age_is_measured_in_days(cli) -> None:
    """Each half measures it by the clock that wrote it."""
    assert cli.run("--purge", "--older-than", "30") == 0
    assert cli.store.purges == [{"table": None, "older_than_days": 30}]


def test_a_purge_of_everything_names_neither(cli) -> None:
    """`--all` is what says the whole archive was meant."""
    assert cli.run("--purge", "--all") == 0
    assert cli.store.purges == [{"table": None, "older_than_days": None}]


def test_an_action_is_required(cli) -> None:
    """A bare run would otherwise do nothing and exit 0."""
    with pytest.raises(SystemExit):
        cli.run()


def test_status_and_purge_together_are_refused(cli) -> None:
    """They are mutually exclusive."""
    with pytest.raises(SystemExit):
        cli.run("--status", "--purge")


def test_an_age_that_is_not_a_number_is_refused(cli) -> None:
    """The parser types it, so the store never sees a string."""
    with pytest.raises(SystemExit):
        cli.run("--purge", "--older-than", "a month")


# ── What a person reads ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("count", "written"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (2048, "2 kB"),
        (5 * 1024 * 1024, "5 MB"),
        (3 * 1024**3, "3.0 GB"),
    ],
)
def test_a_byte_count_is_shown_in_the_largest_unit_above_one(
    count: int, written: str
) -> None:
    """A row saying 3,221,225,472 B is a row nobody reads."""
    assert _sized(count) == written
