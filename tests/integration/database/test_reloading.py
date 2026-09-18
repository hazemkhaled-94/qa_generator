"""A worker picks up a changed setting without being restarted.

The five workers are long-running processes that used to read their settings
once.

The check sits between drains, where nothing is claimed, so a change made
during a long run is picked up when that run finishes.
"""

from __future__ import annotations

import pytest

from stages.cli import reloading

pytestmark = pytest.mark.integration


class Spy:
    """A service that records the setting it was built from."""

    def __init__(self, value: str, drained: list[str]) -> None:
        """Holds the value it was built with, and where to report drains."""
        self.value = value
        self._drained = drained

    def drain(self, stopping=None) -> int:
        """Records that this service worked the queue, and what it holds."""
        del stopping
        self._drained.append(self.value)
        return 0


@pytest.fixture
def store(database):
    """The store, pointed at the container."""
    from settings.store import Settings

    return Settings()


def _building(store, drained: list[str]):
    """A builder that reads TOPIC_PASSES the way a real one reads settings."""

    def build() -> Spy:
        """Builds a service from the settings as they stand."""
        return Spy(store.resolved()["TOPIC_PASSES"], drained)

    return build


def _never() -> bool:
    """No stop has been asked for."""
    return False


def test_the_first_drain_builds_the_service(store) -> None:
    """Nothing is built until there is a queue to work."""
    drained: list[str] = []
    drain = reloading("topics", _building(store, drained))

    drain(_never)

    assert drained == [store.resolved()["TOPIC_PASSES"]]


def test_a_setting_written_between_drains_is_picked_up(store) -> None:
    """What a restart used to be needed for."""
    drained: list[str] = []
    drain = reloading("topics", _building(store, drained))

    drain(_never)
    store.write("TOPIC_PASSES", "42")
    drain(_never)

    assert drained[-1] == "42"


def test_the_service_is_built_again_only_when_something_changed(store) -> None:
    """A rebuild costs 2.2 GB of embedding weights in one of the five stages.

    So the version is compared first, and it is a digest of the stored
    overrides: one small query per poll, and a rebuild only when a value
    actually moved.
    """
    built = []

    def build():
        """Counts how many services were built."""
        built.append(object())
        return Spy("unused", [])

    drain = reloading("topics", build)

    drain(_never)
    drain(_never)
    drain(_never)

    assert len(built) == 1


def test_a_setting_returned_to_the_files_is_picked_up_too(store) -> None:
    """Deleting an override is a change like any other.

    The version is a digest rather than a counter for exactly this: a
    counter would not fall back when a row was deleted, so a worker would
    keep the value somebody had just withdrawn.
    """
    import os

    drained: list[str] = []
    store.write("TOPIC_PASSES", "42")
    drain = reloading("topics", _building(store, drained))

    drain(_never)
    store.clear("TOPIC_PASSES")
    drain(_never)

    assert drained == ["42", os.environ["TOPIC_PASSES"]]


def test_a_change_does_not_interrupt_a_drain_in_progress(store) -> None:
    """A row is worked under the settings the drain was built with.

    A rebuild part-way through would leave one topic written half under each
    mix, which is worse than picking the change up a moment later.
    """
    seen: list[str] = []

    class Slow:
        """A service that changes the settings while it is draining."""

        def __init__(self, value: str) -> None:
            """Holds the value it was built with."""
            self.value = value

        def drain(self, stopping=None) -> int:
            """Works a row, and writes a setting while it is working."""
            del stopping
            seen.append(self.value)
            store.write("TOPIC_PASSES", "99")
            seen.append(self.value)
            return 0

    drain = reloading("topics", lambda: Slow(store.resolved()["TOPIC_PASSES"]))
    drain(_never)

    assert seen[0] == seen[1]

    drain(_never)
    assert seen[-1] == "99"
