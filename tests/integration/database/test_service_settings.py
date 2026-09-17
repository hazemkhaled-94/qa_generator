"""The settings a deployment changed, and what a stage then reads.

The store is a trust boundary twice over: what it writes decides how every
stage behaves on its next row, and what it refuses is the only thing between
a typed value and a worker that will not start.

The environment is what these run under - tests/conftest.py puts
configs/env/backend.env in it - so `resolved` here is that file overlaid with
whatever a test stored.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def store(database):
    """The store, pointed at the container."""
    from settings.store import Settings

    return Settings()


def test_nothing_is_stored_on_a_deployment_nobody_configured(store) -> None:
    """An empty table means the files are being obeyed exactly."""
    assert store.overrides() == {}


def test_an_unconfigured_setting_reads_as_the_environment_says(store) -> None:
    """The store overrides a setting; it does not supply one."""
    import os

    resolved = store.resolved()

    assert resolved["TOPIC_PASSES"] == os.environ["TOPIC_PASSES"]


def test_a_stored_setting_is_what_resolves(store) -> None:
    """What a worker will read on its next row."""
    store.write("TOPIC_PASSES", "42")

    assert store.resolved()["TOPIC_PASSES"] == "42"
    assert store.overrides() == {"TOPIC_PASSES": "42"}


def test_a_stored_setting_reaches_the_stage_that_reads_it(store) -> None:
    """The whole point, end to end: a value stored becomes a loaded setting."""
    from topic_modelling.config import Settings

    store.write("TOPIC_PASSES", "42")

    assert Settings.load(store.resolved()).passes == 42


def test_a_platform_setting_reaches_a_stage_that_is_not_the_platform(
    store,
) -> None:
    """Resolution ignores the service column, or extraction misses LLM_MODEL.

    The one bug a per-service overlay would have: extraction reads its own
    settings and the platform's, and only the second is stored under
    `platform`.
    """
    from llm.config import Settings

    store.write("LLM_MODEL", "ollama_chat/stored")

    assert store.resolved()["LLM_MODEL"] == "ollama_chat/stored"
    assert Settings.load(store.resolved()).model == "ollama_chat/stored"


def test_writing_a_setting_twice_leaves_one_row(store) -> None:
    """Two rows for one setting would make row order decide the value."""
    store.write("TOPIC_PASSES", "42")
    store.write("TOPIC_PASSES", "7")

    assert store.overrides() == {"TOPIC_PASSES": "7"}


def test_clearing_a_setting_returns_it_to_the_environment(store) -> None:
    """The only way back: there is no stored copy of a default to restore."""
    import os

    store.write("TOPIC_PASSES", "42")

    assert store.clear("TOPIC_PASSES") is True
    assert store.overrides() == {}
    assert store.resolved()["TOPIC_PASSES"] == os.environ["TOPIC_PASSES"]


def test_clearing_a_setting_nobody_changed_says_so(store) -> None:
    """Nothing to undo is not a failure, and is worth telling a caller."""
    assert store.clear("TOPIC_PASSES") is False


def test_a_setting_nothing_configures_is_refused(store) -> None:
    """The catalogue is the list, so a typo cannot be stored and forgotten."""
    with pytest.raises(KeyError, match="TOPIC_PASSESS"):
        store.write("TOPIC_PASSESS", "42")


def test_a_setting_the_deployment_owns_is_refused(store) -> None:
    """A pool size is read when a process starts; storing one would do nothing."""
    with pytest.raises(ValueError, match="DATABASE_POOL_SIZE"):
        store.write("DATABASE_POOL_SIZE", "50")


def test_an_empty_value_is_refused_for_a_setting_that_needs_one(store) -> None:
    """Empty is what `required` refuses, so storing one stops the stage."""
    with pytest.raises(ValueError, match="TOPIC_PASSES"):
        store.write("TOPIC_PASSES", "")


def test_an_empty_value_turns_off_a_setting_whose_absence_means_something(
    store,
) -> None:
    """How a verifier model named in the file is overridden back to none.

    Deleting the row would return it to the file, which names one. Absence
    has to be storable or the file's value could never be turned off.
    """
    from question_generation.config import Settings

    store.write("QUESTIONS_VERIFIER_MODEL", "")

    assert store.overrides() == {"QUESTIONS_VERIFIER_MODEL": ""}
    assert Settings.load(store.resolved()).verifier_model is None


def test_a_value_is_trimmed_before_it_is_stored(store) -> None:
    """Surrounding whitespace is not part of a setting."""
    store.write("TOPIC_PASSES", "  42  ")

    assert store.overrides() == {"TOPIC_PASSES": "42"}


def test_a_setting_is_stored_under_the_service_that_configures_it(
    store, engine
) -> None:
    """The column a page selects on, taken from the catalogue not the caller."""
    from sqlalchemy import text

    store.write("TOPIC_PASSES", "42")

    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT service, name FROM service_settings")
        ).all()

    assert rows == [("topics", "TOPIC_PASSES")]


def test_the_version_of_an_unconfigured_deployment_names_the_environment(
    store,
) -> None:
    """A row produced under no overrides still records what produced it."""
    from settings.store import UNCHANGED

    assert store.version() == UNCHANGED


def test_the_version_changes_when_a_setting_changes(store) -> None:
    """What a row records, and what a write refuses to land on top of."""
    before = store.version()
    store.write("TOPIC_PASSES", "42")

    assert store.version() != before


def test_the_version_returns_when_a_setting_is_put_back(store) -> None:
    """A digest names a configuration by its content, so one name per content."""
    before = store.version()
    store.write("TOPIC_PASSES", "42")
    store.clear("TOPIC_PASSES")

    assert store.version() == before


def test_the_version_falls_back_when_an_override_is_deleted(store) -> None:
    """What a counter got wrong, named as a test.

    Two settings stored and the later one deleted: a counter would answer
    with the revision of the one left, which describes a configuration that
    had both. A digest answers with the content there is now.
    """
    store.write("TOPIC_PASSES", "42")
    one = store.version()
    store.write("TOPIC_TOP_TERMS", "5")
    store.clear("TOPIC_TOP_TERMS")

    assert store.version() == one


def test_the_version_does_not_depend_on_the_order_rows_were_written(
    store,
) -> None:
    """Two deployments configured the same way have the same version."""
    store.write("TOPIC_PASSES", "42")
    store.write("TOPIC_TOP_TERMS", "5")
    one = store.version()

    store.clear("TOPIC_PASSES")
    store.clear("TOPIC_TOP_TERMS")
    store.write("TOPIC_TOP_TERMS", "5")
    store.write("TOPIC_PASSES", "42")

    assert store.version() == one


def test_when_a_setting_was_changed_is_recorded(store) -> None:
    """What a page shows beside a setting the file no longer decides."""
    store.write("TOPIC_PASSES", "42")

    assert set(store.changed_at()) == {"TOPIC_PASSES"}


def test_an_unparseable_value_is_the_stage_refusing_it_not_the_store(
    store,
) -> None:
    """The store holds text; what it has to parse as is the stage's own rule.

    Stored here deliberately, to show the split: the store takes it, and the
    stage is what refuses it. A route runs the stage's `load` over the
    resolved source before writing, which is what stops this reaching a
    worker.
    """
    from topic_modelling.config import Settings

    store.write("TOPIC_PASSES", "not a number")

    with pytest.raises(ValueError, match="TOPIC_PASSES"):
        Settings.load(store.resolved())
