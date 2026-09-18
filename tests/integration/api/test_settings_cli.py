"""Changing a setting from a terminal.

The command line and the route are one decision: both call
`settings.changes`, so a value refused on a page is refused here in the same
words, and one of them cannot quietly grow a rule the other lacks.

What is its own is the reading and the printing - which service a command
names, what `--set NAME=VALUE` parses as, and what a person sees.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def cli(database):
    """The command line, pointed at the container."""
    from settings.run import main

    return main


@pytest.fixture
def store(database):
    """The store, for reading back what a command wrote."""
    from settings.store import Settings

    return Settings()


def test_a_setting_is_written_from_the_command_line(cli, store) -> None:
    """The same operation as PATCH, and the same store behind it."""
    assert cli(["--service", "topics", "--set", "TOPIC_PASSES=42"]) == 0
    assert store.overrides() == {"TOPIC_PASSES": "42"}


def test_several_settings_are_written_together(cli, store) -> None:
    """One command is one change, as one request is."""
    assert (
        cli(
            [
                "--service",
                "topics",
                "--set",
                "TOPIC_PASSES=42",
                "--set",
                "TOPIC_TOP_TERMS=5",
            ]
        )
        == 0
    )
    assert store.overrides() == {"TOPIC_PASSES": "42", "TOPIC_TOP_TERMS": "5"}


def test_a_setting_is_returned_to_the_files(cli, store) -> None:
    """`--unset` deletes the override rather than writing a default."""
    cli(["--service", "topics", "--set", "TOPIC_PASSES=42"])

    assert cli(["--service", "topics", "--unset", "TOPIC_PASSES"]) == 0
    assert store.overrides() == {}


def test_reading_reports_without_changing_anything(cli, store) -> None:
    """A command with no change is the GET, and writes nothing."""
    from settings.run import report

    assert cli(["--service", "topics"]) == 0
    assert any("TOPIC_PASSES" in line for line in report("topics", store))
    assert store.overrides() == {}


def test_a_changed_setting_is_reported_beside_what_the_file_says(cli, store) -> None:
    """So a reader sees what was changed without knowing which file to open."""
    import os

    from settings.run import report

    cli(["--service", "topics", "--set", "TOPIC_PASSES=42"])
    said = next(one for one in report("topics", store) if "TOPIC_PASSES" in one)

    assert "42" in said
    assert "stored" in said
    assert f"file says {os.environ['TOPIC_PASSES']}" in said


def test_a_value_the_stage_cannot_parse_is_refused_in_its_own_words(cli, store) -> None:
    """The message a worker would have failed at start-up with."""
    with pytest.raises(SystemExit, match="TOPIC_PASSES"):
        cli(["--service", "topics", "--set", "TOPIC_PASSES=many"])

    assert store.overrides() == {}


def test_a_setting_another_service_owns_is_refused(cli) -> None:
    """One command configures one service, as one page does."""
    with pytest.raises(SystemExit, match="configured by questions"):
        cli(["--service", "topics", "--set", "QUESTIONS_PER_TOPIC=9"])


def test_a_setting_the_deployment_owns_is_refused(cli) -> None:
    """A pool size is read when a process starts; storing one would do nothing."""
    with pytest.raises(SystemExit, match="DATABASE_POOL_SIZE"):
        cli(["--service", "platform", "--set", "DATABASE_POOL_SIZE=50"])


def test_a_set_without_a_value_says_what_it_takes(cli) -> None:
    """A typo in the one place a command line can have one."""
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        cli(["--service", "topics", "--set", "TOPIC_PASSES"])


def test_setting_and_unsetting_one_name_at_once_is_refused(cli) -> None:
    """Only one of those can be what was meant."""
    with pytest.raises(SystemExit, match="both set and unset"):
        cli(
            [
                "--service",
                "topics",
                "--set",
                "TOPIC_PASSES=42",
                "--unset",
                "TOPIC_PASSES",
            ]
        )


def test_a_service_nothing_configures_is_refused_by_the_parser(cli) -> None:
    """The list is closed, so a typo is caught before anything is read."""
    with pytest.raises(SystemExit):
        cli(["--service", "nonsense"])


def test_the_platform_is_configurable_from_a_terminal(cli, store) -> None:
    """The one service with no stage, and so no command line of its own.

    Which is why there is one command for all seven rather than a flag on
    each stage's own line: the model every stage calls has no worker to hang
    a flag off.
    """
    assert cli(["--service", "platform", "--set", "LLM_MODEL=ollama_chat/other"]) == 0

    from llm.config import Settings

    assert Settings.load(store.resolved()).model == "ollama_chat/other"


def test_a_change_that_stales_something_says_so(cli, store) -> None:
    """The remedy costs a corpus-sized run, so it is said rather than done."""
    from settings.run import change

    said = change("platform", {"EMBEDDING_MODEL": "intfloat/other"}, store)

    assert any("chunking, questions" in line for line in said)
    assert any("rerun" in line for line in said)


def test_the_command_line_and_the_route_refuse_the_same_thing(client) -> None:
    """One decision, reached through two doors.

    The command line called the route's own function once, which meant a
    command line that imported FastAPI and the whole composition root. Both
    now call settings.changes, and this is what says they still agree.
    """
    from settings.changes import Refused, apply

    refused = client.patch(
        "/settings/topics", json={"values": {"TOPIC_PASSES": "many"}}
    )

    with pytest.raises(Refused) as directly:
        apply("topics", {"TOPIC_PASSES": "many"})

    assert refused.json()["code"] == directly.value.code
    assert refused.json()["detail"] == directly.value.detail
