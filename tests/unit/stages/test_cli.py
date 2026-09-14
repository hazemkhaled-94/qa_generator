"""The command line every pipeline stage shares."""

from __future__ import annotations

import pytest

from extraction.repository import PassageQueue
from preprocessing.parsing.repository import ParseQueue
from stages.cli import narrowing, parser

ACTIONS = {"status": "report", "start": "queue", "stop": "take back"}


def test_no_narrowing_asked_for_is_the_whole_queue() -> None:
    """`--only` is optional."""
    assert narrowing(ParseQueue(), None) is None


def test_a_scope_and_value_becomes_a_condition() -> None:
    """The same condition the route builds."""
    assert narrowing(ParseQueue(), "document=abc") is not None


@pytest.mark.parametrize("only", ["document", "", "justtext"])
def test_a_narrowing_without_a_value_is_refused(only: str) -> None:
    """The message names the form and the scopes this stage takes."""
    with pytest.raises(SystemExit, match="SCOPE=VALUE"):
        narrowing(ParseQueue(), only)


def test_a_scope_this_stage_does_not_take_is_refused() -> None:
    """Parsing queues over documents and knows nothing of a passage."""
    with pytest.raises(SystemExit, match="narrows to document"):
        narrowing(ParseQueue(), "passage=1")


def test_a_value_the_column_cannot_hold_is_refused() -> None:
    """Refused here rather than at the database."""
    with pytest.raises(SystemExit, match="not a valid passage"):
        narrowing(PassageQueue(), "passage=not-a-number")


def test_two_actions_in_one_command_are_refused() -> None:
    """They used to run only the first, silently."""
    with pytest.raises(SystemExit):
        parser("stage.run", ACTIONS).parse_args(["--start", "--stop"])


def test_each_action_is_a_flag_of_its_own() -> None:
    """The flags mirror the stage's HTTP surface one for one."""
    parsed = parser("stage.run", ACTIONS).parse_args(["--status"])

    assert parsed.status is True
    assert parsed.start is False
    assert parsed.only is None
    assert parsed.watch is False


def test_the_shared_flags_are_always_there() -> None:
    """`--only` and `--watch` are not a stage's to declare."""
    parsed = parser("stage.run", ACTIONS).parse_args(
        ["--only", "document=abc", "--watch"]
    )

    assert parsed.only == "document=abc"
    assert parsed.watch is True
