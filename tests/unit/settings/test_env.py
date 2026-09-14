"""Reading configuration out of the environment.

There are no defaults in code, so every one of these is a trust boundary: a
setting that is missing, empty or malformed must stop the service and name
itself rather than be guessed at.
"""

from __future__ import annotations

import pytest

from settings import boolean, csv, decimal, integer, optional, required

NAME = "QA_TEST_SETTING"


@pytest.fixture(autouse=True)
def unset(monkeypatch) -> None:
    """Leaves the setting absent unless a test sets it."""
    monkeypatch.delenv(NAME, raising=False)


def test_a_present_value_is_read_and_trimmed(monkeypatch) -> None:
    """Surrounding whitespace is not part of a setting."""
    monkeypatch.setenv(NAME, "  postgres://host  ")
    assert required(NAME) == "postgres://host"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_or_empty_required_setting_stops_the_service(
    monkeypatch, value
) -> None:
    """Empty counts as missing: compose turns an unset variable into one."""
    if value is not None:
        monkeypatch.setenv(NAME, value)

    with pytest.raises(KeyError, match=NAME):
        required(NAME)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), ("", None), ("  ", None)]
)
def test_an_absent_optional_setting_is_none(monkeypatch, value, expected) -> None:
    """Absence is itself the answer."""
    if value is not None:
        monkeypatch.setenv(NAME, value)

    assert optional(NAME) is expected


def test_an_optional_setting_that_is_present_is_trimmed(monkeypatch) -> None:
    """Read the same way as a required one."""
    monkeypatch.setenv(NAME, " http://ollama:11434 ")
    assert optional(NAME) == "http://ollama:11434"


@pytest.mark.parametrize(("value", "expected"), [("5", 5), (" 12 ", 12), ("-3", -3)])
def test_a_whole_number_is_read(monkeypatch, value, expected) -> None:
    """Whatever int() accepts."""
    monkeypatch.setenv(NAME, value)
    assert integer(NAME) == expected


@pytest.mark.parametrize("value", ["5.5", "five", "", "5 attempts"])
def test_a_setting_that_is_not_a_whole_number_names_itself(monkeypatch, value) -> None:
    """The variable is named, so the message says what to go and fix."""
    monkeypatch.setenv(NAME, value)

    with pytest.raises((ValueError, KeyError), match=NAME):
        integer(NAME)


@pytest.mark.parametrize(
    ("value", "expected"), [("0.5", 0.5), ("2", 2.0), (" 1e3 ", 1000.0)]
)
def test_a_number_is_read(monkeypatch, value, expected) -> None:
    """Whatever float() accepts."""
    monkeypatch.setenv(NAME, value)
    assert decimal(NAME) == expected


@pytest.mark.parametrize("value", ["half", "", "0.5.5"])
def test_a_setting_that_is_not_a_number_names_itself(monkeypatch, value) -> None:
    """The same refusal as a whole number."""
    monkeypatch.setenv(NAME, value)

    with pytest.raises((ValueError, KeyError), match=NAME):
        decimal(NAME)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_the_accepted_spellings_of_true(monkeypatch, value) -> None:
    """Case and surrounding whitespace do not matter."""
    monkeypatch.setenv(NAME, value)
    assert boolean(NAME) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "nonsense", "2"])
def test_anything_else_is_false(monkeypatch, value) -> None:
    """A boolean that guesses is worse than one that is simply off."""
    monkeypatch.setenv(NAME, value)
    assert boolean(NAME) is False


def test_a_missing_flag_stops_the_service() -> None:
    """Off is a decision; absent is not."""
    with pytest.raises(KeyError, match=NAME):
        boolean(NAME)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de,en", ("de", "en")),
        (" de , en ", ("de", "en")),
        ("de,,en", ("de", "en")),
        ("de,", ("de",)),
        ("de", ("de",)),
    ],
)
def test_a_list_is_split_trimmed_and_emptied(monkeypatch, value, expected) -> None:
    """A trailing comma names no extra language."""
    monkeypatch.setenv(NAME, value)
    assert csv(NAME) == expected


def test_a_list_of_nothing_but_separators_is_missing(monkeypatch) -> None:
    """`,,,` names nothing, and nothing is what `required` refuses."""
    monkeypatch.setenv(NAME, ",,,")
    assert csv(NAME) == ()
