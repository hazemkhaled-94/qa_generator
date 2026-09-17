"""Reading a setting from somewhere other than the process environment.

Every reader takes an optional source, and every stage's `load` threads it
through. That is what lets a value stored for a service reach the stage
without a restart, and what lets the API parse a proposed value with the
same parser the stage itself uses rather than a second one that agrees with
it today.

The environment is still what a container starts from, so the tests here
check both: a source overrides a setting, and nothing about it makes the
environment optional.
"""

from __future__ import annotations

import os

import pytest

from settings import boolean, csv, decimal, integer, mapping, optional, required

NAME = "QA_TEST_SETTING"


def test_a_source_is_read_instead_of_the_environment(monkeypatch) -> None:
    """The environment is not consulted when a source carries the name."""
    monkeypatch.setenv(NAME, "from-the-environment")

    assert required(NAME, {NAME: "from-the-source"}) == "from-the-source"


def test_a_source_is_read_and_trimmed() -> None:
    """Trimmed the same way the environment is."""
    assert required(NAME, {NAME: "  spaced  "}) == "spaced"


@pytest.mark.parametrize("source", [{}, {NAME: ""}, {NAME: "   "}])
def test_a_name_a_source_does_not_carry_is_missing(monkeypatch, source) -> None:
    """A source overrides a setting; it does not stop one being required.

    Empty counts as missing here too, so a row storing an empty string reads
    as the absence it looks like rather than as a value.
    """
    monkeypatch.delenv(NAME, raising=False)

    with pytest.raises(KeyError, match=NAME):
        required(NAME, source)


def test_an_empty_source_does_not_fall_back_to_the_environment(monkeypatch) -> None:
    """A source is the whole answer, so a caller composes it deliberately.

    The resolver overlays the environment itself. Falling back here as well
    would mean a setting could arrive from either of two places depending on
    which reader ran, which is the disagreement this parameter exists to
    prevent.
    """
    monkeypatch.setenv(NAME, "from-the-environment")

    with pytest.raises(KeyError, match=NAME):
        required(NAME, {})


def test_the_environment_is_read_when_no_source_is_given(monkeypatch) -> None:
    """Absent, the source is the process environment, as a container starts."""
    monkeypatch.setenv(NAME, "from-the-environment")

    assert required(NAME) == "from-the-environment"


def test_os_environ_itself_is_an_acceptable_source(monkeypatch) -> None:
    """It is a mapping, so passing it is the same as passing nothing."""
    monkeypatch.setenv(NAME, "either-way")

    assert required(NAME, os.environ) == required(NAME)


@pytest.mark.parametrize(
    ("read", "value", "expected"),
    [
        (optional, "http://host", "http://host"),
        (integer, "7", 7),
        (decimal, "0.25", 0.25),
        (boolean, "yes", True),
        (boolean, "nonsense", False),
        (csv, "de, en", ("de", "en")),
        (mapping, "de:German, en:English", {"de": "German", "en": "English"}),
    ],
)
def test_every_reader_takes_a_source(read, value, expected) -> None:
    """Named one at a time, so a reader left unthreaded says which it was."""
    assert read(NAME, {NAME: value}) == expected


def test_an_absent_optional_setting_is_none_in_a_source_too(monkeypatch) -> None:
    """Absence is the answer whichever place it is absent from."""
    monkeypatch.setenv(NAME, "from-the-environment")

    assert optional(NAME, {}) is None


def test_a_number_that_is_not_one_names_itself_from_a_source() -> None:
    """The message says what to go and fix, wherever the value came from."""
    with pytest.raises(ValueError, match=NAME):
        integer(NAME, {NAME: "seven"})


def _overridden(**values: str) -> dict[str, str]:
    """The tuning values every test runs under, with some replaced.

    A source is the whole answer rather than a layer over the environment,
    so a stage's `load` has to be given everything it reads. This is what
    the resolver will compose; here it is composed by hand.
    """
    return {**os.environ, **values}


#: One stage at a time: its settings module, a setting only it reads, the
#: value to put there and the attribute it should come out as. Every `load`
#: that takes a source is named here, so one left unthreaded fails saying
#: which stage it was.
THREADED = [
    ("ingestion.config", "MAX_FILE_SIZE_MB", "7", "max_file_size_mb", 7),
    (
        "preprocessing.parsing.config",
        "PARSING_TABLE_MODE",
        "fast",
        "table_mode",
        "fast",
    ),
    (
        "preprocessing.chunking.config",
        "CHUNKING_MERGE_PEERS",
        "false",
        "merge_peers",
        False,
    ),
    ("extraction.config", "EXTRACTION_BRIDGE_PASSAGES", "3", "bridge_passages", 3),
    ("topic_modelling.config", "TOPIC_PASSES", "42", "passes", 42),
    ("question_generation.config", "QUESTIONS_PER_TOPIC", "9", "per_topic", 9),
    ("llm.config", "LLM_MODEL", "ollama_chat/other", "model", "ollama_chat/other"),
]


@pytest.mark.parametrize(
    ("module", "name", "value", "attribute", "expected"),
    THREADED,
    ids=[one[0] for one in THREADED],
)
def test_a_stage_loads_from_a_source(module, name, value, attribute, expected) -> None:
    """The value a source carries is the one the stage comes up with."""
    imported = pytest.importorskip(module)

    loaded = imported.Settings.load(_overridden(**{name: value}))

    assert getattr(loaded, attribute) == expected


@pytest.mark.parametrize("module", [one[0] for one in THREADED])
def test_a_stage_reads_the_environment_when_given_no_source(
    monkeypatch, module
) -> None:
    """Loading without one is what every caller does today, unchanged.

    LLM_MODEL is set here because it comes from .env rather than from
    configs/env/backend.env, which is the file the test environment is built
    from.
    """
    monkeypatch.setenv("LLM_MODEL", "ollama_chat/test")
    imported = pytest.importorskip(module)

    assert imported.Settings.load() == imported.Settings.load(os.environ)


def test_the_nested_model_settings_are_read_from_the_same_source() -> None:
    """Topic modelling holds an llm.config.Settings, which must not escape it.

    The one nested `load` in the project. Reading the environment here while
    its parent read a source is how a topic could be named by a model nobody
    configured for it.
    """
    from topic_modelling.config import Settings

    loaded = Settings.load(_overridden(LLM_MODEL="ollama_chat/nested"))

    assert loaded.model is not None
    assert loaded.model.model == "ollama_chat/nested"


def test_a_mix_naming_an_unknown_type_is_refused_from_a_source() -> None:
    """The stage's own validation, reached through a source.

    The point of threading rather than re-implementing: what the API will
    refuse a proposed value with is this message, written once.
    """
    from question_generation.config import Settings

    with pytest.raises(ValueError, match="QUESTIONS_TYPE_MIX"):
        Settings.load(_overridden(QUESTIONS_TYPE_MIX="nonsense:1"))


def test_extraction_still_insists_on_the_atomic_kind_from_a_source() -> None:
    """Every passage is read for its claims, however the setting arrived."""
    from extraction.config import Settings

    with pytest.raises(ValueError, match="EXTRACTION_KINDS"):
        Settings.load(_overridden(EXTRACTION_KINDS="summary,outline"))
