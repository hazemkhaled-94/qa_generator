"""What a deployment may ask extraction to write.

A setting that is wrong stops the service at start-up naming itself, rather
than producing a corpus with a kind silently missing from it.
"""

from __future__ import annotations

import pytest

from database.qa_generator import FactKind
from extraction.config import Settings


def settings(monkeypatch, **overrides: str) -> Settings:
    """Loads the settings with the environment overridden.

    Args:
        monkeypatch: The fixture that sets them.
        **overrides: The values to set, by setting name.

    Returns:
        The loaded settings.
    """
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    return Settings.load()


def test_the_declared_environment_loads(monkeypatch) -> None:
    """What configs/env/backend.env names is what the service reads."""
    read = Settings.load()

    assert FactKind.ATOMIC in read.kinds
    assert 0 < read.digest_share <= 1
    assert read.bridges_per_topic >= 0
    assert read.bridge_passages >= 2


def test_every_kind_may_be_asked_for(monkeypatch) -> None:
    """The four readings, all switched on."""
    read = settings(monkeypatch, EXTRACTION_KINDS="atomic,summary,outline,bridge")

    assert read.kinds == frozenset(FactKind)
    assert read.digests == (FactKind.SUMMARY, FactKind.OUTLINE)
    assert read.bridging


def test_the_digests_come_back_in_storage_order(monkeypatch) -> None:
    """A summary before its outline, whichever order the setting names them."""
    read = settings(monkeypatch, EXTRACTION_KINDS="atomic,outline,summary")
    assert read.digests == (FactKind.SUMMARY, FactKind.OUTLINE)


def test_a_deployment_writing_only_atomic_facts_asks_for_nothing_else(
    monkeypatch,
) -> None:
    """Neither the digest call nor the bridge pass costs anything."""
    read = settings(monkeypatch, EXTRACTION_KINDS="atomic")

    assert read.digests == ()
    assert not read.bridging


def test_spacing_and_case_are_forgiven(monkeypatch) -> None:
    """A list a person edited by hand still reads."""
    read = settings(monkeypatch, EXTRACTION_KINDS=" Atomic , SUMMARY ")
    assert read.kinds == {FactKind.ATOMIC, FactKind.SUMMARY}


def test_a_kind_nothing_answers_to_is_refused(monkeypatch) -> None:
    """A typo would otherwise read as a kind nobody writes."""
    with pytest.raises(ValueError, match="paragraph"):
        settings(monkeypatch, EXTRACTION_KINDS="atomic,paragraph")


def test_leaving_out_the_atomic_kind_is_refused(monkeypatch) -> None:
    """Every passage is read for its claims; the rest are extra."""
    with pytest.raises(ValueError, match="atomic"):
        settings(monkeypatch, EXTRACTION_KINDS="summary,outline")


def test_an_empty_list_is_refused(monkeypatch) -> None:
    """An unset setting stops the service rather than defaulting in code."""
    with pytest.raises(KeyError, match="EXTRACTION_KINDS"):
        settings(monkeypatch, EXTRACTION_KINDS="")


def test_a_share_that_is_not_a_number_is_refused(monkeypatch) -> None:
    """Naming the variable, so the fix is one line in a file."""
    with pytest.raises(ValueError, match="EXTRACTION_DIGEST_MAX_SHARE"):
        settings(monkeypatch, EXTRACTION_DIGEST_MAX_SHARE="most of it")


def test_a_group_size_that_is_not_a_whole_number_is_refused(monkeypatch) -> None:
    """Half a passage cannot be put in front of a model."""
    with pytest.raises(ValueError, match="EXTRACTION_BRIDGE_PASSAGES"):
        settings(monkeypatch, EXTRACTION_BRIDGE_PASSAGES="2.5")
