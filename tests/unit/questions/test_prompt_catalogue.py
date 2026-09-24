"""What each stage declares it sends, against what it can actually send.

The catalogue is what resolves `questions.prompt_version` and
`facts.prompt_version` to a text. A prompt missing from it is a version
that resolves to nothing for the calls that used it, and nothing else
would notice: the stage still sends it, the row still records the version,
and only the lookup comes back empty.
"""

from __future__ import annotations

import pytest

from extraction import prompts as extraction
from extraction.extractors import bridge, digest, llm
from question_generation import phrasing, verifier
from question_generation import prompts as questions
from question_generation.types import PROMPT_VERSION, SPECS


@pytest.fixture(scope="module")
def catalogue() -> dict[str, object]:
    """Question generation's catalogue, by name."""
    return {one.name: one for one in questions.catalogue()}


def test_every_question_type_declares_its_prompt(catalogue) -> None:
    """A type added without one writes questions nothing can resolve."""
    missing = sorted(name for name in SPECS if name not in catalogue)

    assert not missing, f"{missing} can be sent and is not in the catalogue"


def test_a_spanning_variant_is_its_own_prompt(catalogue) -> None:
    """A type sent the spanning instruction was sent a different prompt.

    Recording only the plain one would be a record of half of what the
    stage sends: a single-passage type still gets the spanning prompt
    when its sample happens to cross passages.
    """
    spanning = [name for name in catalogue if name.endswith("(spans)")]

    assert spanning, "no spanning variant was declared"
    for name in spanning:
        plain = name.removesuffix(" (spans)")
        assert catalogue[name].text != catalogue[plain].text


def test_the_judges_are_declared_at_their_own_versions(catalogue) -> None:
    """Three modules, three versions, bumped for different reasons."""
    assert catalogue["factoid"].version == PROMPT_VERSION
    assert catalogue["phrasing: source"].version == phrasing.PROMPT_VERSION
    assert catalogue["verifier: recover"].version == verifier.PROMPT_VERSION


def test_every_verifier_prompt_is_declared(catalogue) -> None:
    """Four of them, and the three below `_VERIFY` are a different ask."""
    declared = {one.text for one in catalogue.values()}
    for text in (
        verifier._VERIFY,
        verifier._SUPPORT,
        verifier._COMPUTES,
        verifier._FOLLOWS,
    ):
        assert text in declared


def test_every_phrasing_prompt_is_declared(catalogue) -> None:
    """Three of them, and `_NAMES` was the one nobody declared.

    The verifier's four had a check of this shape and the phrasing
    judgements did not, so the one that was missing stayed missing:
    `names_something` is sent wherever the parse has already called a
    question thin, and the version on those rows resolved to two prompts
    out of three.
    """
    declared = {one.text for one in catalogue.values()}
    for text in (phrasing._SOURCE, phrasing._NAMES, phrasing._CONTAINED):
        assert text in declared


def test_a_declared_prompt_is_never_empty(catalogue) -> None:
    """An empty one resolves a version to nothing, which is worse than none."""
    for name, one in catalogue.items():
        assert one.text.strip(), name
        assert one.version, name


def test_extraction_declares_each_extractor_at_its_own_version() -> None:
    """Three extractors, three versions."""
    by_name = {one.name: one for one in extraction.catalogue()}

    assert by_name["atomic"].version == llm.PROMPT_VERSION
    assert by_name["digest"].version == digest.PROMPT_VERSION
    assert by_name["bridge"].version == bridge.PROMPT_VERSION


def test_the_atomic_prompt_is_recorded_as_the_cap_makes_it() -> None:
    """The cap is appended to the prompt, so a capped run sends another.

    Recorded without it, the row would be a record of a prompt no model
    was given under any run that caps.
    """
    uncapped = {one.name: one for one in extraction.catalogue()}["atomic"]
    capped = {one.name: one for one in extraction.catalogue(4)}["atomic"]

    assert capped.text != uncapped.text
    assert "4" in capped.text.removeprefix(uncapped.text)
