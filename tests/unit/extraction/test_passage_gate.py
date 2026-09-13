"""Which passages never reach the model.

One model call costs minutes, and a heading or a caption returns its own text
back as a fact.
"""

from __future__ import annotations

import pytest
from factories import passage

from extraction.service import skipped

pytestmark = pytest.mark.nlp


def test_a_passage_carrying_a_claim_is_extracted() -> None:
    """Prose asserting something is not skipped."""
    assert skipped(passage()) is None


@pytest.mark.parametrize(
    "text", ["3.2 Lieferung und Versand", "Tabelle 1: Modelle", "Risiko"]
)
def test_a_passage_with_no_finite_verb_is_skipped(text: str) -> None:
    """A heading or a caption asserts nothing."""
    assert skipped(passage(text, language="de")) == "no finite verb", text


def test_a_passage_repeating_its_own_heading_trail_is_skipped() -> None:
    """A heading that carries a verb is still a heading."""
    heading = passage(
        "The device arrives in March.",
        section_path="Intro > The device arrives in March.",
    )
    assert skipped(heading) == "heading", skipped(heading)


def test_a_table_is_not_judged_on_its_sentences() -> None:
    """A table is read from its grid."""
    assert skipped(passage("| a | b |", block_type="table")) is None


@pytest.mark.parametrize("kwargs", [{}, {"block_type": "table"}])
def test_a_passage_with_nothing_numbered_is_skipped(kwargs) -> None:
    """Nothing can cite a passage that has no numbered units, table or not."""
    assert skipped(passage("| a | b |", sentences=[], **kwargs)) == "no sentences"
