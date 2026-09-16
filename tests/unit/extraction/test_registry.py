"""Which reader a passage's block type routes it to.

Unlike the parsing registry this one has a default: Docling's label set is
open-ended and most of it is prose, so only the kinds readable without a
model are claimed by name.
"""

from __future__ import annotations

import pytest

from extraction.extractors import ExtractorRegistry
from extraction.extractors.base import Extractor
from extraction.models import CandidateFact, PassageToExtract, Provenance


class Reader(Extractor):
    """An extractor claiming whatever it was built for."""

    def __init__(self, *block_types: str, method: str = "deterministic") -> None:
        """Initialises the reader with what it claims."""
        self.block_types = block_types
        self.method = method

    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Returns nothing; what is under test is the routing."""
        return []


def test_a_claimed_block_type_reaches_its_reader() -> None:
    """The table reader is chosen by name."""
    tables, default = Reader("table"), Reader()
    registry = ExtractorRegistry((tables,), default)

    assert registry.for_block_type("table") is tables


@pytest.mark.parametrize("block_type", ["text", "list_item", "", None])
def test_everything_unclaimed_falls_to_the_default(block_type) -> None:
    """Most of Docling's label set is prose."""
    default = Reader()
    registry = ExtractorRegistry((Reader("table"),), default)

    assert registry.for_block_type(block_type) is default


def test_two_readers_claiming_one_block_type_are_refused() -> None:
    """Which of them would run is not something a caller should discover."""
    with pytest.raises(ValueError, match="two extractors claim table"):
        ExtractorRegistry((Reader("table"), Reader("table")), Reader())


def test_the_claimed_types_are_reported_in_order() -> None:
    """What the status panel and the docs list."""
    registry = ExtractorRegistry((Reader("table", "picture"), Reader("code")), Reader())

    assert registry.block_types == ("code", "picture", "table")


def test_a_registry_claiming_nothing_by_name_answers_with_its_default() -> None:
    """A deployment with no deterministic reader still routes."""
    default = Reader()
    registry = ExtractorRegistry((), default)

    assert registry.block_types == ()
    assert registry.for_block_type("table") is default


def test_an_extractor_carries_no_provenance_unless_it_says_so() -> None:
    """A deterministic reader names no model and no prompt."""
    assert Reader().provenance == Provenance()
