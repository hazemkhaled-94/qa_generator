"""The pass that reads a corpus's topics for claims spanning passages.

Not part of the passage queue: the unit of work is a group of passages the
topic model put together, so it runs on its own and replaces what it wrote
last time rather than adding to it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from drivers import DIGEST_SHARE, Catalogue, Model, group, passage

from database.qa_generator import FactKind
from extraction.extractors.bridge import BridgeExtractor
from extraction.service import bridge
from extraction.validation import FactChecker
from llm.client import ModelUnavailable

pytestmark = pytest.mark.nlp

ANSWER = {
    "facts": [
        {
            "passages": [
                {"passage": 0, "sentences": [0]},
                {"passage": 1, "sentences": [0]},
            ],
            "statement": "Support response times are stated separately for "
            "standard and urgent requests.",
        }
    ]
}


def pair():
    """Two passages of two documents, as one topic offers them."""
    return group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )


def run(catalog: Catalogue, model: Model, wanted: int = 5, size: int = 2) -> int:
    """Runs the bridge pass over a catalogue with one scripted model."""
    return bridge(
        catalog,  # type: ignore[arg-type]
        BridgeExtractor(model),  # type: ignore[arg-type]
        FactChecker(DIGEST_SHARE),
        wanted,
        size,
    )


def test_a_topic_of_two_passages_yields_one_bridge() -> None:
    """One group, one call, one stored claim."""
    catalog, model = Catalogue(support=pair()), Model(ANSWER)

    assert run(catalog, model) == 1
    assert model.calls == 1
    assert catalog.kinds == [FactKind.BRIDGE]
    assert catalog.written[0].validated, catalog.written[0].validation_error


def test_the_previous_pass_is_replaced_rather_than_added_to() -> None:
    """Re-running twice must not double the bridges."""
    catalog = Catalogue(support=pair())
    run(catalog, Model(ANSWER))

    assert catalog.cleared == 1, "the clear happens once, before anything is read"


def test_every_topic_is_read() -> None:
    """A run covers the corpus, not one subject of it."""
    catalog, model = Catalogue(support=pair(), delivery=pair()), Model(ANSWER)

    assert run(catalog, model) == 2
    assert model.calls == 2


def test_a_topic_too_small_to_group_is_skipped() -> None:
    """Nothing is asked of a topic that offers no pair."""
    catalog, model = Catalogue(support=pair()[:1]), Model(ANSWER)

    assert run(catalog, model) == 0
    assert model.calls == 0


def test_a_group_the_model_will_not_read_does_not_stop_the_rest() -> None:
    """One unreachable call is not a reason to abandon the corpus."""

    class Flaky(Model):
        """Refuses the first group and answers the second."""

        def answer(self, **kwargs):
            """Raises once, then answers."""
            if not self.asked:
                self.asked.append(kwargs)
                raise ModelUnavailable("Timeout: took too long")
            return super().answer(**kwargs)

    catalog = Catalogue(support=pair(), delivery=pair())

    assert run(catalog, Flaky(ANSWER)) == 1, "the second group still wrote"


def test_a_refused_bridge_is_stored_with_its_reason() -> None:
    """The share that failed is how the pass is judged, as everywhere else."""
    alone = {
        "facts": [
            {
                "passages": [{"passage": 0, "sentences": [0]}],
                "statement": "Standard requests are answered within 48 hours.",
            }
        ]
    }
    catalog = Catalogue(support=pair())

    assert run(catalog, Model(alone)) == 1
    assert not catalog.written[0].validated
    assert catalog.written[0].rejection_code == "not_bridging"


def test_a_topic_that_bridges_nothing_writes_nothing() -> None:
    """Returning no facts is a correct answer."""
    catalog = Catalogue(support=pair())

    assert run(catalog, Model({"facts": []})) == 0
    assert catalog.written == []


class TestThePassageGate:
    """The same gate the per-passage stage applies, applied here too.

    `by_topic` asks only that a passage was segmented, so without this a
    group is spent on a table of contents or a bare heading - which is what
    `skipped` exists to refuse, and the index this corpus renders as a code
    block yielded 95 facts from four passages before it did.
    """

    def test_navigation_is_not_offered_to_the_model(self) -> None:
        """A group of two, one of which is a contents page, is not a group."""
        listing = replace(pair()[1], block_type="document_index", doc_sha256="doc-1")
        catalog, model = Catalogue(support=[pair()[0], listing]), Model(ANSWER)

        assert run(catalog, model) == 0
        assert model.calls == 0, "one readable passage is not a pair"

    def test_a_heading_is_not_offered_either(self) -> None:
        """It asserts nothing, so there is nothing for it to bridge."""
        heading = passage(
            "Response times",
            section_path="Support > Response times",
            id=22,
            doc_sha256="doc-1",
        )
        catalog, model = Catalogue(support=[pair()[0], heading]), Model(ANSWER)

        assert run(catalog, model) == 0
        assert model.calls == 0

    def test_the_readable_passages_of_a_topic_still_group(self) -> None:
        """The gate takes the junk out, it does not refuse the topic."""
        listing = replace(pair()[1], block_type="document_index", doc_sha256="doc-9")
        catalog = Catalogue(support=[*pair(), listing])

        assert run(catalog, Model(ANSWER)) == 1


def test_the_narrowing_reaches_both_the_clear_and_the_read() -> None:
    """`--only` selects the same passages for each half of the pass."""
    catalog = Catalogue(support=pair())
    bridge(
        catalog,  # type: ignore[arg-type]
        BridgeExtractor(Model(ANSWER)),  # type: ignore[arg-type]
        FactChecker(DIGEST_SHARE),
        5,
        2,
        "a-condition",
    )

    assert catalog.narrowed == ["a-condition"]
