"""The message a group of passages becomes, and what comes back from it.

A bridge candidate names the passages it rests on by position, so what the
prompt numbers and what the checker resolves have to be the same numbering.
"""

from __future__ import annotations

import pytest
from drivers import Model, group

from database.qa_generator import FactKind
from extraction.extractors.base import ExtractionFailed
from extraction.extractors.bridge import PROMPT_VERSION, BridgeExtractor
from llm.client import ModelUnavailable

ANSWER = {
    "facts": [
        {"passages": [0, 1], "statement": "Two response times are stated."},
    ]
}


@pytest.fixture
def offered():
    """Two passages of two documents, as a topic group offers them."""
    return group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )


def test_the_provenance_names_the_model_and_the_prompt() -> None:
    """Two prompts are two datasets, so the version is recorded per fact."""
    provenance = BridgeExtractor(Model()).provenance

    assert provenance.model == "ollama/test-model"
    assert provenance.prompt_version == PROMPT_VERSION
    assert provenance.temperature == 0.0


def test_a_group_of_one_is_never_sent(offered) -> None:
    """There is nothing to bridge, so the call is not worth making."""
    model = Model(ANSWER)

    assert BridgeExtractor(model).extract(offered[:1]) == []
    assert BridgeExtractor(model).extract([]) == []
    assert model.calls == 0


def test_the_passages_are_sent_numbered_from_zero(offered) -> None:
    """The number is what a candidate's position resolves against."""
    model = Model(ANSWER)
    BridgeExtractor(model).extract(offered)

    sent = model.sent["user"]
    assert "[P0]" in sent and "[P1]" in sent
    assert sent.index("[P0]") < sent.index("[P1]")
    assert all(one.text in sent for one in offered)


def test_a_heading_is_carried_beside_its_passage(offered) -> None:
    """Which passage a heading belongs to is what makes it worth sending."""
    from dataclasses import replace

    model = Model(ANSWER)
    BridgeExtractor(model).extract(
        [replace(offered[0], section_path="Support > Times"), offered[1]]
    )

    assert "[P0] (Support > Times)" in model.sent["user"]


def test_a_returned_claim_becomes_a_bridge_candidate(offered) -> None:
    """The statement, the positions it rests on, and nothing else."""
    facts = BridgeExtractor(Model(ANSWER)).extract(offered)

    assert len(facts) == 1
    assert facts[0].kind == FactKind.BRIDGE
    assert facts[0].passages == (0, 1)
    assert facts[0].sentences == (), "a bridge cites passages, not sentences"


def test_a_repeated_position_is_kept_once_and_in_order(offered) -> None:
    """The same passage named twice is one passage."""
    answer = {"facts": [{"passages": [1, 0, 1], "statement": "Both are stated."}]}
    facts = BridgeExtractor(Model(answer)).extract(offered)

    assert facts[0].passages == (1, 0)


@pytest.mark.parametrize(
    "returned",
    [
        {"passages": [0, 1], "statement": ""},
        {"passages": [0, 1], "statement": "   "},
        {"passages": [], "statement": "A claim resting on nothing."},
    ],
)
def test_a_claim_with_no_statement_or_no_passage_is_dropped(offered, returned) -> None:
    """Neither can be checked, so neither is a candidate."""
    assert BridgeExtractor(Model({"facts": [returned]})).extract(offered) == []


def test_an_empty_answer_is_a_correct_answer(offered) -> None:
    """Passages that merely share a topic need bridge nothing."""
    assert BridgeExtractor(Model({"facts": []})).extract(offered) == []


def test_a_model_that_will_not_answer_fails_the_group(offered) -> None:
    """The group is skipped with the reason, not left mid-extraction."""
    model = Model(ModelUnavailable("Timeout: took too long"))

    with pytest.raises(ExtractionFailed, match="took too long"):
        BridgeExtractor(model).extract(offered)
