"""What a passage becomes when it is condensed rather than decomposed.

One model call produces both readings, so a deployment that wants only one
of them still pays for one call and not two.
"""

from __future__ import annotations

import pytest
from drivers import Model, passage

from database.qa_generator import FactKind
from extraction.extractors.base import ExtractionFailed
from extraction.extractors.digest import PROMPT_VERSION, DigestExtractor
from extraction.models import BULLET
from llm.client import ModelUnavailable

BOTH = (FactKind.SUMMARY, FactKind.OUTLINE)

ANSWER = {
    "summary": "The device weighs 4 kg.",
    "outline": ["Weighs 4 kg", "Runs for 12 hours"],
}


def test_the_provenance_names_the_model_and_the_prompt() -> None:
    """Two prompts are two datasets, so the version is recorded per fact."""
    provenance = DigestExtractor(Model(), BOTH).provenance

    assert provenance.model == "ollama/test-model"
    assert provenance.prompt_version == PROMPT_VERSION
    assert provenance.temperature == 0.0


def test_one_call_produces_both_readings() -> None:
    """Asking twice would cost twice for the same reading."""
    model = Model(ANSWER)
    facts = DigestExtractor(model, BOTH).extract(passage())

    assert model.calls == 1
    assert [fact.kind for fact in facts] == [FactKind.SUMMARY, FactKind.OUTLINE]


def test_only_the_kinds_asked_for_are_kept() -> None:
    """A deployment writing summaries alone stores no outline."""
    facts = DigestExtractor(Model(ANSWER), (FactKind.SUMMARY,)).extract(passage())

    assert [fact.kind for fact in facts] == [FactKind.SUMMARY]


def test_a_digest_cites_every_sentence_of_its_passage() -> None:
    """It stands in for the whole thing, not for part of it."""
    facts = DigestExtractor(Model(ANSWER), BOTH).extract(passage())

    assert all(fact.sentences == (0, 1) for fact in facts), facts
    assert all(fact.passages == () for fact in facts), "a digest names no group"


def test_the_points_become_one_bulleted_statement() -> None:
    """An outline is one fact whose statement is the list."""
    outline = DigestExtractor(Model(ANSWER), (FactKind.OUTLINE,)).extract(passage())[0]

    assert outline.statement == f"{BULLET}Weighs 4 kg\n{BULLET}Runs for 12 hours"


@pytest.mark.parametrize("marker", ["- ", "• ", "* ", ""])
def test_a_point_the_model_already_marked_is_not_marked_twice(marker) -> None:
    """The model is told to return points, and returns bullets anyway."""
    answer = {"summary": "s", "outline": [f"{marker}Weighs 4 kg", f"{marker}Ships"]}
    outline = DigestExtractor(Model(answer), (FactKind.OUTLINE,)).extract(passage())[0]

    assert outline.statement == f"{BULLET}Weighs 4 kg\n{BULLET}Ships"


def test_an_empty_reading_is_dropped_rather_than_stored() -> None:
    """Nothing is stored for a reading the model did not give."""
    facts = DigestExtractor(Model({"summary": "   ", "outline": []}), BOTH).extract(
        passage()
    )
    assert facts == []


def test_a_blank_point_does_not_become_a_bullet() -> None:
    """An empty line is not a point."""
    answer = {"summary": "", "outline": ["Weighs 4 kg", "  ", ""]}
    outline = DigestExtractor(Model(answer), BOTH).extract(passage())[0]

    assert outline.statement == f"{BULLET}Weighs 4 kg"


def test_a_heading_trail_is_fenced_off_and_labelled() -> None:
    """The same fencing the atomic prompt uses, for the same reason."""
    model = Model(ANSWER)
    DigestExtractor(model, BOTH).extract(passage(section_path="Support > Times"))

    sent = model.sent["user"]
    assert sent.startswith("Context (not part of the excerpt")
    assert "Support > Times" in sent
    assert sent.index("Context") < sent.index("Excerpt:")


def test_the_whole_passage_is_sent_rather_than_numbered_sentences() -> None:
    """A digest cites nothing by number, so nothing is numbered for it."""
    model = Model(ANSWER)
    under = passage()
    DigestExtractor(model, BOTH).extract(under)

    assert under.text in model.sent["user"]
    assert "[0]" not in model.sent["user"]


def test_a_model_that_will_not_answer_fails_the_passage() -> None:
    """The row is failed with the reason, not left mid-extraction."""
    model = Model(ModelUnavailable("Timeout: took too long"))

    with pytest.raises(ExtractionFailed, match="took too long"):
        DigestExtractor(model, BOTH).extract(passage())
