"""The message a passage becomes, and what comes back from it.

The model sees one passage's numbered sentences and its heading trail, never
the whole document, and cites a sentence by number rather than quoting it.
"""

from __future__ import annotations

import pytest
from factories import passage

from extraction.extractors.base import ExtractionFailed
from extraction.extractors.llm import PROMPT_VERSION, LlmExtractor
from llm.client import ModelUnavailable

pytestmark = pytest.mark.nlp


class _Fact:
    """One fact as the model returns it."""

    def __init__(self, statement: str, sentences: list[int]) -> None:
        """Initialises the fact."""
        self.statement, self.sentences = statement, sentences


class _Answer:
    """The model's answer for one passage."""

    def __init__(self, *facts: _Fact) -> None:
        """Initialises the answer."""
        self.facts = list(facts)


class _Client:
    """A model that answers whatever it was told to, and records the ask."""

    model = "ollama/test-model"
    temperature = 0.0

    def __init__(self, answer: _Answer | Exception) -> None:
        """Initialises the client."""
        self._answer, self.asked = answer, {}

    def answer(self, **kwargs: object) -> _Answer:
        """Answers, and records what it was sent."""
        self.asked = kwargs
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def test_the_provenance_names_the_model_and_the_prompt() -> None:
    """Two prompts are two datasets, so the version is recorded per fact."""
    provenance = LlmExtractor(_Client(_Answer())).provenance

    assert provenance.model == "ollama/test-model"
    assert provenance.prompt_version == PROMPT_VERSION
    assert provenance.temperature == 0.0


def test_the_sentences_are_sent_numbered() -> None:
    """The number is what a citation resolves against."""
    client = _Client(_Answer())
    LlmExtractor(client).extract(passage())

    assert "[0] The device weighs 4 kg and runs for 12 hours." in client.asked["user"]
    assert (
        "[1] It arrives in March 2026 from the Hamburg plant." in client.asked["user"]
    )


def test_a_heading_trail_is_fenced_off_and_labelled() -> None:
    """Run together with the excerpt, the model cited it as a sentence."""
    client = _Client(_Answer())
    LlmExtractor(client).extract(passage(section_path="Support > Response times"))

    sent = client.asked["user"]
    assert sent.startswith("Context (not part of the excerpt")
    assert "Support > Response times" in sent
    assert sent.index("Context") < sent.index("Excerpt:")


def test_a_passage_with_no_heading_is_all_excerpt() -> None:
    """Nothing is invented to fill the context in."""
    client = _Client(_Answer())
    LlmExtractor(client).extract(passage())

    assert client.asked["user"].startswith("Excerpt:")


def test_a_returned_fact_becomes_a_candidate() -> None:
    """The statement and the cited numbers, and nothing else."""
    client = _Client(_Answer(_Fact("The device weighs 4 kg.", [0])))
    facts = LlmExtractor(client).extract(passage())

    assert len(facts) == 1
    assert facts[0].statement == "The device weighs 4 kg."
    assert facts[0].sentences == (0,)


@pytest.mark.parametrize(
    "returned",
    [
        _Fact("", [0]),
        _Fact("   ", [0]),
        _Fact("A statement citing nothing.", []),
    ],
)
def test_a_fact_with_no_statement_or_no_citation_is_dropped(returned) -> None:
    """Neither can be checked, so neither is a candidate."""
    assert LlmExtractor(_Client(_Answer(returned))).extract(passage()) == []


def test_a_repeated_citation_is_kept_once_and_in_order() -> None:
    """The same sentence named twice is one citation."""
    client = _Client(_Answer(_Fact("The device weighs 4 kg.", [1, 0, 1])))
    facts = LlmExtractor(client).extract(passage())

    assert facts[0].sentences == (1, 0)


def test_an_empty_answer_is_a_correct_answer() -> None:
    """A sentence with no factual content yields no facts."""
    assert LlmExtractor(_Client(_Answer())).extract(passage()) == []


def test_a_model_that_will_not_answer_fails_the_passage() -> None:
    """The row is failed with the reason, not left mid-extraction."""
    client = _Client(ModelUnavailable("Timeout: took too long"))

    with pytest.raises(ExtractionFailed, match="took too long"):
        LlmExtractor(client).extract(passage())
