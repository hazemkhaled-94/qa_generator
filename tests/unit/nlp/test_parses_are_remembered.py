"""One text read several ways is parsed once.

The gates in `question_generation` are described as free because none of
them calls a model. Each of them still ran a full spaCy pipeline, and none
knew another had already parsed the same string: one candidate through
`QuestionChecker.check` parsed its question six times, its answer seven,
and its cited passages joined together up to three - and that last one is a
couple of thousand characters.
"""

from __future__ import annotations

import pytest

from nlp import analysis

pytestmark = pytest.mark.nlp

QUESTION = "How long is allowed for answering a standard support request?"
ANSWER = "48 hours from the confirmation given by the site manager"


@pytest.fixture(autouse=True)
def forgotten() -> None:
    """Starts each test with nothing remembered."""
    analysis._read.cache_clear()


def test_the_same_text_read_twice_is_parsed_once() -> None:
    """Which is what makes the free gates actually free."""
    analysis.content(QUESTION, "en")
    analysis.claim(QUESTION, "en")
    analysis.pointing(QUESTION, "en")

    assert analysis._read.cache_info().misses == 1


def test_two_texts_are_two_parses() -> None:
    """The cache is keyed on the text, not shared across them."""
    analysis.content(QUESTION, "en")
    analysis.content(ANSWER, "en")

    assert analysis._read.cache_info().misses == 2


def test_one_text_in_two_languages_is_two_parses() -> None:
    """A German pipeline and an English one do not agree about a string.

    Sharing a Doc between them would hand one language's tagset to the
    other's rules, which is the whole reason `pipeline` takes a language.
    """
    analysis.content(ANSWER, "en")
    analysis.content(ANSWER, "de")

    assert analysis._read.cache_info().misses == 2


def test_what_the_readings_say_does_not_change() -> None:
    """A remembered Doc has to answer what a fresh one does.

    Every reader here is read-only, which is what makes sharing one safe;
    this is the check that none of them has started writing to a token.
    """
    first = (
        analysis.content(QUESTION, "en"),
        analysis.claim(QUESTION, "en"),
        analysis.phrases(QUESTION, "en"),
        analysis.interrogatives(QUESTION, "en"),
        analysis.vocabulary(QUESTION, "en"),
    )
    analysis._read.cache_clear()
    again = (
        analysis.content(QUESTION, "en"),
        analysis.claim(QUESTION, "en"),
        analysis.phrases(QUESTION, "en"),
        analysis.interrogatives(QUESTION, "en"),
        analysis.vocabulary(QUESTION, "en"),
    )

    assert first == again
