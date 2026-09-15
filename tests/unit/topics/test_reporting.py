"""The values the service reports, and what the page reads off them.

`stale` is the one derived answer on this boundary, and the page's Model
health table is drawn from it.
"""

from __future__ import annotations

import pytest

from topic_modelling.models import FittedTopic, LanguageFit, TopicFit


def language(**changed) -> LanguageFit:
    """One language whose topics still describe the corpus."""
    return LanguageFit(
        **{
            "language": "de",
            "topics": 12,
            "corpus_passages": 100,
            "corpus_vocabulary": 800,
            "passages_without_topics": 0,
            "fitted_at": "2026-09-14T10:00:00+00:00",
            "live_passages": 100,
            "memberships": 400,
            **changed,
        }
    )


def test_a_model_fitted_over_the_passages_the_corpus_holds_is_not_stale() -> None:
    """Nothing to do."""
    assert language().stale is False


def test_a_language_with_no_topics_is_not_stale() -> None:
    """There is nothing to be behind the corpus."""
    assert language(topics=0, memberships=0, live_passages=0).stale is False


def test_a_model_whose_memberships_are_gone_is_stale() -> None:
    """Re-chunking deleted the passages and the memberships went with them."""
    assert language(memberships=0).stale is True


def test_a_corpus_that_has_grown_leaves_the_model_stale() -> None:
    """The topics describe the corpus as it was."""
    assert language(live_passages=140).stale is True


def test_a_corpus_that_has_shrunk_leaves_the_model_stale() -> None:
    """A deleted document moves every topic, not one."""
    assert language(live_passages=60).stale is True


def test_a_fit_that_recorded_no_passage_count_reads_as_stale() -> None:
    """A NULL cannot equal the live count, so it is not claimed to."""
    assert language(corpus_passages=None).stale is True


def test_the_model_is_stale_when_any_one_language_is() -> None:
    """A fit is all-or-nothing, so one stale language stales the model."""
    model = TopicFit(
        status="modelled",
        error=None,
        requested_at=None,
        topics=24,
        languages=[language(), language(language="en", memberships=0)],
    )

    assert model.stale is True


def test_a_model_of_no_languages_is_not_stale() -> None:
    """A new deployment, which the page has to render."""
    model = TopicFit(status=None, error=None, requested_at=None, topics=0, languages=[])

    assert model.stale is False


@pytest.mark.parametrize(
    ("labels", "counted"),
    [([], 0), ([None], 0), (["Shipping"], 1), (["Shipping", None, "Upkeep"], 2)],
)
def test_a_fit_counts_the_topics_that_hold_a_label(labels, counted) -> None:
    """Whatever named them: the log line reports it after naming."""
    from topic_modelling.models import Fitting, TopicSpace

    empty = TopicSpace(
        topic_term=[], doc_topic=[], doc_lengths=[], vocabulary=[], term_frequency=[]
    )
    fitting = Fitting(
        language="de",
        topics=[
            FittedTopic(index, ["a"], label=label) for index, label in enumerate(labels)
        ],
        weights=[],
        passages=1,
        without_topics=0,
        vocabulary=1,
        space=empty,
    )

    assert fitting.labelled == counted
