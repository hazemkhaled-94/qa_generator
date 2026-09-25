"""The judging templates, and the two things about them that must not drift.

Phoenix is a shared vocabulary. A `hallucination` annotation posted by this
pipeline sits in the same Evaluations view as one posted by anything else
built on arize-phoenix-evals, and a reader sorts and charts them together.
So the name, the labels, the scores and the optimisation direction are that
package's and are pinned here against it - not against a copy written down,
against the package itself, so an upgrade that renamed a label fails this
file rather than producing a column that looks right and sorts backwards.

The prompts are ours. What is checked about those is only that every
placeholder a template carries is one the repository actually supplies: a
template variable nobody fills reaches the model as the braces themselves,
with no error anywhere and an answer that looks fine.
"""

from __future__ import annotations

import re

import pytest

from assessment.templates import (
    PROMPT_VERSION,
    TEMPLATES,
    direction_of,
    metrics_of,
)
from database.qa_generator import ArtifactKind, JudgeMetric

#: Every template, flattened, so one failure names one judgement.
EVERY = [one for group in TEMPLATES.values() for one in group]

#: A `{{name}}` in a template.
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

#: What phoenix-evals itself calls each of these, and how it scores them.
#: Read off the installed package rather than written out, which is the
#: whole point: this fails when their vocabulary moves, not when ours does.
PHOENIX = pytest.importorskip(
    "phoenix.evals.default_templates", reason="arize-phoenix-evals is not installed"
)


def test_every_kind_is_judged() -> None:
    """A kind with no template is a kind the queue would claim and drop."""
    assert set(TEMPLATES) == {
        ArtifactKind.FACT,
        ArtifactKind.TOPIC,
        ArtifactKind.QUESTION,
    }
    assert all(TEMPLATES.values()), "a kind with no judgement is not judged"


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_a_metric_is_one_the_database_accepts(one) -> None:
    """A metric outside the CHECK constraint fails at write, not at ask."""
    assert one.metric in set(JudgeMetric)


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_every_placeholder_is_declared(one) -> None:
    """A placeholder nobody fills reaches the model as its own braces.

    The static check over the other prompt modules counts each name twice,
    once in the template and once at its `.replace`. This module renders
    generically, so the pairing is between the template and the `fields`
    it declares - and that is what `rendered` refuses to go without.
    """
    assert set(PLACEHOLDER.findall(one.user)) == set(one.fields)


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_rendering_without_a_field_is_refused(one) -> None:
    """Rather than sending the braces and getting a plausible answer back."""
    with pytest.raises(KeyError, match=one.fields[0]):
        one.rendered()


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_rendering_leaves_no_placeholder_behind(one) -> None:
    """Every declared field is substituted, not just checked for."""
    filled = one.rendered(**{name: f"<{name}>" for name in one.fields})

    assert not PLACEHOLDER.findall(filled)
    assert all(f"<{name}>" in filled for name in one.fields)


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_the_labels_are_the_ones_phoenix_uses(one) -> None:
    """The rails, against phoenix-evals' own.

    Its `EvalCriteria` holds the rails for the four metrics borrowed here.
    A label of ours outside that set is a Phoenix column that looks like
    the one a reader knows and holds something else.
    """
    criteria = {
        JudgeMetric.HALLUCINATION: "HALLUCINATION",
        JudgeMetric.RELEVANCE: "RELEVANCE",
        JudgeMetric.QA_CORRECTNESS: "QA",
        JudgeMetric.SUMMARIZATION: "SUMMARIZATION",
    }[one.metric]
    theirs = set(PHOENIX.EvalCriteria[criteria].value.rails)

    assert set(one.labels) == theirs, f"{one.metric} is not phoenix-evals' {criteria}"
    assert one.good in theirs


@pytest.mark.parametrize("one", EVERY, ids=lambda one: f"{one.kind}-{one.metric}")
def test_a_score_is_zero_or_one_and_the_shape_is_closed(one) -> None:
    """The column is CHECKed between 0 and 1, and the answer is a Literal.

    A `Literal` of the rails is what makes an answer outside them a schema
    failure instructor retries, rather than a string nothing maps to a
    score.
    """
    assert set(one.scores.values()) == {0.0, 1.0}
    assert one.scores[one.good] in (0.0, 1.0)

    field = one.shape().model_fields["label"]

    assert set(getattr(field.annotation, "__args__", ())) == set(one.labels)


def test_hallucination_is_scored_the_way_phoenix_scores_it() -> None:
    """The one metric here whose good label scores ZERO.

    Its optimisation direction is minimise, so `hallucinated` is 1.0 and
    `factual` is 0.0 - the opposite of the other three. This is the single
    most confusable thing in this package: a reader summing scores across
    metrics would be adding a hallucination rate to a relevance rate, which
    is why `approved` is stored per metric rather than derived from the
    score.
    """
    one = TEMPLATES[ArtifactKind.FACT][0]

    assert one.metric == JudgeMetric.HALLUCINATION
    assert one.good == "factual"
    assert one.scores["factual"] == 0.0
    assert one.scores["hallucinated"] == 1.0
    assert direction_of(JudgeMetric.HALLUCINATION) == "minimize"


@pytest.mark.parametrize(
    "metric",
    [JudgeMetric.RELEVANCE, JudgeMetric.QA_CORRECTNESS, JudgeMetric.SUMMARIZATION],
)
def test_every_other_metric_is_maximised(metric: str) -> None:
    """So a mean read as "how much the judge backed" is right for these."""
    assert direction_of(metric) == "maximize"


def test_a_shape_is_named_after_its_metric() -> None:
    """The shape's name is what reaches the span and the log line.

    `tag.tags` in Phoenix and `llm.shape` in Grafana are both the class
    name, so seven judgements sharing one would be seven calls nothing
    could tell apart.
    """
    names = {one.shape().__name__ for one in EVERY}

    assert len(names) == len({one.metric for one in EVERY})
    assert "_Hallucination" in names


def test_a_question_is_judged_on_more_than_a_fact() -> None:
    """Three judgements against two, and the extra one is the answer.

    A fact is asked whether its evidence supports it and whether the
    evidence is about the right thing. A question is asked both of those
    about its ANSWER, plus whether the answer answers the question - which
    is the judgement that has no counterpart upstream.
    """
    assert metrics_of(ArtifactKind.QUESTION) == (
        JudgeMetric.HALLUCINATION,
        JudgeMetric.QA_CORRECTNESS,
        JudgeMetric.RELEVANCE,
    )
    assert metrics_of(ArtifactKind.FACT) == (
        JudgeMetric.HALLUCINATION,
        JudgeMetric.RELEVANCE,
    )


def test_a_template_says_the_material_may_be_in_any_language() -> None:
    """The German half is why this phase cannot be a gate.

    `evaluation/README.md` records a judgement answered 7/7 in English and
    6/12 in German. Every template says so explicitly, because a judge
    that starts reasoning about the language rather than the artefact is
    the measured failure and not a hypothetical one.
    """
    for one in EVERY:
        assert "any language" in one.system, one.metric


def test_the_version_is_a_version() -> None:
    """Recorded on every row, so two template versions are two judgements."""
    assert PROMPT_VERSION and PROMPT_VERSION.strip() == PROMPT_VERSION
