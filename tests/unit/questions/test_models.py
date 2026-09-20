"""Which model writes a question, and which one checks it.

Two gates may only be applied by a model that did not write the answer: the
phrasing opinion, which can reject, and the entailment pass, which can only
accept. A writer running either over its own work waves through whatever its
own recall missed.

What decides is whether the two models came out different, not whether a
setting was set. Once the writer can be overridden too, those are different
questions: naming one model in both settings is the same problem as naming
neither, and naming only the writer leaves a verifier that is independent
without QUESTIONS_VERIFIER_MODEL saying so.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm.config import Settings as ModelSettings
from question_generation.factory import models

SHARED = ModelSettings(
    model="ollama/shared",
    base_url=None,
    structured_mode="JSON_SCHEMA",
    temperature=0.0,
    timeout_seconds=900.0,
    max_attempts=3,
    num_ctx=None,
    reasoning_effort=None,
)


def settings(writer: str | None, verifier: str | None):
    """Question settings naming these two models and nothing else of note."""
    from question_generation.config import Settings

    loaded = Settings.load()
    return replace(loaded, model=writer, verifier_model=verifier)


@pytest.mark.parametrize(
    ("writer", "verifier", "writes", "checks", "independent"),
    [
        # Neither named: one model does both, which is the state the worker
        # warns about on every start.
        (None, None, "ollama/shared", "ollama/shared", False),
        # Only the verifier named, which is how this worked before a writer
        # could be overridden at all.
        (None, "ollama/checker", "ollama/shared", "ollama/checker", True),
        # Only the writer named. The verifier is still the shared model, and
        # it is still a different one, so the gates apply.
        ("ollama/writer", None, "ollama/writer", "ollama/shared", True),
        # Both named, differently: the arrangement this is all for.
        ("ollama/writer", "ollama/checker", "ollama/writer", "ollama/checker", True),
        # Both named the same. A setting was set, twice, and there is still
        # only one model marking its own work.
        ("ollama/same", "ollama/same", "ollama/same", "ollama/same", False),
        # The writer overridden to the shared model by name, and no verifier.
        ("ollama/shared", None, "ollama/shared", "ollama/shared", False),
    ],
)
def test_which_model_writes_and_which_checks(
    writer, verifier, writes, checks, independent
) -> None:
    """Every combination of the two settings, and what it comes out as."""
    written, checked = models(settings(writer, verifier), SHARED)

    assert written.model == writes
    assert checked.model == checks
    assert (written.model != checked.model) is independent


def test_both_models_are_served_where_the_deployment_says() -> None:
    """A stage names a model, not an address: one runtime serves both."""
    written, checked = models(
        settings("ollama/writer", "ollama/checker"),
        replace(SHARED, base_url="http://served:11434"),
    )

    assert written.base_url == "http://served:11434"
    assert checked.base_url == "http://served:11434"


def test_both_models_are_held_to_the_same_patience() -> None:
    """The lease is derived from these, so one stage may not stretch them."""
    written, checked = models(settings("ollama/writer", "ollama/checker"), SHARED)

    assert written.timeout_seconds == SHARED.timeout_seconds
    assert checked.max_attempts == SHARED.max_attempts
    assert written.lease == SHARED.lease


# ── What the writer is told about where a fact came from ──────────────────


def test_one_document_is_shown_the_way_it_always_was() -> None:
    """Most samples are one document, and a prompt that moves for no reason
    cannot be compared against the last run."""
    facts = group(source(1, document="a", passage_id=1, section_path="Support"))

    assert facts.context == (("Under: Support\n", facts.facts[0].passage_text),)


def test_two_documents_are_told_apart() -> None:
    """The defect: a writer that cannot tell two sources apart states one's
    claim as the other's, and two syllabi with different chapter counts came
    back as one question answered "once three, once eight"."""
    facts = group(
        source(1, document="a", passage_id=1, section_path="Chapters"),
        source(2, document="b", passage_id=2, section_path="Chapters"),
    )

    headings = [heading for heading, _ in facts.context]
    assert headings == [
        "Document A, under: Chapters\n",
        "Document B, under: Chapters\n",
    ]


def test_a_letter_belongs_to_a_document_not_to_a_passage() -> None:
    """Two passages of one document are one source, and a third document is a
    third letter."""
    facts = group(
        source(1, document="a", passage_id=1),
        source(2, document="a", passage_id=2),
        source(3, document="b", passage_id=3),
    )

    letters = [heading.split(",")[0] for heading, _ in facts.context]
    assert letters == ["Document A", "Document A", "Document B"]


def test_a_document_with_no_heading_is_still_named() -> None:
    """The label is what says two facts are two sources; a missing heading
    does not take it away."""
    facts = group(
        source(1, document="a", passage_id=1, section_path=None),
        source(2, document="b", passage_id=2, section_path=None),
    )

    assert [heading for heading, _ in facts.context] == [
        "Document A\n",
        "Document B\n",
    ]
