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
