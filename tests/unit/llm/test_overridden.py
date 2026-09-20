"""One stage calling a different model from the rest.

The model is one deployment's choice and the stages do not all want the same
one: reading a passage for its claims, naming a topic from its terms and
writing a question are three jobs, and the cheapest model that does one is
not the cheapest that does another.

Only the model is replaced. Where it is served, how it is made to answer in
a shape and how patient to be stay the deployment's, because a stage that
could set its own timeout would be a stage whose lease nobody could derive.
"""

from __future__ import annotations

import pytest

from llm.config import Settings


def shared(**overrides) -> Settings:
    """The settings every stage shares, for a model nothing calls."""
    return Settings(
        **{
            "model": "ollama/shared",
            "base_url": "http://served:11434",
            "structured_mode": "JSON_SCHEMA",
            "temperature": 0.0,
            "timeout_seconds": 30.0,
            "max_attempts": 2,
            "num_ctx": 8192,
            "reasoning_effort": None,
            **overrides,
        }
    )


def test_a_named_model_replaces_the_shared_one() -> None:
    """What a stage's own setting buys."""
    assert shared().overridden("ollama/other").model == "ollama/other"


@pytest.mark.parametrize("absent", [None, ""])
def test_no_name_keeps_the_shared_model(absent) -> None:
    """Every override is optional, and absent is the usual case."""
    assert shared().overridden(absent).model == "ollama/shared"


def test_nothing_but_the_model_changes() -> None:
    """A stage names a model; it does not get its own deployment."""
    before = shared()
    after = before.overridden("ollama/other")

    assert after.base_url == before.base_url
    assert after.structured_mode == before.structured_mode
    assert after.temperature == before.temperature
    assert after.timeout_seconds == before.timeout_seconds
    assert after.max_attempts == before.max_attempts
    assert after.num_ctx == before.num_ctx
    assert after.reasoning_effort == before.reasoning_effort


def test_the_lease_is_unchanged_by_an_override() -> None:
    """Derived from the timeout and the attempts, neither of which moved.

    A stage that could lengthen its own lease could make a row unclaimable
    for longer than the queue expects, with nothing but time to undo it.
    """
    before = shared()

    assert before.overridden("ollama/other").lease == before.lease


def test_the_shared_settings_are_left_alone() -> None:
    """Frozen, and shared by every stage: one override must not reach another."""
    before = shared()
    before.overridden("ollama/other")

    assert before.model == "ollama/shared"


def test_keeping_the_shared_model_returns_the_same_settings() -> None:
    """No copy when nothing changed, so a caller needs no branch."""
    before = shared()

    assert before.overridden(None) is before


def test_a_self_hosted_override_is_sent_where_it_is_served() -> None:
    """Litellm picks the provider off the prefix, so the address follows it.

    A stage naming `ollama_chat/...` while LLM_BASE_URL points at Azure
    sends an Ollama request to Azure, which answers 404. That happened: a
    whole topic's phrasing judgements abstained because every one of them
    was posted to the wrong host.
    """
    settings = shared(
        base_url="https://example.openai.azure.com",
        ollama_base_url="http://localhost:11434",
    )

    moved = settings.overridden("ollama_chat/gemma4:12b")

    assert moved.model == "ollama_chat/gemma4:12b"
    assert moved.base_url == "http://localhost:11434"


def test_a_hosted_override_keeps_the_shared_address() -> None:
    """Only a self-hosted prefix moves; everything else is one provider."""
    settings = shared(
        base_url="https://example.openai.azure.com",
        ollama_base_url="http://localhost:11434",
    )

    assert settings.overridden("azure/gpt-4.1").base_url == (
        "https://example.openai.azure.com"
    )


def test_a_self_hosted_override_with_nowhere_to_send_it_is_left_alone() -> None:
    """A deployment that never named a runtime gets the old behaviour."""
    settings = shared(base_url="http://one-place", ollama_base_url=None)

    assert settings.overridden("ollama_chat/other").base_url == "http://one-place"
