"""What a published prompt carries beside its text.

Phoenix showed no invocation parameters and no response format for any
prompt here, and both were true of what was sent: `PromptVersion(...)`
accepts a message list, a model name and a template format and NOTHING
ELSE, so every version published through it carried an empty parameter
block and no shape.

`from_openai` takes the whole call. What it does not take it DROPS IN
SILENCE - it has no field for `num_ctx` or `timeout`, and it will not hold
Ollama's `off` for a reasoning effort - so the half of this that matters is
that the publisher sends what Phoenix keeps and reports the rest rather
than watching it disappear.

Nothing here reaches Phoenix. The version is built and read back.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from phoenix.client.types import PromptVersion

from evaluation.prompts import (
    _response_format,
    aside,
    named,
    order,
    parameters,
    provider,
)
from llm.config import Settings as ModelSettings


def settings(model: str = "ollama/gemma3:12b", **overrides) -> ModelSettings:
    """A model's settings as a deployment would resolve them."""
    return ModelSettings(
        **{
            "model": model,
            "base_url": None,
            "ollama_base_url": None,
            "temperature": 0.2,
            "timeout_seconds": 900.0,
            "max_attempts": 3,
            "num_ctx": None,
            "reasoning_effort": None,
            "structured_mode": "json_schema",
            **overrides,
        }
    )


class Row:
    """One recorded prompt, as `stored` hands it over."""

    service = "extraction"
    name = "atomic"
    version = "8"
    digest = "0123456789abcdef"
    text = "ROLE\nYou are a claims analyst."
    user_text = "{{context}}Excerpt:\n{{excerpt}}"
    response_schema: ClassVar[dict | None] = {
        "title": "_Facts",
        "type": "object",
        "properties": {"facts": {"type": "array"}},
    }


def built(model: ModelSettings, row: Row) -> dict:
    """The version the publisher would send, read back as Phoenix stores it."""
    call = {
        "model": model.model,
        "messages": [
            {"role": "system", "content": row.text},
            {"role": "user", "content": row.user_text},
        ],
        **parameters(model),
    }
    if (shape := _response_format(row)) is not None:
        call["response_format"] = shape
    return PromptVersion.from_openai(
        call, template_format="MUSTACHE", model_provider=provider(model.model)
    )._dumps()


def test_a_published_version_carries_both_messages() -> None:
    """One system message with nothing under it is half of what was sent."""
    sent = built(settings(), Row())

    assert [one["role"] for one in sent["template"]["messages"]] == ["system", "user"]


def test_a_published_version_carries_the_temperature_it_was_sent_at() -> None:
    """The block was empty, and an empty block reads as a default."""
    sent = built(settings(), Row())

    assert sent["invocation_parameters"]["ollama"]["temperature"] == 0.2


def test_a_published_version_carries_the_shape_the_answer_came_back_in() -> None:
    """Every call asks for one, and no version said which."""
    sent = built(settings(), Row())

    assert sent["response_format"]["json_schema"]["schema"] == Row.response_schema


def test_the_user_template_is_published_as_a_template() -> None:
    """MUSTACHE, so `{{excerpt}}` is a variable and not four characters."""
    sent = built(settings(), Row())

    assert sent["template_format"] == "MUSTACHE"
    assert "{{excerpt}}" in sent["template"]["messages"][1]["content"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ollama/gemma3:12b", "OLLAMA"),
        ("ollama_chat/gemma3:12b", "OLLAMA"),
        ("azure/gpt-4.1", "AZURE_OPENAI"),
        ("gpt-4.1", "OPENAI"),
        ("gemini/gemini-2.5-pro", "OPENAI"),
    ],
)
def test_the_provider_comes_off_the_litellm_prefix(model: str, expected: str) -> None:
    """The parameter block is filed under a provider, and it names one."""
    assert provider(model) == expected


def test_a_reasoning_effort_phoenix_will_hold_is_sent_as_a_parameter() -> None:
    """`low` is one of the six its field takes."""
    asked = parameters(settings(reasoning_effort="low"))

    assert asked["reasoning_effort"] == "low"


def test_ollamas_off_is_reported_rather_than_dropped() -> None:
    """The converter takes six values and `off` is not one of them.

    Sent as a parameter it vanishes without a word, and the published
    prompt then says nothing about the setting worth 414.7 seconds against
    12.6 on this corpus's writer prompt. So it goes in the description.
    """
    model = settings()
    assert model.thinking == "off"

    assert "reasoning_effort" not in parameters(model)
    assert "reasoning_effort off" in aside(model)


def test_what_phoenix_has_no_field_for_is_reported_beside_it() -> None:
    """`num_ctx` and `timeout` are dropped by the converter in silence."""
    written = aside(settings(num_ctx=6144))

    assert "num_ctx 6144" in written
    assert "timeout 900s" in written


def test_a_row_recorded_before_the_shape_was_stored_publishes_without_one() -> None:
    """NULL is a row from before the column, not a call that asked nothing."""

    class Old(Row):
        response_schema = None

    assert _response_format(Old()) is None
    assert "response_format" not in built(settings(), Old())


def test_a_prompt_name_survives_the_punctuation_in_it() -> None:
    """Phoenix takes an identifier, and these carry colons and brackets."""
    assert named("questions", "phrasing: source") == "questions-phrasing-source"
    assert named("questions", "factoid (spans)") == "questions-factoid-spans"


def test_versions_are_published_oldest_first() -> None:
    """Phoenix shows the version created LAST as the prompt's current one.

    `stored` returns newest first, so publishing in the order it hands them
    over makes the OLDEST recorded version the current one - and a prompt
    opened beside a trace is then the one the source has moved past, which
    is the opposite of what the table is for.
    """
    assert sorted(["9", "8", "2"], key=order) == ["2", "8", "9"]


def test_the_tenth_version_is_newer_than_the_ninth() -> None:
    """These are compared as text, where `10` sorts before `9`."""
    assert sorted(["9", "10", "8"], key=order) == ["8", "9", "10"]
