"""Publishing the recorded prompts to Phoenix, so a span links to one.

A span already carries the prompt it sent, filled in with that call's
passages. What it cannot show is the TEMPLATE - the prompt with nothing
substituted into it - and `llm.prompt_template.version` on the span names
a version with nothing in Phoenix under it.

This closes that: each recorded prompt becomes a Phoenix prompt, and a
span's version reads against one that can be opened beside the trace.

A published version carries the WHOLE call - the system message, the user
message as its template, the invocation parameters and the response format.
It carried only the first of those, because `PromptVersion(...)` accepts
nothing else; `from_openai` is the constructor that does.

Read from the DATABASE rather than from the code, which is what makes it
work for a version the source has moved past - the same reason the table
exists. It also keeps the stages out of it: a publisher that asked
`question_generation.prompts` would load litellm to read a string.

Here rather than in a package of its own because this is where the
host-side Phoenix client already lives, beside the golden sets. Like
those, it never runs in a container.

**The source is the code.** Phoenix's UI allows an edit, and an edit there
reaches nothing: the pipeline composes from the source, the row records
what it sent, and the next publish writes over whatever the UI did. That
is worth knowing before somebody changes a prompt in a browser and waits
for the questions to change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evaluation.config import Settings
from llm.config import Settings as ModelSettings
from stages.prompts import stored

if TYPE_CHECKING:
    from database.qa_generator import Prompt

log = logging.getLogger(__name__)


def named(service: str, name: str) -> str:
    """What one prompt is called in Phoenix.

    Slugged, because Phoenix takes a name that is a valid identifier and
    these carry spaces, colons and brackets - `phrasing: source` and
    `factoid (spans)`.
    """
    slug = "".join(
        character if character.isalnum() else "-" for character in f"{service}-{name}"
    )
    return "-".join(part for part in slug.split("-") if part).lower()


#: What Phoenix's `reasoning_effort` field will hold. Ollama's own spelling
#: for the setting is `off`, which is not one of these and is dropped by the
#: converter without a word; `aside` reports it instead of losing it.
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})

#: A litellm prefix to the provider Phoenix files the parameters under.
#: Only the five `from_openai` accepts; anything else is an OpenAI-shaped
#: API, which is what litellm makes of it anyway.
_PROVIDERS = {
    "ollama": "OLLAMA",
    "ollama_chat": "OLLAMA",
    "azure": "AZURE_OPENAI",
    "deepseek": "DEEPSEEK",
    "xai": "XAI",
}


def provider(model: str) -> str:
    """Which provider Phoenix files one model's parameters under."""
    return _PROVIDERS.get(model.split("/", 1)[0] if "/" in model else "", "OPENAI")


def parameters(model: ModelSettings) -> dict[str, Any]:
    """The invocation parameters one call is made with, as OpenAI names them.

    Read off the same settings object `llm.client` builds its call from, so
    what Phoenix shows beside a prompt is what the pipeline sent.

    ONLY WHAT PHOENIX WILL HOLD. Its parameter block has fields for
    temperature and reasoning effort and none for `num_ctx` or `timeout`,
    and the converter drops a key it has no field for WITHOUT SAYING SO - so
    sending the whole call here would publish two parameters and quietly
    lose two. `aside` is where the other two go.

    `llm.config` imports no litellm, which is what lets this run host-side
    beside the rest of the publisher. See that module's own note.
    """
    asked: dict[str, Any] = {"temperature": model.temperature}
    if model.thinking in _EFFORTS:
        asked["reasoning_effort"] = model.thinking
    return asked


def aside(model: ModelSettings) -> str:
    """What the call was made with that Phoenix has no field for.

    The window and the patience, and the reasoning effort when it is
    Ollama's `off` rather than one of the six Phoenix knows. Written into
    the description, which is free text, because a parameter shown nowhere
    is one somebody reproducing a run has to go and read the deployment for.
    """
    written = [f"timeout {model.timeout_seconds:g}s"]
    if model.window:
        written.append(f"num_ctx {model.window}")
    if model.thinking and model.thinking not in _EFFORTS:
        written.append(f"reasoning_effort {model.thinking}")
    return ", ".join(written)


def _response_format(row: Prompt) -> dict[str, Any] | None:
    """One row's shape, as an OpenAI `response_format`.

    None for a row recorded before the column existed, which publishes with
    no response format rather than with an empty one.
    """
    if not row.response_schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": row.response_schema.get("title", row.name),
            "schema": row.response_schema,
            "strict": True,
        },
    }


def publish(
    settings: Settings, model: ModelSettings, service: str | None = None
) -> int:
    """Sends every recorded prompt to Phoenix, and reports how many went.

    One Phoenix prompt per (service, name), with a VERSION per recorded
    version - which is Phoenix's own model and this table's, lined up: it
    keeps the history and shows the diff between two.

    BOTH MESSAGES, the parameters and the shape. `PromptVersion(...)` takes
    a message list, a model name and a template format and nothing else, so
    a version built that way carries an EMPTY invocation-parameters block
    and no response format at all - which is what Phoenix was showing.
    `from_openai` is the constructor that takes the whole call, and the
    whole call is what a prompt is.

    `model` is what the prompt is FOR: its identifier names the model
    Phoenix offers it against in the playground, its prefix decides which
    provider's parameter block the version carries, and its settings are
    what goes in that block and in the description beside it.

    Returns:
        How many versions were sent.
    """
    from phoenix.client import Client
    from phoenix.client.types import PromptVersion

    client = Client(
        base_url=settings.base_url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    asked = parameters(model)
    also = aside(model)
    named_provider = provider(model.model)
    rows = sorted(stored(service=service), key=lambda row: (row.service, row.name))
    sent = 0
    for row in rows:
        messages: list[dict[str, str]] = [{"role": "system", "content": row.text}]
        if row.user_text:
            messages.append({"role": "user", "content": row.user_text})
        call: dict[str, Any] = {
            "model": model.model,
            "messages": messages,
            **asked,
        }
        if (shape := _response_format(row)) is not None:
            call["response_format"] = shape
        try:
            client.prompts.create(
                name=named(row.service, row.name),
                prompt_description=(
                    f"{row.service}: {row.name}, at PROMPT_VERSION {row.version} "
                    f"({row.digest}). Also sent: {also}. Composed in the source "
                    f"and recorded by the stage that sends it; an edit here "
                    f"reaches nothing."
                ),
                version=PromptVersion.from_openai(
                    call,  # type: ignore[arg-type] - a plain dict of the same keys
                    # MUSTACHE, so the `{{name}}` in a user template reads as
                    # the variable it is. Safe for the system half: no prompt
                    # in the source carries a double brace, and the single
                    # ones in bridge's worked example are not a substitution.
                    template_format="MUSTACHE",
                    model_provider=named_provider,  # type: ignore[arg-type]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - one bad prompt is not the rest
            log.warning(
                "could not publish %s's %r: %s: %s",
                row.service,
                row.name,
                type(exc).__name__,
                exc,
            )
            continue
        sent += 1
    log.info("published %d of %d prompt version(s) to Phoenix", sent, len(rows))
    return sent
