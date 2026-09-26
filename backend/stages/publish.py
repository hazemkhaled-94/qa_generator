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

**Called by `record`, so a run publishes what it just wrote.** This used
to be host-side only, beside the golden sets, and a deployment therefore
had prompts in its table and none in Phoenix until somebody remembered
`make prompts-publish` - which is every deployment nobody remembered. It
is here instead, beside the write, and it goes wherever that goes: a host
command, a worker container, the stage a Dagster run set going.

That placement is also what makes it idempotent for nothing. `record`
hands over the prompts it actually WROTE, which is none on every start-up
after the first, and `client.prompts.create` posts a new Phoenix version
every time it is called. Publishing everything on every run would pile up
an identical version per prompt per run, which is the growth
`telemetry/projects.py` exists to clean up in the other store.

`publish_stored` is the full republish the make target still runs, for the
case the table cannot detect: a Phoenix that was wiped while the rows
stayed.

Reads what the stage RECORDED rather than the code, which is what makes
the republish work for a version the source has moved past - the same
reason the table exists. It also keeps the stages out of each other: a
publisher that asked `question_generation.prompts` would load litellm to
read a string.

**The source is the code.** Phoenix's UI allows an edit, and an edit there
reaches nothing: the pipeline composes from the source, the row records
what it sent, and the next publish writes over whatever the UI did. That
is worth knowing before somebody changes a prompt in a browser and waits
for the questions to change.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from llm.config import Settings as ModelSettings
from stages.prompts import Composed, stored
from telemetry.evaluations import ENDPOINT, client

if TYPE_CHECKING:
    from database.qa_generator import Prompt

log = logging.getLogger(__name__)


def composed_from(row: Prompt) -> Composed:
    """One stored row as the thing a stage would have handed over.

    So there is one shape to publish rather than two. A protocol covering
    both does not work: a mapped column is `Mapped[str]` to a type checker
    until it is read off an instance, so `Prompt` matches no protocol
    spelled in the types it actually returns.

    The digest is recomputed rather than read across, and is the same value
    - `_write` writes the text and its digest together.
    """
    return Composed(
        row.name,
        row.version,
        row.text,
        row.user_text,
        row.response_schema,
        row.model,
    )


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
    the VERSION's description, which is free text and is written fresh on
    every publish, because a parameter shown nowhere is one somebody
    reproducing a run has to go and read the deployment for.
    """
    written = [f"timeout {model.timeout_seconds:g}s"]
    if model.window:
        written.append(f"num_ctx {model.window}")
    if model.thinking and model.thinking not in _EFFORTS:
        written.append(f"reasoning_effort {model.thinking}")
    return ", ".join(written)


def order(version: str) -> tuple[int, str]:
    """Where one version goes in the publish order, OLDEST FIRST.

    Phoenix keeps the version created LAST as the prompt's current one, and
    `stored` hands its rows back newest first - so publishing in the order
    they arrive makes the oldest recorded version the one Phoenix shows,
    and the prompt somebody opens beside a trace is the one the source has
    moved past. This is what turns that round.

    Numerically where the version is a number, because these are counted
    and compared as text: `10` sorts before `9` and the newest prompt would
    be buried again the tenth time one is bumped.
    """
    return (int(version), "") if version.isdigit() else (0, version)


def _response_format(row: Composed) -> dict[str, Any] | None:
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


def publish(service: str, composed: Iterable[Composed], model: ModelSettings) -> int:
    """Sends some prompts to Phoenix, and reports how many went.

    Takes what to publish rather than reading the table, because the caller
    that matters knows: `record` hands over the prompts it just WROTE, and
    on every start-up after the first that is none. See this module's own
    note on why publishing everything each time is the wrong shape.

    One Phoenix prompt per (service, name), with a VERSION per recorded
    version - which is Phoenix's own model and this table's, lined up: it
    keeps the history and shows the diff between two.

    OLDEST FIRST, so the newest recorded version is the one created last
    and Phoenix shows it as current. See `order`.

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

    PER PROMPT, not per service, through `Composed.model`. One service can
    send to more than one: question generation writes with QUESTIONS_MODEL,
    judges wording with QUESTIONS_PHRASING_MODEL and checks with
    QUESTIONS_VERIFIER_MODEL, and the four verifier prompts opened against
    the writer's model are four prompts replayed in the playground against
    a model that never sees them. `overridden` returns `model` itself where
    a prompt names none, so most rows resolve to exactly what was passed.

    Never raises. A stage that cannot publish still sent the prompt and
    still recorded it; what is lost is the copy Phoenix shows beside the
    trace, and that must not fail a run.

    Returns:
        How many versions were sent.
    """
    rows = sorted(composed, key=lambda row: (row.name, order(row.version)))
    if not rows:
        return 0
    # No address, no publish - the same bargain `Evaluations` makes, and
    # here it is a guard rather than a courtesy: the Phoenix client's own
    # default is localhost:6006, so a process that was never told where
    # Phoenix is would otherwise write to whatever happens to answer there.
    # A test suite on a developer's laptop is exactly that process.
    base_url = os.getenv(ENDPOINT)
    if not base_url:
        log.info(
            "%s is unset, so %s's prompts stay in the table and are not published",
            ENDPOINT,
            service,
        )
        return 0
    connected = client(base_url)
    if connected is None:
        log.info(
            "arize-phoenix-client is not installed, so %s's prompts are in the "
            "table and not in Phoenix. `make prompts-publish` sends them.",
            service,
        )
        return 0

    from phoenix.client.types import PromptVersion

    sent, refused = 0, 0
    for row in rows:
        # Resolved per row, because a prompt may name a model of its own.
        # `overridden(None)` is `model`, so this is a lookup and not a
        # branch, and the three calls below are string work either way.
        asked_of = model.overridden(row.model)
        asked = parameters(asked_of)
        also = aside(asked_of)
        named_provider = provider(asked_of.model)
        messages: list[dict[str, str]] = [{"role": "system", "content": row.text}]
        if row.user_text:
            messages.append({"role": "user", "content": row.user_text})
        call: dict[str, Any] = {
            "model": asked_of.model,
            "messages": messages,
            **asked,
        }
        if (shape := _response_format(row)) is not None:
            call["response_format"] = shape
        try:
            connected.prompts.create(
                name=named(service, row.name),
                # Nothing that goes stale. Phoenix sets this when it CREATES
                # the prompt and ignores it on every version after, so a
                # version or a digest written here is frozen at whichever
                # one was published first and then quietly wrong.
                prompt_description=(
                    f"{service}: {row.name}. Composed in the source and "
                    f"recorded by the stage that sends it; an edit here "
                    f"reaches nothing."
                ),
                version=PromptVersion.from_openai(
                    call,  # type: ignore[arg-type] - a plain dict of the same keys
                    # What DOES move, on the version, which is written fresh
                    # every publish: the version it came from, its digest,
                    # and the parameters Phoenix has no field for.
                    description=(
                        f"PROMPT_VERSION {row.version} ({row.digest}). "
                        f"Also sent: {also}."
                    ),
                    # MUSTACHE, so the `{{name}}` in a user template reads as
                    # the variable it is. Safe for the system half: no prompt
                    # in the source carries a double brace, and the single
                    # ones in bridge's worked example are not a substitution.
                    template_format="MUSTACHE",
                    model_provider=named_provider,  # type: ignore[arg-type]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - one bad prompt is not the rest
            # The first in full and the rest counted. A Phoenix that is down
            # refuses every prompt, and a stage recording thirty-one of them
            # would otherwise open its log with thirty-one copies of one
            # sentence - the failure `stages/cli.py` already quietens for an
            # unreachable model.
            if not refused:
                log.warning(
                    "could not publish %s's %r, so it is in the table and not "
                    "in Phoenix: %s: %s",
                    service,
                    row.name,
                    type(exc).__name__,
                    exc,
                )
            refused += 1
            continue
        sent += 1
    if refused:
        log.warning(
            "%d of %s's %d prompt(s) did not reach Phoenix", refused, service, len(rows)
        )
    elif sent:
        log.info("published %d %s prompt version(s) to Phoenix", sent, service)
    return sent


def publish_stored(model: ModelSettings, service: str | None = None) -> int:
    """Republishes every prompt the table holds, and reports how many went.

    What `make prompts-publish` runs, and the one case `record` cannot
    cover: a Phoenix wiped while the rows stayed. The table then matches
    what the stages send and `record` writes nothing, so nothing would
    republish without asking for it.

    Also the way in for a version the SOURCE has moved past, which is the
    reason the table is read here rather than the code.

    **A stage's own model choices survive it**, because the row records
    them: the judge's templates go back against ASSESSMENT_JUDGE_MODEL and
    question generation's verifier prompts against
    QUESTIONS_VERIFIER_MODEL, exactly as the stage published them.
    `model` is the fallback for a row that names none, which is every row
    of a stage that calls the shared one.
    """
    held: dict[str, list[Composed]] = {}
    for row in stored(service=service):
        held.setdefault(row.service, []).append(composed_from(row))
    return sum(publish(one, group, model) for one, group in held.items())
