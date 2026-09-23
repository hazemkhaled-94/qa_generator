"""Publishing the recorded prompts to Phoenix, so a span links to one.

A span already carries the prompt it sent, filled in with that call's
passages. What it cannot show is the TEMPLATE - the prompt with nothing
substituted into it - and `llm.prompt_template.version` on the span names
a version with nothing in Phoenix under it.

This closes that: each recorded prompt becomes a Phoenix prompt, and a
span's version reads against one that can be opened beside the trace.

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

from evaluation.config import Settings
from stages.prompts import stored

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


def publish(settings: Settings, model_name: str, service: str | None = None) -> int:
    """Sends every recorded prompt to Phoenix, and reports how many went.

    One Phoenix prompt per (service, name), with a VERSION per recorded
    version - which is Phoenix's own model and this table's, lined up: it
    keeps the history and shows the diff between two.

    `model_name` is what the prompt is FOR, which Phoenix requires on a
    version and uses to offer it in the playground. The configured model,
    so a prompt opened there is replayed against the one that sent it.

    Returns:
        How many versions were sent.
    """
    from phoenix.client import Client
    from phoenix.client.types import PromptVersion

    client = Client(
        base_url=settings.base_url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    rows = sorted(stored(service=service), key=lambda row: (row.service, row.name))
    sent = 0
    for row in rows:
        try:
            client.prompts.create(
                name=named(row.service, row.name),
                prompt_description=(
                    f"{row.service}: {row.name}, at PROMPT_VERSION {row.version} "
                    f"({row.digest}). Composed in the source and recorded by the "
                    f"stage that sends it; an edit here reaches nothing."
                ),
                version=PromptVersion(
                    [{"role": "system", "content": row.text}],
                    model_name=model_name,
                    # NONE, not MUSTACHE: these prompts are already
                    # composed and their worked examples carry braces.
                    # Left to a template format, Phoenix reads those as
                    # variables and the prompt it shows is not the one
                    # that was sent.
                    template_format="NONE",
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
