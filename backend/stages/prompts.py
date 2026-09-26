"""Recording the prompts a stage sends, under the version that composed them.

`facts.prompt_version` and `questions.prompt_version` name a version and
nothing in the deployment resolved it: the span carrying the prompt belongs
to one Phoenix project with its own retention, and the row outlives it. A
dataset exported six months later pointed at version 5 and could not say
what version 5 asked for.

Every stage writes its OWN prompts and no other stage's, because no backend
service imports another. What is shared is this: the shape of a declaration
and the write.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.qa_generator import Prompt, sessions

if TYPE_CHECKING:
    # Behind TYPE_CHECKING so the api, which imports `stored` from here to
    # serve GET /prompts, keeps the import graph
    # `tests/static/test_api_stays_light.py` reads.
    from llm.config import Settings as ModelSettings

log = logging.getLogger(__name__)


def digest(text: str) -> str:
    """The first sixteen hex characters of one prompt's SHA-256.

    Spelled as `tests/static/test_prompts_pinned.py` spells it, so a stored
    row and that file's pin can be compared without converting either.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Composed:
    """One prompt as its stage sends it.

    Attributes:
        name: What it is, within its service. A question type, `perturb`,
            one of the phrasing judgements, one of the verifier's four.
        version: The PROMPT_VERSION its own module declares. Per module
            rather than per stage: question generation sends prompts from
            three, and they are bumped separately because they change for
            different reasons.
        text: The SYSTEM prompt as composed, which is what the model is
            given as its instructions.
        user_text: The USER message as its template, with `{{name}}` where
            the call fills a passage, a fact or a question in. Every call
            sends both halves; this is the one the row used not to carry.
        response_schema: The JSON schema the answer has to come back in,
            taken from the Pydantic shape the call asks for.
        model: Which model this one is sent to, where the stage names
            another instead of its own. A NAME rather than settings,
            because a name is what a deployment overrides and the rest
            follows from it - `Settings.overridden` moves the address with
            the provider, and the publisher asks it. Question generation is
            why this exists: its writer, its three phrasing judgements and
            its verifier's four are three model choices inside one service,
            so a model per service would publish four of them against a
            model that never sees them.

            Recorded, so `make prompts-publish` can put the same prompt
            back against the same model. Changing it rewrites the row and
            republishes, and is NOT reported as drift: repointing a stage
            at another model changes nothing about what the prompt asks.
    """

    name: str
    version: str
    text: str
    user_text: str = ""
    response_schema: dict[str, Any] | None = None
    model: str | None = None

    @property
    def digest(self) -> str:
        """This text's digest.

        Over the SYSTEM half only, as `tests/static/test_prompts_pinned.py`
        takes it, so a stored row and that file's pin can still be compared
        without converting either.
        """
        return digest(self.text)


def stored(
    service: str | None = None,
    version: str | None = None,
    name: str | None = None,
) -> list[Prompt]:
    """The prompts recorded, newest version first, narrowed by what is given.

    Read by the api, which is why this is here and not in either stage's
    own package: `question_generation.prompts` imports the modules that
    compose the prompts, and those import the model client - so an api
    that asked the code what a prompt says would load litellm to answer
    it. It asks the table instead, which is also the only thing that can
    answer for a version the code has moved past.
    """
    conditions = [
        column == value
        for column, value in (
            (Prompt.service, service),
            (Prompt.version, version),
            (Prompt.name, name),
        )
        if value is not None
    ]
    with sessions().begin() as session:
        rows = list(
            session.scalars(
                select(Prompt)
                .where(*conditions)
                .order_by(Prompt.service, Prompt.version.desc(), Prompt.name)
            )
        )
        session.expunge_all()
    return rows


def record(service: str, composed: Iterable[Composed], model: ModelSettings) -> int:
    """Writes one stage's prompts, and reports how many rows moved.

    `model` is the one this stage calls, and it is what a published prompt
    is offered against in Phoenix's playground: opened there it is replayed
    at the temperature, the window and the provider that sent it. Asked for
    rather than read off `LLM_MODEL` here, because a stage that overrides
    it - the judge, the topic labeller - would otherwise publish its
    prompts against a model that never sees them. A prompt naming one of
    its own is resolved against this; see `Composed.model`.

    Idempotent: a start-up that changes nothing writes nothing. Called
    where a stage starts rather than per row - the prompts cannot change
    while a process runs, because they are module constants and composed
    from them.

    A text that has CHANGED under an unchanged version is drift, and the
    warning is the whole point of the digest column. It is stored anyway
    and the run continues: `tests/static/test_prompts_pinned.py` is what
    fails a pull request over this, and a worker that would not start
    because somebody fixed a typo in a prompt is worse than one that says
    so. The row then matches what is actually being sent, which is the
    more useful of the two things it could say.

    **What moved is also sent to Phoenix**, so a span's
    `llm.prompt_template.version` opens against a prompt without anybody
    remembering a command. What MOVED and not everything: the table is the
    record of what is being sent, so a start-up that changes nothing has
    nothing to publish - and `prompts.create` posts a new Phoenix version
    every time it is called, so republishing the lot each run would pile up
    an identical version per prompt per run. See `stages/publish.py`.

    Never raises. A stage that cannot record its prompts still has them;
    what is lost is the ability to read them back, and that must not stop
    a run.

    Returns:
        How many rows were written or rewritten. 0 when everything
        already matched, which is every start-up after the first.
    """
    held = list(composed)
    if not held:
        return 0
    try:
        moved = _write(service, held)
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        log.warning(
            "could not record %s's prompts, so questions written now cannot be "
            "resolved to the prompt that wrote them: %s: %s",
            service,
            type(exc).__name__,
            exc,
        )
        return 0
    _publish(service, moved, model)
    return len(moved)


def _publish(service: str, moved: Sequence[Composed], model: ModelSettings) -> None:
    """Sends the prompts just written to Phoenix, if there are any.

    Imported here rather than at the top, and that is the whole reason this
    is a function: `stages.publish` reads this module, so a module-level
    import either way round is a cycle - and the api imports `stored` from
    here and must not load a Phoenix client to serve `GET /prompts`.

    Never raises, for the reason `record` does not: the prompt was sent and
    recorded either way, and what is lost is the copy shown beside a trace.
    """
    if not moved:
        return
    try:
        from stages.publish import publish

        publish(service, moved, model)
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        log.warning(
            "could not publish %s's prompts to Phoenix, so they are in the "
            "table and not beside the traces: %s: %s",
            service,
            type(exc).__name__,
            exc,
        )


#: The halves of a prompt whose change under a fixed version is DRIFT, in
#: the order `_write` reports them. The model beside them is not one of
#: these: repointing a stage at another model is a deployment's decision
#: and changes nothing about what the prompt asks, so it is compared - the
#: row is rewritten and republished - and not warned about.
_HALVES = ("system prompt", "user template", "response schema")


def _write(service: str, held: Sequence[Composed]) -> list[Composed]:
    """Upserts the prompts, warning about any that changed under a version.

    Returns the ones it wrote rather than a count, because that list is
    also exactly what there is to publish.
    """
    with sessions().begin() as session:
        stored = {
            (row.name, row.version): (
                row.digest,
                row.user_text,
                row.response_schema,
                row.model,
            )
            for row in session.scalars(select(Prompt).where(Prompt.service == service))
        }
        moved: list[Composed] = []
        for one in held:
            was = stored.get((one.name, one.version))
            sending = (one.digest, one.user_text, one.response_schema, one.model)
            if was == sending:
                continue
            # The model alone moving is not drift; see `_HALVES`. Sliced
            # rather than compared field by field so the two stay in step:
            # a fourth half added to the warning is added to that tuple.
            if was is not None and was[: len(_HALVES)] != sending[: len(_HALVES)]:
                # All three halves, not the digest alone: the digest is over
                # the system prompt, and a user template edited under a fixed
                # version is the same two datasets.
                log.warning(
                    "%s's %r prompt has changed (%s) but its PROMPT_VERSION is "
                    "still %s. Two prompts under one version are two datasets "
                    "that cannot be told apart. Bump the version and update "
                    "tests/static/test_prompts_pinned.py.",
                    service,
                    one.name,
                    ", ".join(
                        half
                        for half, before, now in zip(
                            _HALVES, was, sending, strict=False
                        )
                        if before != now
                    ),
                    one.version,
                )
            session.execute(
                insert(Prompt)
                .values(
                    service=service,
                    name=one.name,
                    version=one.version,
                    text=one.text,
                    user_text=one.user_text,
                    response_schema=one.response_schema,
                    model=one.model,
                    digest=one.digest,
                )
                # first_seen_at is deliberately not touched: what it would
                # then record is the last restart.
                .on_conflict_do_update(
                    constraint="prompts_service_version_name_unique",
                    set_={
                        "text": one.text,
                        "user_text": one.user_text,
                        "response_schema": one.response_schema,
                        "model": one.model,
                        "digest": one.digest,
                    },
                )
            )
            moved.append(one)
    if moved:
        log.info("recorded %d %s prompt(s)", len(moved), service)
    return moved
