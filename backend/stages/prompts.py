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

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.qa_generator import Prompt, sessions

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
        text: The prompt as composed, which is what the model is given.
    """

    name: str
    version: str
    text: str

    @property
    def digest(self) -> str:
        """This text's digest."""
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


def record(service: str, composed: Iterable[Composed]) -> int:
    """Writes one stage's prompts, and reports how many rows moved.

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
        return _write(service, held)
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        log.warning(
            "could not record %s's prompts, so questions written now cannot be "
            "resolved to the prompt that wrote them: %s: %s",
            service,
            type(exc).__name__,
            exc,
        )
        return 0


def _write(service: str, held: Sequence[Composed]) -> int:
    """Upserts the prompts, warning about any that changed under a version."""
    with sessions().begin() as session:
        stored = {
            (row.name, row.version): row.digest
            for row in session.scalars(select(Prompt).where(Prompt.service == service))
        }
        moved = 0
        for one in held:
            was = stored.get((one.name, one.version))
            if was == one.digest:
                continue
            if was is not None:
                log.warning(
                    "%s's %r prompt has changed but its PROMPT_VERSION is still "
                    "%s: stored %s, sending %s. Two prompts under one version are "
                    "two datasets that cannot be told apart. Bump the version and "
                    "update tests/static/test_prompts_pinned.py.",
                    service,
                    one.name,
                    one.version,
                    was,
                    one.digest,
                )
            session.execute(
                insert(Prompt)
                .values(
                    service=service,
                    name=one.name,
                    version=one.version,
                    text=one.text,
                    digest=one.digest,
                )
                # first_seen_at is deliberately not touched: what it would
                # then record is the last restart.
                .on_conflict_do_update(
                    constraint="prompts_service_version_name_unique",
                    set_={"text": one.text, "digest": one.digest},
                )
            )
            moved += 1
    if moved:
        log.info("recorded %d %s prompt(s)", moved, service)
    return moved
