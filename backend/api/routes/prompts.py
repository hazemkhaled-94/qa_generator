"""What a prompt version asked for.

`questions.prompt_version` and `facts.prompt_version` name a version. This
is what resolves one: the prompt as it was composed when that version was
current, which for a version the code has moved past exists nowhere else -
the span that carried it belongs to a Phoenix project with its own
retention, and the row outlives it.

Read-only, and there is no route that writes one. A prompt is changed in
the source and recorded by the stage that sends it; a row here is the
record, and editing it would change nothing about what any process asks.

Serves the table rather than asking the code, which is what keeps litellm
out of this process: the modules that compose these prompts import the
model client. See `tests/static/test_api_stays_light.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fastapi import APIRouter, Query

from api.params import NAME_MAX, NO_NUL
from stages.prompts import stored

router = APIRouter(prefix="/prompts", tags=["prompts"])


@dataclass(frozen=True)
class Prompt:
    """One prompt, as it was composed under one version."""

    #: The stage that sends it: `questions`, `extraction` or `topics`.
    service: str
    #: What it is within that stage - a question type, `perturb`, one of
    #: the verifier's four.
    name: str
    #: The PROMPT_VERSION its module declared when this text was written.
    version: str
    #: The SYSTEM prompt as composed, which is what the model was given as
    #: its instructions.
    text: str
    #: The USER message as its template, with `{{name}}` where the call
    #: fills a passage, a fact or a question in. Empty on a row recorded
    #: before the column existed.
    user_text: str
    #: The JSON schema the answer had to come back in. Null on a row
    #: recorded before the column existed.
    response_schema: dict | None
    #: The first sixteen hex characters of the text's SHA-256, spelled as
    #: `tests/static/test_prompts_pinned.py` spells it, so a row and that
    #: file's pin can be compared without converting either.
    digest: str
    #: When this version of this prompt was first recorded.
    first_seen_at: datetime


@router.get("", summary="The prompts recorded, and what each version asked for")
def read(
    service: str | None = Query(
        None,
        max_length=NAME_MAX,
        pattern=NO_NUL,
        description="questions, extraction or topics",
    ),
    version: str | None = Query(
        None, max_length=NAME_MAX, pattern=NO_NUL, description="a PROMPT_VERSION"
    ),
    name: str | None = Query(
        None,
        max_length=NAME_MAX,
        pattern=NO_NUL,
        description="one prompt within a service",
    ),
) -> list[Prompt]:
    """Every recorded prompt, narrowed by whatever is given.

    All three filters are optional and combine. Naming all three is how a
    page resolves one question: its service, the `prompt_version` on its
    row and its `question_type`.

    Each is bounded, as every other free-text parameter on this API is.
    They were three bare strings, and PostgreSQL text cannot hold a NUL:
    `?service=%00` reached psycopg and came back a server error rather
    than the 422 the document already promises.

    The text comes with it rather than behind a second call. It is what
    the caller wanted, a prompt is a few kilobytes, and a listing that
    made you ask twice for the only interesting column would be two round
    trips to read one thing.
    """
    return [
        Prompt(
            service=row.service,
            name=row.name,
            version=row.version,
            text=row.text,
            user_text=row.user_text,
            response_schema=row.response_schema,
            digest=row.digest,
            first_seen_at=row.first_seen_at,
        )
        for row in stored(service=service, version=version, name=name)
    ]
