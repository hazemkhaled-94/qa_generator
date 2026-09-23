"""The prompts table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Identity, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from database.qa_generator.base import Base


class Prompt(Base):
    """One prompt, as it was composed, under the version that composed it.

    `facts.prompt_version` and `questions.prompt_version` name a version and
    nothing resolved it. A row lives for ever - `questions` is append-only
    and never hard-deleted - and the span carrying the prompt that wrote it
    does not: a Phoenix project is one run and has a retention of its own.
    So a question written under version 5 pointed at a prompt that existed
    only in a git commit, and an exported dataset could not say what wrote
    it without the repository beside it.

    This is that pointer made resolvable, and it is why the text is stored
    COMPOSED rather than as the pieces it is built from. A type's system
    prompt is the shared rules plus its own directive and worked examples,
    assembled at call time; the pieces are in the source and the thing the
    model was actually given is here.

    Written by the code and read by everything else. A row is a record of
    what a version meant, not a place to change it: editing one here
    changes nothing about what any process sends, and the next start-up
    writes the source's text back over it.

    No `settings_version`. What a prompt says does not move when a setting
    does - that is the whole reason this column is not covered by the one
    beside it on `questions`.
    """

    __tablename__ = "prompts"
    __table_args__ = (
        # One text per prompt per version per service. Two rows would make
        # "what did version 8 say" depend on row order, which is the
        # question this table exists to answer. The service is in the key
        # because a version is per MODULE: extraction's digest extractor
        # and question generation's verifier are both at version 1, and
        # they are unrelated.
        UniqueConstraint(
            "service", "version", "name", name="prompts_service_version_name_unique"
        ),
        {
            "comment": "Every prompt this pipeline sends, as it was composed, keyed "
            "by the PROMPT_VERSION that composed it. What facts.prompt_version and "
            "questions.prompt_version point at. Written at stage start-up from the "
            "source; read-only to everything else, and editing a row changes nothing "
            "about what is sent."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    service: Mapped[str] = mapped_column(
        Text,
        index=True,
        comment="Which stage sends it: extraction or questions. The stage writes "
        "its own and no other's, because no backend service imports another.",
    )
    name: Mapped[str] = mapped_column(
        Text,
        comment="What the prompt is, within its service: a question type, "
        "`perturb`, `follow`, one of the phrasing judgements, or one of the "
        "verifier's four.",
    )
    version: Mapped[str] = mapped_column(
        Text,
        index=True,
        comment="The PROMPT_VERSION its module declared when this text was "
        "written. Indexed: resolving a question's prompt_version is the query "
        "this table is for.",
    )
    text: Mapped[str] = mapped_column(
        Text, comment="The prompt as composed, which is what the model was given."
    )
    digest: Mapped[str] = mapped_column(
        Text,
        comment="The first sixteen hex characters of the text's SHA-256, spelled "
        "as tests/static/test_prompts_pinned.py spells it, so a row and that "
        "file's pin can be compared by eye. What a changed prompt under an "
        "unchanged version is noticed by at run time.",
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When this version of this prompt was first recorded. Not "
        "updated when the text is rewritten under the same version: what that "
        "would record is the last restart, and the warning is where a rewrite "
        "is reported.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="When the text was last written. Moves only when the text "
        "actually changed, which under an unchanged version is drift.",
    )
