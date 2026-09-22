"""Two gates the column would not accept.

`restates_question` and `answer_incomplete` were added to
`QuestionRejection` in 5acd189, and `questions.rejected_reason` is held to
that enum by a CHECK the model builds with `one_of`. A database CREATED
after that commit has both. A database MIGRATED across it has neither,
because no revision widened the constraint - so the gate fires, the row is
refused by Postgres, and the topic fails with an IntegrityError rather than
a verdict.

Nothing caught it, and the reason is worth writing down: **alembic's
autogenerate does not diff CHECK constraints.** `compare_metadata` is what
`test_the_models_and_the_migrations_agree` runs, and it compares tables,
columns, indexes and foreign keys - a CHECK that has fallen behind its enum
is invisible to it. Every test passed because the test container builds its
schema by migration and then the models declare the same stale constraint
over it; only a long-lived database shows the gap, and only when a gate
that was added late finally fires.

`test_every_enum_check_matches_its_model` is the answer to that, and it
fails on this revision's parent.

The list is written out rather than built from the enum. A migration is a
record of what was done, so a future member has to arrive with a revision
of its own instead of silently changing what this one meant.

Revision ID: 9c04a7f1e6b2
Revises: 4e81b7c23d05
"""

from __future__ import annotations

from alembic import op

revision: str = "9c04a7f1e6b2"
down_revision: str | None = "4e81b7c23d05"
branch_labels: str | None = None
depends_on: str | None = None

NAME = "questions_rejected_reason_valid"

#: Every gate that may stop a question, as of this revision.
AFTER = (
    "malformed",
    "compound",
    "answer_too_short",
    "answer_too_long",
    "wrong_form",
    "wrong_type",
    "explanation_unusable",
    "restates_question",
    "answer_incomplete",
    "leaks_source",
    "unanchored",
    "not_recoverable",
    "duplicate",
    "answerable_after_all",
    "answerable_elsewhere",
    "off_topic",
    "asks_nothing_new",
    "off_thread",
    "source_changed",
)

#: What it held before, which is the same list without the two.
BEFORE = tuple(one for one in AFTER if one not in ("restates_question", "answer_incomplete"))


def _check(values: tuple[str, ...]) -> str:
    """The CHECK body for one set of gates."""
    listed = ", ".join(f"'{one}'" for one in values)
    return f"rejected_reason IS NULL OR rejected_reason IN ({listed})"


def upgrade() -> None:
    """Widens the constraint to the gates the checker can actually reach."""
    op.drop_constraint(NAME, "questions", type_="check")
    op.create_check_constraint(NAME, "questions", _check(AFTER))


def downgrade() -> None:
    """Narrows it again, refusing the rows the two gates wrote.

    Deletes them first. A CHECK cannot be added to a table that already
    violates it, so leaving them would make this downgrade fail rather
    than reverse - and a question refused for restating its own prompt is
    a rejection, which is drop-rate evidence and not a question anybody
    loses sleep over. The archive trigger keeps a copy either way.
    """
    op.execute(
        "DELETE FROM questions WHERE rejected_reason IN "
        "('restates_question', 'answer_incomplete');"
    )
    op.drop_constraint(NAME, "questions", type_="check")
    op.create_check_constraint(NAME, "questions", _check(BEFORE))
