"""An accepted fact invented nothing, and the table says so.

`units_added` and `unresolved_references` are what the two gates nobody
could write lexically leave behind. README.md says an accepted fact has
them empty - that is the whole claim `unsupported_addition` makes - and
until now the only thing enforcing it was the order of the checks in
`FactChecker._by_kind`. A bug there would have stored the opposite, and
question generation, which asks only for `validated`, would have offered
it.

Two constraints rather than one, because the invariant is not the same
for all four kinds, and the corpus says so plainly:

    kind     method          accepted  units_added  unresolved_refs
    atomic   llm                 3770            0                0
    atomic   deterministic        392          284               37
    bridge   llm                  343            0                0
    outline  llm                  539            0               40
    summary  llm                  526            0              251

`units_added` is empty for every accepted statement a model wrote, of
every kind: all four face `_supported`.

`unresolved_references` is not. `_self_contained` is applied to a claim
and not to a digest - a summary stands in for a whole passage rather than
asserting something about it, and 291 accepted ones open with a pronoun.
So the second constraint covers `atomic` and `bridge`, which are the
kinds a question is written from. The column comment claimed otherwise
and is corrected here rather than the data being forced to match it.

Neither holds for a deterministic statement. One composed from a grid
faces the copy check and nothing else, because it is neither written nor
a sentence, so it is accepted carrying whatever the reading found - 284
of the 392 do.

Nothing is rewritten. Both constraints are true of every row already
stored, which is the point of measuring before adding them.

Revision: b8d41e7a02c5
Parent:   e5a02c7b1946
Created:  2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8d41e7a02c5"
down_revision: str | None = "e5a02c7b1946"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INVENTED = "facts_accepted_invented_nothing"
_ALONE = "facts_accepted_claim_stands_alone"

_WAS = (
    "Pronouns leaving the statement dependent on its context. Empty on every "
    "accepted fact, which is what makes a question from it answerable alone."
)

_NOW = (
    "Pronouns leaving the statement dependent on its context. Empty on every "
    "accepted atomic fact and bridge, which is what makes a question from one "
    "answerable alone. A summary and an outline are not held to it: they stand "
    "in for a whole passage rather than assert something about it, so they are "
    "checked on length and on inventing nothing and may open with a pronoun."
)


def upgrade() -> None:
    """Writes the two invariants the checks already hold to."""
    op.create_check_constraint(
        _INVENTED,
        "facts",
        "extraction_method <> 'llm' OR validated IS FALSE OR "
        "cardinality(units_added) = 0",
    )
    op.create_check_constraint(
        _ALONE,
        "facts",
        "extraction_method <> 'llm' OR validated IS FALSE OR "
        "kind NOT IN ('atomic', 'bridge') OR "
        "cardinality(unresolved_references) = 0",
    )
    op.alter_column(
        "facts",
        "unresolved_references",
        existing_type=sa.ARRAY(sa.Text()),
        existing_nullable=False,
        existing_comment=_WAS,
        comment=_NOW,
    )


def downgrade() -> None:
    """Drops them. No row has to move either way."""
    op.alter_column(
        "facts",
        "unresolved_references",
        existing_type=sa.ARRAY(sa.Text()),
        existing_nullable=False,
        existing_comment=_NOW,
        comment=_WAS,
    )
    op.drop_constraint(_ALONE, "facts", type_="check")
    op.drop_constraint(_INVENTED, "facts", type_="check")
