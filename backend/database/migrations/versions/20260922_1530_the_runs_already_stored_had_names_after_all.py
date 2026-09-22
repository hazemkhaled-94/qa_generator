"""The runs already stored had names after all.

The revision before this one added `run_id` and said nothing could be
backfilled, because a row written before runs were named was produced by a
run that had none. That was wrong, and the corpus in front of it is the
counter-example: 7,432 questions written over two days, and `created_at`
separates them into three runs on its own.

A run is hours of continuous writing with hours of nothing either side.
Measured over those 7,432 rows:

    largest gap INSIDE a run      39m 08s
    smallest gap BETWEEN runs      4h 54m

`GAP` is two hours, which sits between them with 1h21m of margin below and
2h54m above. At 45 minutes - the first threshold tried - the margin below
is six minutes, which is one slow topic away from splitting a run in two.

**Checked against a run whose numbers were written down.** The first batch
this recovers is the one `evaluation/README.md` records as "The full run
under all of it", and every figure agrees:

    written 3,094   accepted 1,744   not_recoverable 512
    duplicate 180   leaks_source 89  unanchored 82

Seven numbers, none of them chosen here. That is what makes this a
recovery rather than a guess - the alternative was deleting a corpus that
took twelve hours of model time and is the baseline every later run is
measured against.

Named `recovered-<n>-<date>` rather than given a uuid. A uuid would be
honest and useless; these are the runs somebody will compare against, and
they have to be typeable.

Facts and topics are deliberately left alone. Extraction ran over days
rather than in sittings, and a topic fit is a handful of rows written in
one transaction - neither has the shape this reads, and inventing a
boundary where the data does not show one is the thing the revision before
this was right to refuse.

Revision ID: 4e81b7c23d05
Revises: 7a3f2c9e4b18
"""

from __future__ import annotations

from alembic import op

revision: str = "4e81b7c23d05"
down_revision: str | None = "7a3f2c9e4b18"
branch_labels: str | None = None
depends_on: str | None = None

#: How long a pause has to be before it separates two runs. See the module
#: docstring for the two measurements this sits between.
GAP = "2 hours"

#: Only rows that have no run. Idempotent, so running it twice is not two
#: sets of names, and a deployment that has already named some runs by
#: hand keeps them.
BACKFILL = f"""
WITH boundaries AS (
    SELECT id,
           created_at,
           CASE
               WHEN created_at - lag(created_at) OVER (ORDER BY created_at, id)
                    > interval '{GAP}'
               THEN 1 ELSE 0
           END AS starts_one
    FROM questions
    WHERE run_id IS NULL
),
batched AS (
    SELECT id,
           created_at,
           sum(starts_one) OVER (ORDER BY created_at, id) AS batch
    FROM boundaries
),
named AS (
    SELECT id,
           'recovered-'
             || (dense_rank() OVER (ORDER BY batch))
             || '-'
             || to_char(min(created_at) OVER (PARTITION BY batch), 'YYYYMMDD')
             AS run
    FROM batched
)
UPDATE questions
SET run_id = named.run
FROM named
WHERE questions.id = named.id
  AND questions.run_id IS NULL;
"""


def upgrade() -> None:
    """Names the runs `created_at` already separates."""
    op.execute(BACKFILL)


def downgrade() -> None:
    """Takes the recovered names back off, and only those.

    A run named by hand or by `RUN_ID` is left alone: this put the
    `recovered-` ones there and this is what may remove them.
    """
    op.execute("UPDATE questions SET run_id = NULL WHERE run_id LIKE 'recovered-%';")
