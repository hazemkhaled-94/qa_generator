"""Three more judgements the first four could not make.

`assessment_metrics.metric` is constrained to the names the evaluation
phase asks, which were four. Three more apply to this pipeline's artefacts
and one of them catches something no other check in the repository does.

**`refusal`** is that one. A question marked answerable whose stored answer
is "the document does not specify" is a refusal written into the dataset as
though it were an answer. Every gate passes it: it is well formed, the
right length, cites its facts and leaks no source, and a benchmark built
from it scores a chatbot against a non-answer. `answered` is the label that
approves, so this is the second metric here whose good label scores zero.

**`conciseness`** reads the target answer as what it is - the string a
chatbot is scored against - where the existing bounds read it as a length.
QUESTIONS_ANSWER_CHARS refuses an answer that is too long; nothing refused
one that spent its allowance apologising.

**`toxicity`** is about what the pipeline WROTE rather than what the corpus
is about, and the templates say so at length: a corpus may document
harassment, and a neutral question about it is not toxic. What this catches
is a model that produced something offensive of its own, which is rare and
worth knowing before a dataset is published.

Not added, and named here so the next person does not have to work it out
again: `faithfulness` and `correctness` are phoenix-evals' own near-
duplicates of `hallucination` and `qa_correctness` - the same judgement
with different rails - and the span-level variants are the same again.
`code_readability`, `code_functionality`, `sql_gen_eval`, the five
tool-calling ones, `reference_link_correctness`, `human_vs_ai` and
`user_frustration` judge things this pipeline does not produce.

A widened CHECK, so nothing already stored moves. Rows judged under
template version 1 keep the metrics that version asked for; the version is
on the row, which is what makes three missing metrics readable rather than
mysterious. `make assess-rerun` is what re-judges them.

Revision ID: c1a7e93b04d6
Revises: 6e2f8c0d3a91
"""

from __future__ import annotations

from alembic import op

revision: str = "c1a7e93b04d6"
down_revision: str | None = "6e2f8c0d3a91"
branch_labels: str | None = None
depends_on: str | None = None

_CONSTRAINT = "assessment_metrics_metric_valid"

#: What the phase asks now.
_METRICS = (
    "'hallucination', 'relevance', 'qa_correctness', 'summarization', "
    "'toxicity', 'conciseness', 'refusal'"
)

#: What it asked under template version 1.
_WAS = "'hallucination', 'relevance', 'qa_correctness', 'summarization'"


def upgrade() -> None:
    """Applies the change."""
    op.drop_constraint(_CONSTRAINT, "assessment_metrics", type_="check")
    op.create_check_constraint(
        _CONSTRAINT, "assessment_metrics", f"metric IN ({_METRICS})"
    )


def downgrade() -> None:
    """Takes it back out.

    The three new metrics go with it: a row carrying one could not satisfy
    the narrower constraint, and leaving it would make the downgrade fail
    on any corpus that had been judged. Their assessments keep their other
    metrics and their verdict, which is then a verdict over fewer
    judgements than it was reached on - so re-run the phase after going
    back.
    """
    op.execute(
        "DELETE FROM assessment_metrics "
        "WHERE metric IN ('toxicity', 'conciseness', 'refusal')"
    )
    op.drop_constraint(_CONSTRAINT, "assessment_metrics", type_="check")
    op.create_check_constraint(
        _CONSTRAINT, "assessment_metrics", f"metric IN ({_WAS})"
    )
