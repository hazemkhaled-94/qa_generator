"""The three datasets, as Argilla holds them.

One per thing a model decided and a person may disagree with:

    facts         is this statement true, and does the evidence support it?
    topic-labels  is this a good name for these terms, and is it a subject?
    questions     would somebody ask this, and is the answer right?

Each carries the machine's own verdict as metadata rather than hiding it.
A reviewer told nothing sees a wall of statements; a reviewer told "the
checker refused this one for unsupported_addition" is answering the
question the review is actually for, which is whether the checker is
right. The risk is anchoring, and it is the lesser one: a sample nobody
can interpret gets abandoned.

The database stays the source of truth. Argilla holds a copy of the rows
put in front of somebody and the answers they gave; `review-pull` brings
the answers back and the copy is disposable.
"""

from __future__ import annotations

import argilla as rg

from review.records import FactRow, QuestionRow

#: The dataset names, which are also what the CLI takes.
FACTS = "facts"
TOPIC_LABELS = "topic-labels"
QUESTIONS = "questions"
NAMES = (FACTS, TOPIC_LABELS, QUESTIONS)

#: The one question every verdict dataset asks. Two labels and no middle:
#: "unsure" is where a review goes to not happen, and a reviewer who
#: genuinely cannot tell should leave the record unanswered, which Argilla
#: already distinguishes from an answer.
_VERDICT = ["accepted", "rejected"]


def fact_settings() -> rg.Settings:
    """What a reviewer is shown about a fact, and what they are asked."""
    return rg.Settings(
        guidelines=(
            "Does the statement follow from the evidence beside it, and does "
            "it stand on its own?\n\n"
            "Accept when the evidence supports every part of the statement "
            "and the statement needs no surrounding context to be understood. "
            "Reject when it adds anything the evidence does not say, when it "
            "leaves a pronoun pointing outside itself, or when it is a "
            "paraphrase of the passage rather than a claim drawn from it.\n\n"
            "`rejection_code` is what the automatic checker decided. You are "
            "judging the statement, not agreeing with the checker - the "
            "disagreements are the point of the exercise."
        ),
        fields=[
            rg.TextField(name="statement", title="The fact as written"),
            rg.TextField(name="evidence", title="The sentences it cites"),
        ],
        questions=[
            rg.LabelQuestion(
                name="verdict",
                title="Does this fact hold?",
                labels=_VERDICT,
                required=True,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="kind", title="Fact kind"),
            rg.TermsMetadataProperty(name="checker", title="What the checker said"),
            rg.IntegerMetadataProperty(name="fact_id", title="Fact id"),
        ],
    )


def topic_settings() -> rg.Settings:
    """What a reviewer is shown about a topic's name, and what they are asked."""
    return rg.Settings(
        guidelines=(
            "Is this a good name for these terms, and is it a subject the "
            "corpus should be asked about?\n\n"
            "Correct the label if it is wrong or vague; leave it alone if it "
            "is right. A topic that is an artefact of the fitting rather than "
            "a subject - boilerplate, page furniture, a mix of everything - "
            "should be marked out of coverage, which means no questions are "
            "written about it at all."
        ),
        fields=[
            rg.TextField(name="terms", title="Its strongest terms"),
            rg.TextField(name="label", title="What the model named it"),
        ],
        questions=[
            rg.TextQuestion(
                name="corrected_label",
                title="The name it should have",
                required=True,
            ),
            rg.LabelQuestion(
                name="in_coverage",
                title="Is this a subject worth asking about?",
                labels=["yes", "no"],
                required=True,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="language", title="Language"),
            rg.TermsMetadataProperty(name="labelled_by", title="Named by"),
            rg.IntegerMetadataProperty(name="topic_id", title="Topic id"),
            rg.IntegerMetadataProperty(name="passages", title="Passages held"),
        ],
    )


def question_settings() -> rg.Settings:
    """What a reviewer is shown about a question, and what they are asked."""
    return rg.Settings(
        guidelines=(
            "Would somebody actually ask this, and is the answer right?\n\n"
            "Accept when the question is one a person would put to a service "
            "desk, the answer follows from the facts beside it, and the "
            "question does not say where its own answer is. Reject a question "
            "that names its source, that asks two things at once, or whose "
            "answer is not recoverable from the facts shown.\n\n"
            "An unanswerable question is not automatically wrong: some are "
            "written deliberately, to check that a chatbot says it does not "
            "know. `answerable` says which kind this is."
        ),
        fields=[
            rg.TextField(name="question", title="The question"),
            rg.TextField(name="answer", title="The answer it expects"),
            rg.TextField(name="facts", title="The facts it was written from"),
        ],
        questions=[
            rg.LabelQuestion(
                name="verdict",
                title="Is this question good enough to keep?",
                labels=_VERDICT,
                required=True,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="difficulty", title="Difficulty band"),
            rg.TermsMetadataProperty(name="question_type", title="Type"),
            rg.TermsMetadataProperty(name="language", title="Language"),
            rg.TermsMetadataProperty(name="answerable", title="Answerable"),
            rg.TermsMetadataProperty(name="gate", title="What the gates said"),
            rg.IntegerMetadataProperty(name="question_id", title="Question id"),
        ],
    )


SETTINGS = {
    FACTS: fact_settings,
    TOPIC_LABELS: topic_settings,
    QUESTIONS: question_settings,
}


def fact_record(row: FactRow) -> rg.Record:
    """One fact as a record."""
    return rg.Record(
        # The database id, so a pulled answer finds its row again. Argilla
        # deduplicates on it too, so pushing a record twice updates rather
        # than doubling it.
        id=str(row.id),
        fields={"statement": row.statement, "evidence": row.evidence},
        metadata={
            "kind": row.kind,
            "checker": row.rejection_code or "accepted",
            "fact_id": row.id,
        },
    )


def topic_record(topic) -> rg.Record:
    """One topic as a record."""
    return rg.Record(
        id=str(topic.id),
        fields={
            "terms": ", ".join(topic.top_terms),
            "label": topic.label or "(the model named nothing)",
        },
        metadata={
            "language": topic.language or "unknown",
            "labelled_by": topic.labelled_by or "nobody",
            "topic_id": topic.id,
            "passages": topic.passages,
        },
    )


def question_record(row: QuestionRow) -> rg.Record:
    """One question as a record."""
    return rg.Record(
        id=str(row.id),
        fields={
            "question": row.question_text,
            "answer": row.target_answer or "(deliberately unanswerable)",
            "facts": "\n".join(f"- {one}" for one in row.facts) or "(none recorded)",
        },
        metadata={
            "difficulty": row.difficulty or "unbanded",
            "question_type": row.question_type or "untyped",
            "language": row.language,
            "answerable": str(row.answerable).lower(),
            "gate": row.rejected_reason or row.status,
            "question_id": row.id,
        },
    )
