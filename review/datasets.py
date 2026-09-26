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

from review.judged import UNJUDGED, Opinion
from review.records import FactRow, QuestionRow
from telemetry import pipeline

#: What the CLI takes, and what the code branches on.
FACTS = "facts"
TOPIC_LABELS = "topic-labels"
QUESTIONS = "questions"
NAMES = (FACTS, TOPIC_LABELS, QUESTIONS)

#: Which stage produces each kind. What Argilla shows is numbered with the
#: stage's place in the pipeline, because Argilla sorts its datasets by
#: name and `facts, questions, topic-labels` is not the order they are
#: produced in. The number means what it means in Phoenix's projects and
#: Dagster's groups: the same step seen from another side.
STAGES = {
    FACTS: "extraction",
    TOPIC_LABELS: "topic_modelling",
    QUESTIONS: "question_generation",
}


def dataset_name(name: str) -> str:
    """What Argilla holds one kind under, numbered by its stage."""
    return pipeline.numbered(name, STAGES.get(name))


#: The one question every verdict dataset asks. Two labels and no middle:
#: "unsure" is where a review goes to not happen, and a reviewer who
#: genuinely cannot tell should leave the record unanswered, which Argilla
#: already distinguishes from an answer.
_VERDICT = ["accepted", "rejected"]


def _judge_metadata() -> list[rg.TermsMetadataProperty]:
    """What the evaluation phase said, as the two terms it is filtered by.

    Two properties rather than one: `judge_verdict` is the verdict and is
    what a reviewer narrows by, and `judge_disagrees` is the pair worth a
    sitting of their own - the artefacts the pipeline kept and the judge
    refused.

    `judge_verdict` rather than `judge`, because the FIELD carrying the
    reasoning is called that and **Argilla requires every name in a
    dataset's settings to be unique across fields, questions and metadata
    together**. Two of them called `judge` is refused at dataset creation
    with `SettingsError`, which reaches a person as a push that will not
    run. See `tests/static/test_review_datasets.py`.

    A function and not a constant, like every other declaration in this
    module: building an `rg.` object reaches for the default client, so one
    at import time makes importing this module require a running Argilla.
    """
    return [
        rg.TermsMetadataProperty(name="judge_verdict", title="What the LLM judge said"),
        rg.TermsMetadataProperty(
            name="judge_disagrees", title="Judge disagrees with the pipeline"
        ),
    ]


def _judge_field() -> rg.TextField:
    """The field carrying the judge's reasoning.

    Not required: the phase may be off, or may not have reached this row,
    and a required field would make a record unpushable in both cases.
    """
    return rg.TextField(
        name="judge", title="What the LLM judge said, and why", required=False
    )


#: The paragraph added to every dataset's guidelines. One wording for all
#: three, because the thing it has to establish is the same in each: the
#: judge is not an answer key.
_JUDGE_GUIDELINE = (
    "\n\n"
    "The `judge` field is what an LLM judge said about this row, and the "
    "`judge_verdict` metadata is its verdict in one word. It is NOT the "
    "answer. It "
    "is a second machine opinion recorded beside the checker's, and it "
    "gates nothing - one was measured at chance on German text. Where it "
    "says it disagrees with the pipeline, that pair is the reason this row "
    "is in front of you: one of the two is wrong and only you can say which."
)

#: Added after it. Argilla's job is the artefacts a person judges, and a
#: reviewer who needs the calls behind one has to be told where they are.
_WHERE_GUIDELINE = (
    "\n\n"
    "The number in this dataset's name is the stage that produced it: "
    "4 extraction, 5 topic modelling, 6 question generation. Argilla holds "
    "the artefacts and your verdicts on them, and nothing else. The calls "
    "that produced a row are in Phoenix under a project numbered the same "
    "way, every line the run wrote is in Grafana, and which run it was is "
    "in Dagster. The application's own pages show one artefact's whole "
    'chain under "How this was produced".'
)


def judge_metadata(opinion: Opinion) -> dict[str, str]:
    """The two metadata terms one artefact's verdict becomes."""
    return {
        "judge_verdict": opinion.verdict,
        "judge_disagrees": "yes" if opinion.disagrees else "no",
    }


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
            + _JUDGE_GUIDELINE
            + _WHERE_GUIDELINE
        ),
        fields=[
            rg.TextField(name="statement", title="The fact as written"),
            rg.TextField(name="evidence", title="The sentences it cites"),
            _judge_field(),
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
            *_judge_metadata(),
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
            "written about it at all." + _JUDGE_GUIDELINE + _WHERE_GUIDELINE
        ),
        fields=[
            rg.TextField(name="terms", title="Its strongest terms"),
            rg.TextField(name="label", title="What the model named it"),
            _judge_field(),
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
            *_judge_metadata(),
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
            "know. `answerable` says which kind this is.\n\n"
            "`facts` is what was written from, `evidence` is the sentences "
            "those facts cite, and `explanation` is why the writer says "
            "that is the answer. Judge the answer against the evidence: a "
            "fact can be a fair reading of a sentence that does not say "
            "what the answer claims." + _JUDGE_GUIDELINE + _WHERE_GUIDELINE
        ),
        fields=[
            rg.TextField(name="question", title="The question"),
            rg.TextField(name="answer", title="The answer it expects"),
            rg.TextField(name="facts", title="The facts it was written from"),
            rg.TextField(name="evidence", title="The sentences those facts cite"),
            rg.TextField(
                name="explanation", title="Why that is the answer, as the writer put it"
            ),
            _judge_field(),
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
            *_judge_metadata(),
            # A margin in [0, 1], so a reviewer can open the queue at the
            # questions that survived a gate by the least.
            rg.FloatMetadataProperty(
                name="confidence", title="Confidence", min=0.0, max=1.0
            ),
            rg.IntegerMetadataProperty(name="question_id", title="Question id"),
        ],
    )


SETTINGS = {
    FACTS: fact_settings,
    TOPIC_LABELS: topic_settings,
    QUESTIONS: question_settings,
}


def fact_record(row: FactRow, judge: Opinion = UNJUDGED) -> rg.Record:
    """One fact as a record."""
    return rg.Record(
        # The database id, so a pulled answer finds its row again. Argilla
        # deduplicates on it too, so pushing a record twice updates rather
        # than doubling it.
        id=str(row.id),
        fields={
            "statement": row.statement,
            "evidence": row.evidence,
            "judge": judge.detail,
        },
        metadata={
            "kind": row.kind,
            "checker": row.rejection_code or "accepted",
            **judge_metadata(judge),
            "fact_id": row.id,
        },
    )


def topic_record(topic, judge: Opinion = UNJUDGED) -> rg.Record:
    """One topic as a record."""
    return rg.Record(
        id=str(topic.id),
        fields={
            "terms": ", ".join(topic.top_terms),
            "label": topic.label or "(the model named nothing)",
            "judge": judge.detail,
        },
        metadata={
            "language": topic.language or "unknown",
            "labelled_by": topic.labelled_by or "nobody",
            **judge_metadata(judge),
            "topic_id": topic.id,
            "passages": topic.passages,
        },
    )


def question_record(row: QuestionRow, judge: Opinion = UNJUDGED) -> rg.Record:
    """One question as a record."""
    return rg.Record(
        id=str(row.id),
        fields={
            "question": row.question_text,
            "answer": row.target_answer or "(deliberately unanswerable)",
            "facts": "\n".join(f"- {one}" for one in row.facts) or "(none recorded)",
            "evidence": "\n".join(f"- {one}" for one in row.evidence)
            or "(none recorded)",
            # A question written to have no answer explains nothing, which
            # is the ordinary case rather than a gap: the field says so
            # rather than arriving empty, which Argilla refuses.
            "explanation": row.answer_explanation or "(none written)",
            "judge": judge.detail,
        },
        metadata={
            "difficulty": row.difficulty or "unbanded",
            "question_type": row.question_type or "untyped",
            "language": row.language,
            "answerable": str(row.answerable).lower(),
            "gate": row.rejected_reason or row.status,
            **judge_metadata(judge),
            "question_id": row.id,
            # What a reviewer sorts the queue by. Omitted rather than sent
            # as a sentinel where nothing measured it: a filter on a range
            # cannot tell -1 from a real reading, and a row with no
            # confidence is not a row with a bad one.
            **({} if row.confidence is None else {"confidence": row.confidence}),
        },
    )
