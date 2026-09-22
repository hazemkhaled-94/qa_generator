"""A second judge over one run's questions, recorded and never enforced.

The checker decides what is kept. Everything in `evaluation/README.md`
argues that it should stay that way - a model's answers move between two
runs at the same temperature, and the German half of this corpus is where
an LLM judge was measured at chance. So nothing here gates anything.

What a second judge is good for is the DISAGREEMENT. A question the gates
accepted and an independent judge calls unsupported is either a gate that
let something through or a judge that is wrong, and which of those it is
cannot be decided from a terminal - it needs a person. That is what
`review/` is: a hundred of those decisions made in a row instead of one at
a time through a table. This produces the queue.

The judging is `phoenix.evals` and not another prompt written here. What
`phrasing.py` hand-rolls per judgement - a template, a constrained answer,
retries, concurrency and a rate limit - is what `create_classifier` and
`evaluate_dataframe` already are, and the hallucination template it is
pointed at is one somebody else has already tested. This is the one place
in this repository where an LLM judge is the right tool, because the output
is a queue for a person rather than a verdict on a row.

Host-side, like `review/` and the rest of `evaluation/`: no image carries
`arize-phoenix-evals`, and a worker has no business holding a judge.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select

from database.qa_generator import (
    Fact,
    FactPassage,
    Passage,
    Question,
    QuestionFact,
    QuestionStatus,
)
from database.qa_generator.repository import Repository

log = logging.getLogger(__name__)

#: What the shipped rails call an answer the reference supports, and one it
#: does not. Read off `HALLUCINATION_PROMPT_RAILS_MAP` rather than written
#: here, in `_classifier`; these are the names for the two directions.
FACTUAL = "factual"
HALLUCINATED = "hallucinated"

#: How many questions one command judges by default. A run writes about
#: three thousand and judging all of them is three thousand model calls for
#: a queue nobody will work through in one sitting.
SAMPLE = 200


@dataclass(frozen=True)
class Judged:
    """One question, what the gates said, and what the judge said.

    Attributes:
        id: The question's row id, which is what `review/` needs to push it.
        question: The question as written.
        answer: The target answer the gates accepted or refused.
        gate: The gate that stopped it, or None where none did.
        label: What the judge called it - `factual` or `hallucinated`.
        explanation: Why the judge said so, which is what a reviewer reads
            first and the reason the template with explanations is used.
    """

    id: int
    question: str
    answer: str
    gate: str | None
    label: str
    explanation: str

    @property
    def accepted(self) -> bool:
        """Whether the gates kept it."""
        return self.gate is None

    @property
    def disagrees(self) -> bool:
        """Whether the two judgements point opposite ways.

        Only for an ACCEPTED question the judge calls unsupported. The
        other direction - a rejected question the judge calls factual - is
        not a disagreement worth a person's time: most rejections are for
        something the judge is not being asked about at all, and
        `leaks_source` or `compound` says nothing about whether the answer
        is in the passages.
        """
        return self.accepted and self.label == HALLUCINATED


class AnsweredQuestions(Repository):
    """Reads the answerable questions one run wrote, with their passages."""

    def sample(self, run: str, limit: int = SAMPLE) -> list[tuple]:
        """One run's answerable questions and the text they rest on.

        Answerable only: the judge is asked whether the passages support
        the answer, and an unanswerable question has none. Ordered by id so
        the same run gives the same sample twice.
        """
        passages = (
            select(func.string_agg(Passage.text, "\n\n"))
            .select_from(QuestionFact)
            .join(Fact, Fact.id == QuestionFact.fact_id)
            .join(FactPassage, FactPassage.fact_id == Fact.id)
            .join(Passage, Passage.id == FactPassage.passage_id)
            .where(QuestionFact.question_id == Question.id)
            .scalar_subquery()
        )
        wanted = (
            select(
                Question.id,
                Question.question_text,
                Question.target_answer,
                Question.rejected_reason,
                passages.label("passages"),
            )
            .where(
                Question.run_id == run,
                Question.answerable.is_(True),
                Question.status != QuestionStatus.DRAFT,
                Question.target_answer.is_not(None),
            )
            .order_by(Question.id)
            .limit(limit)
        )
        with self._session() as session:
            return list(session.execute(wanted))


def _classifier(model: str):
    """The hallucination classifier, pointed at the served model.

    Through litellm, because that is how every other model call in this
    repository is addressed: `LLM_MODEL` is an `azure/...` or an
    `ollama_chat/...` id, and phoenix-evals speaks the same dialect when
    told to use the litellm client. One model id, one place it is
    configured.
    """
    # The two `pyright: ignore`s are the library's omission rather than a
    # reach into its internals: both names are documented API and neither
    # is listed in `phoenix.evals.__all__`, which is what makes a checker
    # call them private. Ignored by rule and by name, so a different error
    # on the same line still fails.
    from phoenix import evals

    provider, _, rest = model.partition("/")
    rails = evals.HALLUCINATION_PROMPT_RAILS_MAP  # pyright: ignore[reportPrivateImportUsage]
    template = evals.HALLUCINATION_PROMPT_TEMPLATE_WITH_EXPLANATION  # pyright: ignore[reportPrivateImportUsage]
    return evals.create_classifier(
        name="hallucination",
        prompt_template=str(template),
        llm=evals.LLM(provider=provider, model=rest or model, client="litellm"),
        # `factual` is the good direction, so it scores 1 and the run's
        # mean reads as "how much of this the judge backs".
        choices={str(rails[False]): 1.0, str(rails[True]): 0.0},
        direction="maximize",
    )


def judge(run: str, model: str, limit: int = SAMPLE) -> list[Judged]:
    """Puts one run's accepted answers to an independent judge.

    Raises:
        RuntimeError: If the run wrote no answerable question, which means
            the id is wrong or the run has been deleted - and unlike the
            gate counts, this one cannot be read out of the archive,
            because it needs the passages the questions rest on.
    """
    import pandas as pd
    from phoenix.evals import evaluate_dataframe

    rows = AnsweredQuestions().sample(run, limit)
    if not rows:
        raise RuntimeError(
            f"run {run!r} has no answerable question with an answer stored. "
            f"`make questions-runs` lists the runs there are."
        )

    frame = pd.DataFrame(
        [
            {
                "input": question,
                "output": answer,
                "reference": passages or "",
            }
            for _, question, answer, _, passages in rows
        ]
    )
    log.info("judging %d answer(s) from run %s with %s", len(frame), run, model)
    scored = evaluate_dataframe(frame, [_classifier(model)])

    judged = []
    for row, (question_id, question, answer, gate, _) in zip(
        scored.to_dict("records"), rows, strict=True
    ):
        label, explanation = _read(row)
        judged.append(
            Judged(
                id=question_id,
                question=question,
                answer=answer,
                gate=gate,
                label=label,
                explanation=explanation,
            )
        )
    return judged


def _read(row: dict) -> tuple[str, str]:
    """The label and the explanation out of one scored row.

    Read by suffix rather than by an exact column name: phoenix-evals names
    the columns after the evaluator, and pinning the full name here would
    make a rename of theirs a KeyError of ours.
    """
    label = next(
        (value for key, value in row.items() if key.endswith("label")),
        "",
    )
    explanation = next(
        (value for key, value in row.items() if key.endswith("explanation")),
        "",
    )
    return str(label or ""), str(explanation or "")


def report(judged: Sequence[Judged]) -> list[str]:
    """What the two judgements made of each other, as lines to print."""
    accepted = [one for one in judged if one.accepted]
    disagreed = [one for one in judged if one.disagrees]
    backed = sum(1 for one in accepted if one.label == FACTUAL)

    share = f" - {backed / len(accepted):.0%}" if accepted else ""
    lines = [
        (
            f"judged {len(judged)} answer(s); {len(accepted)} of them "
            f"accepted by the gates"
        ),
        f"the judge backs {backed} of those {len(accepted)}{share}",
        (
            f"{len(disagreed)} disagreement(s): the gates kept these and "
            f"the judge says the passages do not support the answer"
        ),
        "",
    ]
    for one in disagreed:
        lines.append(f"  [{one.id}] {one.question}")
        lines.append(f"        answer: {one.answer}")
        lines.append(f"        judge:  {' '.join(one.explanation.split())[:160]}")
    if disagreed:
        lines.append("")
        lines.append(
            "Neither side is right by default. Push them to Argilla and have "
            "somebody look: make review-push-questions IDS="
            + ",".join(str(one.id) for one in disagreed[:20])
            + ("..." if len(disagreed) > 20 else "")
        )
    return lines
