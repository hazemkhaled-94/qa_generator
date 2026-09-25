"""What the judge and the gates make of each other.

`evaluation/second_opinion.py` asks an independent judge whether the
passages support the answer the gates kept, and the whole point of it is
the pair where the two disagree. Nothing tested any of that: the module was
reached only sideways, through the CLI test, which is why it sat at 38%.

The judging itself needs a served model and is the eval layer's. What is
here is the part that decides what a disagreement IS, which is a rule with
a direction, and the report that rule is read off.
"""

from __future__ import annotations

from evaluation.second_opinion import FACTUAL, HALLUCINATED, Judged, _read, report


def judged(
    row_id: int = 1,
    gate: str | None = None,
    label: str = FACTUAL,
    question: str = "How long is allowed for a reply?",
    answer: str = "Five days.",
    explanation: str = "the passage says five working days",
) -> Judged:
    """One judged question, accepted by the gates and backed by the judge."""
    return Judged(
        id=row_id,
        question=question,
        answer=answer,
        gate=gate,
        label=label,
        explanation=explanation,
    )


# ── Which pairs are worth a person's time ─────────────────────────────────


def test_a_question_no_gate_stopped_reads_as_accepted() -> None:
    """The column is NULL when nothing refused it, and NULL is the verdict."""
    assert judged(gate=None).accepted is True
    assert judged(gate="compound").accepted is False


def test_the_gates_keeping_what_the_judge_calls_unsupported_is_the_pair() -> None:
    """The one combination the whole module exists to surface."""
    assert judged(gate=None, label=HALLUCINATED).disagrees is True


def test_the_gates_and_the_judge_both_keeping_it_is_not_a_disagreement() -> None:
    """Agreement is the ordinary case and is not a queue for anybody."""
    assert judged(gate=None, label=FACTUAL).disagrees is False


def test_a_refused_question_the_judge_backs_is_not_counted_as_one() -> None:
    """The other direction, which is deliberately not a disagreement.

    Most rejections are for something the judge is not asked about:
    `leaks_source` and `compound` say nothing about whether the answer is
    in the passages, so pairing them would fill the queue with rows where
    the two were never answering the same question.
    """
    assert judged(gate="leaks_source", label=FACTUAL).disagrees is False


def test_a_refused_question_the_judge_also_refuses_is_not_a_disagreement() -> None:
    """Both against it is the clearest agreement there is."""
    assert judged(gate="compound", label=HALLUCINATED).disagrees is False


# ── Reading one scored row back ───────────────────────────────────────────


def test_the_label_and_the_reason_are_read_by_suffix() -> None:
    """phoenix-evals names its columns after the evaluator.

    Pinning the full name would make a rename of theirs a KeyError of ours.
    """
    label, explanation = _read(
        {
            "hallucination_label": HALLUCINATED,
            "hallucination_explanation": "no number of days appears",
        }
    )

    assert label == HALLUCINATED
    assert explanation == "no number of days appears"


def test_a_row_the_judge_scored_nothing_into_reads_as_empty_strings() -> None:
    """A missing column is a row to skip, not an AttributeError later."""
    assert _read({}) == ("", "")


def test_a_null_in_a_scored_column_reads_as_empty_rather_than_none() -> None:
    """`str(None)` would put the word None in a reviewer's queue."""
    assert _read({"x_label": None, "x_explanation": None}) == ("", "")


# ── What the report says ──────────────────────────────────────────────────


def test_the_report_counts_what_was_judged_and_what_was_accepted() -> None:
    """Two figures, because the share below is over the accepted ones."""
    lines = report([judged(1), judged(2, gate="compound"), judged(3)])

    assert "judged 3 answer(s); 2 of them accepted by the gates" in lines[0]


def test_the_share_is_taken_over_what_the_gates_kept() -> None:
    """Not over everything judged: the denominator is the accepted ones."""
    lines = report(
        [
            judged(1, label=FACTUAL),
            judged(2, label=HALLUCINATED),
            judged(3, gate="compound", label=HALLUCINATED),
        ]
    )

    assert "the judge backs 1 of those 2 - 50%" in lines[1]


def test_nothing_accepted_prints_no_share_rather_than_dividing_by_zero() -> None:
    """A run where every question was refused is a run with no denominator."""
    lines = report([judged(1, gate="compound")])

    assert "the judge backs 0 of those 0" in lines[1]
    assert "%" not in lines[1]


def test_an_empty_run_is_a_report_rather_than_a_crash() -> None:
    """Judging nothing is what a run with no answerable questions does."""
    lines = report([])

    assert "judged 0 answer(s)" in lines[0]
    assert "0 disagreement(s)" in lines[2]


def test_each_disagreement_is_named_with_what_the_judge_said() -> None:
    """A reviewer reads the explanation first, which is why it is fetched."""
    lines = report([judged(7, label=HALLUCINATED, explanation="no days appear")])
    said = "\n".join(lines)

    assert "[7] How long is allowed for a reply?" in said
    assert "answer: Five days." in said
    assert "no days appear" in said


def test_a_long_explanation_is_collapsed_onto_one_line() -> None:
    """The judge writes paragraphs; a queue is read as a list."""
    lines = report(
        [judged(1, label=HALLUCINATED, explanation="a\n\n  b\t c" + " x" * 200)]
    )
    (said,) = [one for one in lines if "judge:" in one]

    assert "\n" not in said
    assert "a b c" in said
    assert len(said.strip()) <= len("judge:  ") + 160 + 8


def test_a_run_with_no_disagreements_offers_nothing_to_push() -> None:
    """The instruction is there to be acted on, so it needs a queue."""
    said = "\n".join(report([judged(1, label=FACTUAL)]))

    assert "review-push-questions" not in said


def test_the_disagreements_are_offered_as_a_command_that_pushes_them() -> None:
    """Neither side is right by default, so the next step is a person."""
    said = "\n".join(
        report([judged(4, label=HALLUCINATED), judged(9, label=HALLUCINATED)])
    )

    assert "make review-push-questions IDS=4,9" in said


def test_a_queue_longer_than_a_command_line_is_cut_and_says_so() -> None:
    """Twenty ids, because the rest would be a shell argument nobody reads."""
    said = "\n".join(report([judged(one, label=HALLUCINATED) for one in range(1, 26)]))

    assert "IDS=" + ",".join(str(one) for one in range(1, 21)) + "..." in said
