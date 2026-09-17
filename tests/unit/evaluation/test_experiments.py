"""The scores an experiment records, and the shape it records them in.

Pinned because a wrong evaluator is worse than no evaluator: the numbers
still appear, still plot, and still get compared - they just measure
something else. Neither of these calls a model.
"""

from __future__ import annotations

import pytest

from evaluation import cases, experiments


def test_recall_is_what_was_found_over_what_is_there() -> None:
    """Two of four claims is half of them."""
    assert experiments._recall({"validated": 2}, {"claims": 4}) == 0.5


def test_recall_is_capped_at_everything() -> None:
    """Recall is capped at everything there was to find.

    A model proposing six facts for a passage carrying four has not
    achieved 150% recall: it has over-decomposed, and that belongs in
    precision. Uncapped, a model that split every sentence into three
    would score best on a metric meant to catch the opposite.
    """
    assert experiments._recall({"validated": 6}, {"claims": 4}) == 1.0


def test_recall_of_a_passage_carrying_nothing_is_zero_not_an_error() -> None:
    """A heading has no claims, and a division here would take the run."""
    assert experiments._recall({"validated": 0}, {"claims": 0}) == 0.0


def test_precision_is_what_survived_over_what_was_proposed() -> None:
    """Three of four proposals passing the checks is 75%."""
    assert experiments._precision({"proposed": 4, "validated": 3}, {}) == 0.75


def test_precision_of_nothing_proposed_is_zero_not_an_error() -> None:
    """Precision of nothing proposed is zero, not an error.

    A model that answered with nothing scores nothing, and does not stop
    the experiment the other cases are in.
    """
    assert experiments._precision({"proposed": 0, "validated": 0}, {}) == 0.0


@pytest.mark.parametrize("name", experiments.NAMES)
def test_every_dataset_splits_into_inputs_and_expectations(name: str) -> None:
    """Phoenix keeps them apart, and so must this.

    A task is handed the input and never the expectation. An expectation
    that leaked into the input would let a task read the answer, and the
    experiment would score the leak.
    """
    inputs, outputs = experiments._examples(name)

    assert len(inputs) == len(outputs), name
    assert inputs and outputs, name
    for given, expected in zip(inputs, outputs, strict=True):
        assert not set(given) & set(expected), f"{name}: {set(given) & set(expected)}"


def test_the_extraction_inputs_carry_no_claim_count() -> None:
    """The one leak that would matter: a task told how many to find."""
    inputs, _ = experiments._examples(experiments.EXTRACTION)

    assert all("claims" not in one for one in inputs)


def test_the_question_inputs_carry_no_verdict() -> None:
    """`recoverable` is the answer, and belongs only in the expectation."""
    inputs, outputs = experiments._examples(experiments.QUESTIONS)

    assert all("recoverable" not in one for one in inputs)
    assert all("recoverable" in one for one in outputs)


def test_every_case_reaches_a_dataset() -> None:
    """Every case reaches a dataset.

    One added to evaluation/cases.py and not to _examples is a case
    nothing scores.
    """
    assert len(experiments._examples(experiments.EXTRACTION)[0]) == len(
        cases.EXTRACTION
    )
    assert len(experiments._examples(experiments.QUESTIONS)[0]) == len(cases.QUESTIONS)


def test_the_cases_are_data_and_cost_nothing_to_read() -> None:
    """evaluation/cases.py imports no model, converter or gate.

    Read as syntax rather than by importing, like
    tests/static/test_api_stays_light.py, so a failure names the import
    that added the weight. The file is read by the eval tests and by the
    experiment runner, and one that loaded torch to hand over three
    passages would make `--help` take a minute.
    """
    import ast
    from pathlib import Path

    source = Path(cases.__file__).read_text()
    imported = {
        name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    heavy = imported & {
        "torch",
        "transformers",
        "litellm",
        "instructor",
        "spacy",
        "docling",
        "gensim",
        "extraction",
        "question_generation",
        "nlp",
        "llm",
    }

    assert not heavy, f"evaluation/cases.py imports {', '.join(sorted(heavy))}"
