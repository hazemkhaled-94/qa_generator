"""How well a real served model reads facts out of a passage.

Never a gate. A model's answers move between versions, between quantisations
and between runs at the same temperature, so a threshold here would fail on
somebody else's Tuesday rather than on a regression. It is a measurement,
and the numbers are what it prints.

Needs a model: set LLM_MODEL and LLM_BASE_URL and run

    make test-eval

Skipped otherwise, which is every CI run.
"""

from __future__ import annotations

import os

import pytest
from factories import passage

from evaluation.cases import EXTRACTION as GOLDEN
from extraction.validation import FactChecker

pytestmark = [pytest.mark.eval, pytest.mark.nlp]

#: The cases live in evaluation/cases.py: `make eval-experiment` reads
#: the same ones, and one definition is what keeps the two measuring the
#: same thing.


@pytest.fixture(scope="module")
def extractor():
    """The real extractor against the served model, or a skip."""
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is unset; no model to measure")

    from extraction.extractors.llm import LlmExtractor
    from llm.client import Client, ModelUnavailable
    from llm.config import Settings

    built = LlmExtractor(Client(Settings.load()))
    try:
        built.extract(passage("The device weighs 4 kg.", language="en"))
    except ModelUnavailable as exc:
        pytest.skip(f"the model is not answering: {exc}")
    return built


@pytest.mark.parametrize("example", GOLDEN, ids=lambda one: one["language"])
def test_the_model_reads_the_claims_a_passage_carries(extractor, example) -> None:
    """Reports what it found, and fails only on finding nothing at all.

    Counting exactly is what a person does when reviewing; what can be
    asserted without a person is that the model answered and that every
    answer survived the checks it will be held to.
    """
    checker = FactChecker()
    under = passage(example["text"], language=example["language"])

    candidates = extractor.extract(under)
    checked = [checker.check(under, one, "llm") for one in candidates]
    kept = [one for one in checked if one.validated]

    print(
        f"\n{example['language']}: {len(candidates)} proposed, "
        f"{len(kept)} validated, {example['claims']} expected"
    )
    for one in checked:
        mark = "ok " if one.validated else one.rejection_code
        print(f"  [{mark}] {one.statement}")

    assert candidates, "the model proposed nothing at all"
    assert kept, "not one proposal survived the checks"


def test_the_whole_golden_set_is_scored(extractor) -> None:
    """One number for the set, printed rather than asserted against."""
    checker = FactChecker()
    proposed = validated = expected = 0

    for example in GOLDEN:
        under = passage(example["text"], language=example["language"])
        candidates = extractor.extract(under)
        proposed += len(candidates)
        validated += sum(
            1 for one in candidates if checker.check(under, one, "llm").validated
        )
        expected += example["claims"]

    print(
        f"\ngolden set: {proposed} proposed, {validated} validated, "
        f"{expected} expected; "
        f"precision {validated / proposed:.0%}, recall {validated / expected:.0%}"
    )

    assert proposed, "the model proposed nothing for any passage"
