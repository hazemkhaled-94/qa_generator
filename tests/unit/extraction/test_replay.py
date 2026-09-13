"""Re-judging a stored fact.

The re-judgement rebuilds the candidate from the columns the first one wrote,
so a citation has to survive that round trip.
"""

from __future__ import annotations

import pytest
from factories import passage

from extraction.models import CandidateFact
from extraction.validation import FactChecker

pytestmark = pytest.mark.nlp


@pytest.mark.parametrize(
    ("statement", "cited"),
    [
        ("The device weighs 4 kg.", (0,)),
        ("The device weighs 7 kg.", (0,)),
        ("It arrives in March 2026.", (1,)),
        ("The device arrives in March 2026.", (0, 1)),
        ("Anything.", (7,)),
        ("Anything.", ()),
    ],
)
def test_a_second_judgement_reaches_the_first_verdict(statement, cited) -> None:
    """The verdict, the code and the resolved span do not move."""
    checker = FactChecker()
    prose = passage()

    first = checker.check(prose, CandidateFact(statement, cited), "llm")
    # What the catalogue hands back: the statement and the citation as stored.
    again = checker.check(
        prose,
        CandidateFact(first.statement, tuple(first.evidence_sentence_ids)),
        first.extraction_method,
    )

    assert again.validated == first.validated
    assert again.rejection_code == first.rejection_code
    assert again.evidence_text == first.evidence_text
    assert again.evidence_sentence_ids == first.evidence_sentence_ids
    assert again.units_added == first.units_added
