"""Re-judging a stored fact.

The re-judgement rebuilds the candidate from the columns the first one wrote,
so a citation has to survive that round trip. What the model wrote is kept;
only what the checks read off it is replaced.
"""

from __future__ import annotations

import pytest
from drivers import DIGEST_SHARE, group, passage

from database.qa_generator import FactKind
from extraction.models import CandidateFact, Cited
from extraction.validation import FactChecker

pytestmark = pytest.mark.nlp


def same(first, again) -> None:
    """Asserts that a second judgement reached the first verdict."""
    assert again.validated == first.validated
    assert again.rejection_code == first.rejection_code
    assert again.evidence_text == first.evidence_text
    assert again.evidence_sentence_ids == first.evidence_sentence_ids
    assert again.units_added == first.units_added
    assert again.kind == first.kind
    assert again.citations == first.citations


@pytest.mark.parametrize(
    ("statement", "cited", "kind"),
    [
        ("The device weighs 4 kg.", (0,), FactKind.ATOMIC),
        ("The device weighs 7 kg.", (0,), FactKind.ATOMIC),
        ("It arrives in March 2026.", (1,), FactKind.ATOMIC),
        ("The device arrives in March 2026.", (0, 1), FactKind.ATOMIC),
        ("Anything.", (7,), FactKind.ATOMIC),
        ("Anything.", (), FactKind.ATOMIC),
        ("The device weighs 4 kg.", (0, 1), FactKind.SUMMARY),
        ("- Weighs 4 kg\n- Ships", (0, 1), FactKind.OUTLINE),
        ("- Weighs 4 kg", (0, 1), FactKind.OUTLINE),
    ],
)
def test_a_second_judgement_reaches_the_first_verdict(statement, cited, kind) -> None:
    """The verdict, the code and the resolved span do not move."""
    checker = FactChecker(DIGEST_SHARE)
    prose = passage()

    first = checker.check(prose, CandidateFact(statement, cited, kind=kind), "llm")
    # What the catalogue hands back: the statement, the kind and the citation
    # as stored.
    again = checker.check(
        prose,
        CandidateFact(
            first.statement, tuple(first.evidence_sentence_ids), kind=first.kind
        ),
        first.extraction_method,
    )

    same(first, again)


@pytest.mark.parametrize(
    ("statement", "rests_on"),
    [
        ("Support response times are stated separately for both kinds.", (0, 1)),
        ("Standard requests are answered within 48 hours.", (0,)),
        ("Anything at all happens.", ()),
    ],
)
def test_a_bridge_is_judged_the_same_way_from_its_stored_group(
    statement, rests_on
) -> None:
    """The group comes back from fact_passages, in the order it was shown."""
    checker = FactChecker(DIGEST_SHARE)
    offered = group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )
    candidate = CandidateFact(
        statement,
        (),
        kind=FactKind.BRIDGE,
        passages=tuple(
            Cited(position=position, sentences=(0,)) for position in rests_on
        ),
    )

    first = checker.check_bridge(offered, candidate)
    # The catalogue rebuilds the positions and the citations from whatever
    # the link table held for this fact.
    held = [one.passage_id for one in first.citations]
    stored = [one for one in offered if one.id in held]
    again = checker.check_bridge(
        stored,
        CandidateFact(
            first.statement,
            (),
            kind=FactKind.BRIDGE,
            passages=tuple(
                Cited(position=position, sentences=tuple(one.sentence_ids))
                for position, one in enumerate(first.citations)
            ),
        ),
    )

    same(first, again)


def test_a_re_judgement_never_rewrites_what_the_model_wrote() -> None:
    """The statement and the kind are the record of one extraction."""
    checker = FactChecker(DIGEST_SHARE)
    prose = passage()

    first = checker.check(
        prose,
        CandidateFact("The device weighs 7 kg.", (0,), kind=FactKind.ATOMIC),
        "llm",
    )
    again = checker.check(
        prose,
        CandidateFact(first.statement, (0,), kind=first.kind),
        first.extraction_method,
    )

    assert again.statement == first.statement
    assert again.kind == first.kind
    assert again.extraction_method == first.extraction_method


def test_a_change_to_the_share_changes_the_verdict_without_a_model() -> None:
    """This is what re-judging is for: today's checks over yesterday's facts."""
    prose = passage()
    digest = CandidateFact(
        "The device weighs 4 kg. It arrives in March 2026.",
        (0, 1),
        kind=FactKind.SUMMARY,
    )

    assert FactChecker(0.6).check(prose, digest, "llm").validated
    assert not FactChecker(0.2).check(prose, digest, "llm").validated
