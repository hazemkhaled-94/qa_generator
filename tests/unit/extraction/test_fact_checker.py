"""The verdict a proposed fact is judged to.

A rejected fact's code is the quality signal this project reports, so each
failure must reach its own code rather than share a bucket with another.
"""

from __future__ import annotations

import pytest
from factories import passage

from database.qa_generator import Rejection
from extraction.models import CandidateFact
from extraction.validation import FactChecker

pytestmark = pytest.mark.nlp


@pytest.fixture(scope="module")
def checker() -> FactChecker:
    """The checker under test."""
    return FactChecker()


@pytest.fixture(scope="module")
def prose():
    """Two English sentences, the first carrying two claims."""
    return passage()


def test_one_claim_drawn_from_its_sentence_is_accepted(checker, prose) -> None:
    """The fact is validated and carries the span its citation covers."""
    good = checker.check(prose, CandidateFact("The device weighs 4 kg.", (0,)), "llm")

    assert good.validated, (good.rejection_code, good.validation_error)
    assert good.evidence_text == "The device weighs 4 kg and runs for 12 hours."
    assert good.evidence_sentence_ids == [0]
    assert good.statement_predicates == 1
    assert good.evidence_predicates == 2, "the cited sentence carries two claims"
    assert not good.units_added, good.units_added
    assert good.rejection_code is None and good.validation_error is None


def test_the_source_own_words_are_accepted_when_they_narrow_the_claim(
    checker, prose
) -> None:
    """One claim taken out of a sentence carrying two is still a fact."""
    narrowed = checker.check(
        prose, CandidateFact("The device runs for 12 hours.", (0,)), "llm"
    )
    assert narrowed.validated, (narrowed.rejection_code, narrowed.validation_error)


@pytest.mark.parametrize("cited", [(7,), ()])
def test_a_citation_naming_no_sentence_of_the_passage_is_refused(
    checker, prose, cited
) -> None:
    """Out of range and absent are the same failure."""
    judged = checker.check(prose, CandidateFact("Anything.", cited), "llm")
    assert judged.rejection_code == Rejection.EVIDENCE_ABSENT
    assert not judged.validated


def test_evidence_copied_rather_than_written_is_refused(checker, prose) -> None:
    """A statement equal to evidence carrying two claims restates it."""
    copied = checker.check(
        prose,
        CandidateFact("The device weighs 4 kg and runs for 12 hours.", (0,)),
        "llm",
    )
    assert copied.rejection_code == Rejection.COPIED, copied.validation_error


def test_a_sentence_carrying_one_claim_may_be_quoted(checker) -> None:
    """Evidence that is already one claim is the fact; restating two is not."""
    quoted = passage(
        "The market presented a mixed picture in 2025. "
        "Transaction volumes remained low and prices fell further."
    )
    assert [s.predicates for s in quoted.sentences] == [1, 2], quoted.sentences

    verbatim = checker.check(
        quoted, CandidateFact(quoted.sentences[0].text, (0,)), "llm"
    )
    assert verbatim.validated, (verbatim.rejection_code, verbatim.validation_error)

    restated = checker.check(
        quoted, CandidateFact(quoted.sentences[1].text, (1,)), "llm"
    )
    assert restated.rejection_code == Rejection.COPIED, restated.validation_error


def test_two_claims_in_one_statement_are_refused(checker, prose) -> None:
    """Two claims are two facts."""
    both = checker.check(
        prose,
        CandidateFact("The device weighs 4 kg and it runs for 12 hours.", (0,)),
        "llm",
    )
    assert both.rejection_code == Rejection.NOT_ATOMIC, both.validation_error
    assert both.statement_predicates == 2, both.statement_predicates


def test_a_statement_with_no_finite_verb_is_refused(checker, prose) -> None:
    """Naming something is not asserting anything about it."""
    naming = checker.check(prose, CandidateFact("The 4 kg device.", (0,)), "llm")
    assert naming.rejection_code == Rejection.NOT_ATOMIC, naming.validation_error


@pytest.mark.parametrize(
    ("statement", "cited", "added"),
    [
        ("The device weighs 7 kg.", (0,), "7"),
        ("The Bundesbank delivers the device in March 2026.", (1,), "bundesbank"),
    ],
)
def test_a_unit_the_cited_text_does_not_carry_is_refused(
    checker, prose, statement, cited, added
) -> None:
    """A number or a name the source never gave is an unsupported addition."""
    invented = checker.check(prose, CandidateFact(statement, cited), "llm")
    assert invented.rejection_code == Rejection.UNSUPPORTED_ADDITION, (
        invented.validation_error
    )
    assert added in invented.units_added, invented.units_added


def test_a_year_ending_a_sentence_is_the_same_year(checker) -> None:
    """A year with a full stop attached is the same unit as the bare year."""
    dated = passage(
        "Im Jahr 2026 überwacht die Bafin die Kreditrisiken der Institute. "
        "Der Bericht erscheint spaeter.",
        language="de",
    )
    ending = checker.check(
        dated,
        CandidateFact("Die Bafin überwacht die Kreditrisiken im Jahr 2026.", (0,)),
        "llm",
    )
    assert not ending.units_added, ending.units_added
    assert ending.validated, (ending.rejection_code, ending.validation_error)


@pytest.mark.parametrize(
    "statement",
    [
        "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen.",
        "Die Digitalisierung schreitet fort.",
    ],
)
def test_neither_a_common_noun_nor_a_verb_is_a_unit(checker, statement) -> None:
    """The two sides are parsed separately; only units are compared."""
    german = passage(
        "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen "
        "und fortschreitender Digitalisierung. Die Bafin beobachtet das.",
        language="de",
    )
    echoed = checker.check(german, CandidateFact(statement, (0,)), "llm")
    assert not echoed.units_added, (statement, echoed.units_added)


def test_a_statement_that_cannot_be_read_alone_is_refused(checker, prose) -> None:
    """An unresolved pronoun cannot become a question."""
    dangling = checker.check(
        prose, CandidateFact("It arrives in March 2026.", (1,)), "llm"
    )
    assert dangling.rejection_code == Rejection.UNRESOLVED_REFERENCE, (
        dangling.validation_error
    )
    assert "it" in dangling.unresolved_references, dangling.unresolved_references


@pytest.mark.parametrize(
    ("statement", "cited"),
    [
        ("Ein zentrales Risiko ergibt sich aus Handelskonflikten.", 0),
        ("Es besteht Potenzial für plötzliche Marktkorrekturen.", 1),
    ],
)
def test_a_german_reflexive_or_expletive_is_not_a_reference(
    checker, statement, cited
) -> None:
    """A reflexive belongs to its verb and an expletive stands in for nothing."""
    german = passage(
        "Ein zentrales Risiko ergibt sich aus Handelskonflikten. "
        "Es besteht Potenzial für plötzliche Marktkorrekturen.",
        language="de",
    )
    # Cited from the other sentence, so the copy check leaves it alone.
    judged = checker.check(german, CandidateFact(statement, (1 - cited,)), "llm")
    assert judged.rejection_code != Rejection.UNRESOLVED_REFERENCE, (
        f"{statement!r}: {judged.unresolved_references}"
    )


def test_two_cited_sentences_span_from_the_first_to_the_last(checker, prose) -> None:
    """The evidence runs from the start of the first to the end of the last."""
    joined = checker.check(
        prose, CandidateFact("The device arrives in March 2026.", (0, 1)), "llm"
    )
    assert joined.evidence_sentence_ids == [0, 1]
    assert joined.evidence_start == 0
    assert joined.evidence_end == len(prose.text)


def test_a_deterministic_statement_is_judged_on_its_citation_alone(
    checker, prose
) -> None:
    """A statement composed from a grid is neither a sentence nor written."""
    composed = checker.check(
        prose, CandidateFact("Device - Mass: 4 kg", (0,)), "deterministic"
    )
    assert composed.validated, (composed.rejection_code, composed.validation_error)


def test_each_failure_reaches_its_own_code(checker, prose) -> None:
    """Five refusals, five codes, every one declared."""
    codes = {
        checker.check(prose, CandidateFact(statement, cited), "llm").rejection_code
        for statement, cited in (
            ("Anything.", (7,)),
            ("The device weighs 4 kg and runs for 12 hours.", (0,)),
            ("The device weighs 4 kg and it runs for 12 hours.", (0,)),
            ("The device weighs 7 kg.", (0,)),
            ("It arrives in March 2026.", (1,)),
        )
    }
    assert len(codes) == 5, codes
    assert all(code in set(Rejection) for code in codes), codes
