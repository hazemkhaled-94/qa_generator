"""The verdict a proposed fact is judged to.

A rejected fact's code is the quality signal this project reports, so each
failure must reach its own code rather than share a bucket with another.
"""

from __future__ import annotations

import pytest
from drivers import Checker, group, passage

from database.qa_generator import FactKind, Rejection

pytestmark = pytest.mark.nlp


@pytest.fixture(scope="module")
def checker() -> Checker:
    """Two English sentences, the first carrying two claims."""
    return Checker()


def test_one_claim_drawn_from_its_sentence_is_accepted(checker) -> None:
    """The fact is validated and carries the span its citation covers."""
    good = checker.atomic("The device weighs 4 kg.")

    assert good.validated, (good.rejection_code, good.validation_error)
    assert good.evidence_text == "The device weighs 4 kg and runs for 12 hours."
    assert good.evidence_sentence_ids == [0]
    assert good.statement_predicates == 1
    assert good.evidence_predicates == 2, "the cited sentence carries two claims"
    assert not good.units_added, good.units_added
    assert good.rejection_code is None and good.validation_error is None
    assert good.kind == FactKind.ATOMIC
    assert good.passage_ids == [], "a single-passage fact records no group"


def test_the_source_own_words_are_accepted_when_they_narrow_the_claim(checker) -> None:
    """One claim taken out of a sentence carrying two is still a fact."""
    narrowed = checker.atomic("The device runs for 12 hours.")
    assert narrowed.validated, (narrowed.rejection_code, narrowed.validation_error)


@pytest.mark.parametrize("cited", [(7,), (), (-1,), (0, 9)])
def test_a_citation_naming_no_sentence_of_the_passage_is_refused(
    checker, cited
) -> None:
    """Out of range, negative and absent are the same failure."""
    judged = checker.atomic("Anything.", cited)
    assert judged.rejection_code == Rejection.EVIDENCE_ABSENT
    assert not judged.validated


def test_evidence_copied_rather_than_written_is_refused(checker) -> None:
    """A statement equal to evidence carrying two claims restates it."""
    copied = checker.atomic("The device weighs 4 kg and runs for 12 hours.")
    assert copied.rejection_code == Rejection.COPIED, copied.validation_error


def test_a_sentence_carrying_one_claim_may_be_quoted() -> None:
    """Evidence that is already one claim is the fact; restating two is not."""
    quoted = Checker(
        passage(
            "The market presented a mixed picture in 2025. "
            "Transaction volumes remained low and prices fell further."
        )
    )
    assert [s.predicates for s in quoted.passage.sentences] == [1, 2]

    verbatim = quoted.atomic(quoted.passage.sentences[0].text, 0)
    assert verbatim.validated, (verbatim.rejection_code, verbatim.validation_error)

    restated = quoted.atomic(quoted.passage.sentences[1].text, 1)
    assert restated.rejection_code == Rejection.COPIED, restated.validation_error


def test_two_claims_in_one_statement_are_refused(checker) -> None:
    """Two claims are two facts."""
    both = checker.atomic("The device weighs 4 kg and it runs for 12 hours.")
    assert both.rejection_code == Rejection.NOT_ATOMIC, both.validation_error
    assert both.statement_predicates == 2, both.statement_predicates


def test_a_statement_with_no_finite_verb_is_refused(checker) -> None:
    """Naming something is not asserting anything about it."""
    naming = checker.atomic("The 4 kg device.")
    assert naming.rejection_code == Rejection.ASSERTS_NOTHING, naming.validation_error
    assert naming.statement_predicates == 0


@pytest.mark.parametrize(
    ("statement", "cited", "added"),
    [
        ("The device weighs 7 kg.", 0, "7"),
        ("The Bundesbank delivers the device in March 2026.", 1, "bundesbank"),
    ],
)
def test_a_unit_the_cited_text_does_not_carry_is_refused(
    checker, statement, cited, added
) -> None:
    """A number or a name the source never gave is an unsupported addition."""
    invented = checker.atomic(statement, cited)
    assert invented.rejection_code == Rejection.UNSUPPORTED_ADDITION, (
        invented.validation_error
    )
    assert added in invented.units_added, invented.units_added


def test_a_year_ending_a_sentence_is_the_same_year() -> None:
    """A year with a full stop attached is the same unit as the bare year."""
    dated = Checker(
        passage(
            "Im Jahr 2026 überwacht die Bafin die Kreditrisiken der Institute. "
            "Der Bericht erscheint spaeter.",
            language="de",
        )
    )
    ending = dated.atomic("Die Bafin überwacht die Kreditrisiken im Jahr 2026.", 0)

    assert not ending.units_added, ending.units_added
    assert ending.validated, (ending.rejection_code, ending.validation_error)


@pytest.mark.parametrize(
    "statement",
    [
        "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen.",
        "Die Digitalisierung schreitet fort.",
    ],
)
def test_neither_a_common_noun_nor_a_verb_is_a_unit(statement) -> None:
    """The two sides are parsed separately; only units are compared."""
    german = Checker(
        passage(
            "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen "
            "und fortschreitender Digitalisierung. Die Bafin beobachtet das.",
            language="de",
        )
    )
    echoed = german.atomic(statement, 0)
    assert not echoed.units_added, (statement, echoed.units_added)


def test_a_statement_that_cannot_be_read_alone_is_refused(checker) -> None:
    """An unresolved pronoun cannot become a question."""
    dangling = checker.atomic("It arrives in March 2026.", 1)
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
def test_a_german_reflexive_or_expletive_is_not_a_reference(statement, cited) -> None:
    """A reflexive belongs to its verb and an expletive stands in for nothing."""
    german = Checker(
        passage(
            "Ein zentrales Risiko ergibt sich aus Handelskonflikten. "
            "Es besteht Potenzial für plötzliche Marktkorrekturen.",
            language="de",
        )
    )
    # Cited from the other sentence, so the copy check leaves it alone.
    judged = german.atomic(statement, 1 - cited)
    assert judged.rejection_code != Rejection.UNRESOLVED_REFERENCE, (
        f"{statement!r}: {judged.unresolved_references}"
    )


def test_two_cited_sentences_span_from_the_first_to_the_last(checker) -> None:
    """The evidence runs from the start of the first to the end of the last."""
    joined = checker.atomic("The device arrives in March 2026.", (0, 1))

    assert joined.evidence_sentence_ids == [0, 1]
    assert joined.evidence_start == 0
    assert joined.evidence_end == len(checker.passage.text)


def test_a_citation_naming_one_sentence_twice_resolves_once(checker) -> None:
    """A repeated index is one sentence, not a span of two."""
    repeated = checker.atomic("The device weighs 4 kg.", (0, 0))
    assert repeated.evidence_sentence_ids == [0]


def test_a_deterministic_statement_is_judged_on_its_citation_alone(checker) -> None:
    """A statement composed from a grid is neither a sentence nor written."""
    composed = checker.composed("Device - Mass: 4 kg")
    assert composed.validated, (composed.rejection_code, composed.validation_error)


def test_a_deterministic_statement_faces_no_check_a_writer_faces(checker) -> None:
    """Every written check would refuse this; none of them is applied."""
    composed = checker.composed("Device - Mass: 7 kg - it")
    assert composed.validated, (composed.rejection_code, composed.validation_error)
    assert composed.units_added, "the reading is still recorded"


def test_a_deterministic_copy_of_its_evidence_is_still_refused(checker) -> None:
    """The one check a composed statement does face."""
    copied = checker.composed("The device weighs 4 kg and runs for 12 hours.")
    assert copied.rejection_code == Rejection.COPIED, copied.validation_error


class TestDigest:
    """A summary and an outline stand in for a whole passage."""

    def test_a_condensed_summary_of_the_passage_is_accepted(self, checker) -> None:
        """Shorter than its passage, asserting something, inventing nothing."""
        made = checker.summary("The device weighs 4 kg. It arrives in March 2026.")

        assert made.validated, (made.rejection_code, made.validation_error)
        assert made.kind == FactKind.SUMMARY
        assert made.evidence_text == checker.passage.text
        assert made.evidence_sentence_ids == [0, 1], "a digest cites the whole passage"

    def test_a_condensed_outline_of_the_passage_is_accepted(self, checker) -> None:
        """Two points, shorter than the passage, inventing nothing."""
        made = checker.outline("Weighs 4 kg", "Arrives in March 2026")

        assert made.validated, (made.rejection_code, made.validation_error)
        assert made.kind == FactKind.OUTLINE
        assert made.evidence_text == checker.passage.text

    def test_several_claims_are_what_a_summary_is_for(self, checker) -> None:
        """The atomicity check is not applied to a digest."""
        several = checker.summary("The device weighs 4 kg. It arrives in March 2026.")

        assert several.statement_predicates > 1, several.statement_predicates
        assert several.validated, several.rejection_code

    def test_a_dangling_pronoun_is_allowed_inside_a_summary(self, checker) -> None:
        """A summary names its subject once and may refer back to it."""
        referring = checker.summary("The device weighs 4 kg. It arrives in March 2026.")

        assert referring.unresolved_references, "the pronoun is still recorded"
        assert referring.validated, referring.rejection_code

    def test_a_digest_as_long_as_its_passage_is_refused(self, checker) -> None:
        """Something the length of what it replaces saves a reader nothing."""
        copied = checker.summary(checker.passage.text)

        assert copied.rejection_code == Rejection.NOT_CONDENSED, copied.validation_error
        assert "%" in (copied.validation_error or ""), "the measurement is reported"

    def test_a_digest_asserting_nothing_is_refused(self, checker) -> None:
        """A noun phrase is a label, not a reading."""
        naming = checker.summary("The 4 kg device from Hamburg.")
        assert naming.rejection_code == Rejection.ASSERTS_NOTHING

    def test_a_digest_inventing_a_value_is_refused(self, checker) -> None:
        """Condensing a passage may not add to it."""
        invented = checker.summary("The device weighs 9 kg.")

        assert invented.rejection_code == Rejection.UNSUPPORTED_ADDITION
        assert "9" in invented.units_added, invented.units_added

    def test_bullets_are_read_as_one_statement(self, checker) -> None:
        """An outline is one fact whose statement is a list."""
        listed = checker.outline("Weighs 4 kg", "Runs for 12 hours")

        assert listed.validated, (listed.rejection_code, listed.validation_error)
        assert listed.statement == "- Weighs 4 kg\n- Runs for 12 hours"

    def test_a_bullet_written_as_a_fragment_is_not_asked_for_a_verb(
        self, checker
    ) -> None:
        """A fragment parses as having none, so no outline would ever pass."""
        from nlp.analysis import claim

        fragments = checker.outline("Weighs 4 kg", "Runs for 12 hours")

        assert claim(fragments.statement, "en").predicates == 0, (
            "the parse this check would have been read off"
        )
        assert fragments.validated, fragments.rejection_code

    def test_an_outline_of_one_point_is_a_label(self, checker) -> None:
        """A list of one is not a list."""
        single = checker.outline("Weighs 4 kg")
        assert single.rejection_code == Rejection.NOT_LISTED, single.validation_error

    def test_an_outline_that_lost_its_bullets_is_refused(self, checker) -> None:
        """The shape is what is checked, so the shape has to be there."""
        loose = checker.raw_outline("Weighs 4 kg\nRuns for 12 hours")
        assert loose.rejection_code == Rejection.NOT_LISTED, loose.validation_error

    def test_a_digest_of_a_passage_with_no_sentences_is_refused(self) -> None:
        """There is nothing to stand in for."""
        empty = Checker(passage("| a | b |", sentences=[]))
        assert empty.summary("Anything at all.").rejection_code == (
            Rejection.EVIDENCE_ABSENT
        )


@pytest.fixture(scope="module")
def offered():
    """Two passages of two documents, as a topic group offers them."""
    return group(
        "Standard requests are answered within 48 hours.",
        "Urgent requests are answered within 4 hours.",
    )


class TestBridge:
    """A bridge carries a claim no single passage states."""

    def test_a_claim_resting_on_both_passages_is_accepted(
        self, checker, offered
    ) -> None:
        """It records every passage it rests on, anchored to the first."""
        bridged = checker.bridge(
            "Support response times are stated separately for standard and "
            "urgent requests.",
            offered,
        )

        assert bridged.validated, (bridged.rejection_code, bridged.validation_error)
        assert bridged.kind == FactKind.BRIDGE
        assert bridged.passage_ids == [11, 22]
        assert bridged.passage_id == 11, "the anchor is the first passage offered"

    def test_the_anchor_evidence_is_the_whole_of_its_passage(
        self, checker, offered
    ) -> None:
        """So the span still resolves in the passage the fact points at."""
        bridged = checker.bridge("Two request kinds are named.", offered)

        assert bridged.evidence_text == offered[0].text
        assert bridged.evidence_start == 0
        assert bridged.evidence_end == len(offered[0].text)
        assert offered[0].text[bridged.evidence_start : bridged.evidence_end] == (
            bridged.evidence_text
        )

    def test_a_claim_resting_on_one_passage_is_refused(self, checker, offered) -> None:
        """That is an ordinary fact, and the atomic pass has it already."""
        alone = checker.bridge(
            "Standard requests are answered within 48 hours.", offered, rests_on=(0,)
        )
        assert alone.rejection_code == Rejection.NOT_BRIDGING, alone.validation_error

    @pytest.mark.parametrize("named", [(), (7,), (-1,)])
    def test_a_claim_naming_no_offered_passage_is_refused(
        self, checker, offered, named
    ) -> None:
        """Nothing resolves, so nothing supports it."""
        nowhere = checker.bridge("Anything at all happens.", offered, rests_on=named)
        assert nowhere.rejection_code == Rejection.EVIDENCE_ABSENT
        assert nowhere.passage_ids == []

    def test_a_unit_from_the_second_passage_is_supported(
        self, checker, offered
    ) -> None:
        """The vocabulary checked against is the union of every passage."""
        across = checker.bridge("An urgent request waits 4 hours.", offered)

        assert not across.units_added, across.units_added
        assert across.validated, across.rejection_code

    def test_a_computed_value_is_in_neither_passage(self, checker, offered) -> None:
        """Arithmetic is not reading, so a total nobody wrote is an addition."""
        summed = checker.bridge("The two times differ by 44 hours.", offered)

        assert summed.rejection_code == Rejection.UNSUPPORTED_ADDITION
        assert "44" in summed.units_added, summed.units_added

    def test_two_claims_in_one_bridge_are_refused(self, checker, offered) -> None:
        """A bridge is atomic, like any other claim."""
        both = checker.bridge(
            "Standard requests wait 48 hours and urgent requests wait 4 hours.",
            offered,
        )
        assert both.rejection_code == Rejection.NOT_ATOMIC, both.validation_error

    def test_a_group_naming_the_positions_out_of_order_still_anchors_first(
        self, checker, offered
    ) -> None:
        """Position 0 is the anchor whichever order the model named them in."""
        reversed_order = checker.bridge(
            "Two request kinds are named.", offered, rests_on=(1, 0)
        )
        assert reversed_order.passage_ids == [11, 22]


def test_each_failure_reaches_its_own_code(checker, offered) -> None:
    """Every rejection code the checks can emit, and no bucket shared."""
    codes = {
        checker.atomic("Anything.", 7).rejection_code,
        checker.atomic("The device weighs 4 kg and runs for 12 hours.").rejection_code,
        checker.atomic("The 4 kg device.").rejection_code,
        checker.atomic(
            "The device weighs 4 kg and it runs for 12 hours."
        ).rejection_code,
        checker.atomic("The device weighs 7 kg.").rejection_code,
        checker.atomic("It arrives in March 2026.", 1).rejection_code,
        checker.summary(checker.passage.text).rejection_code,
        checker.outline("Weighs 4 kg").rejection_code,
        checker.bridge(
            "Standard requests are answered within 48 hours.", offered, rests_on=(0,)
        ).rejection_code,
    }

    assert codes == set(Rejection), sorted(set(Rejection) - codes)
