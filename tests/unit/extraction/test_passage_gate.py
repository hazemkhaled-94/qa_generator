"""Which passages never reach the model.

One model call costs minutes, and a heading or a caption returns its own text
back as a fact.
"""

from __future__ import annotations

import pytest
from factories import passage

from extraction.service import skipped

pytestmark = pytest.mark.nlp


def test_a_passage_carrying_a_claim_is_extracted() -> None:
    """Prose asserting something is not skipped."""
    assert skipped(passage()) is None


@pytest.mark.parametrize(
    "text", ["3.2 Lieferung und Versand", "Tabelle 1: Modelle", "Risiko"]
)
def test_a_passage_with_no_finite_verb_is_skipped(text: str) -> None:
    """A heading or a caption asserts nothing."""
    assert skipped(passage(text, language="de")) == "no finite verb", text


def test_a_passage_repeating_its_own_heading_trail_is_skipped() -> None:
    """A heading that carries a verb is still a heading."""
    heading = passage(
        "The device arrives in March.",
        section_path="Intro > The device arrives in March.",
    )
    assert skipped(heading) == "heading", skipped(heading)


def test_a_table_is_not_judged_on_its_sentences() -> None:
    """A table is read from its grid."""
    assert skipped(passage("| a | b |", block_type="table")) is None


@pytest.mark.parametrize("kwargs", [{}, {"block_type": "table"}])
def test_a_passage_with_nothing_numbered_is_skipped(kwargs) -> None:
    """Nothing can cite a passage that has no numbered units, table or not."""
    assert skipped(passage("| a | b |", sentences=[], **kwargs)) == "no sentences"


@pytest.mark.parametrize("block_type", ["document_index", "code"])
def test_a_navigation_passage_is_skipped(block_type: str) -> None:
    """A table of contents says where something is, not what it says."""
    listing = passage("Das Team managen dauert 225 Minuten.", block_type=block_type)

    assert skipped(listing) == f"{block_type}: navigation rather than content"


def test_a_navigation_passage_is_skipped_before_its_sentences_are_read() -> None:
    """It is refused on what kind of block it is, not on what it says."""
    prose = passage(
        "The device weighs 4 kg and runs for 12 hours.", block_type="document_index"
    )

    assert prose.claims, "this passage does assert something"
    assert skipped(prose) is not None


# ── Navigation the parser did not label ────────────────────────────────────
#
# Every text below is a real passage from a corpus this was measured on,
# shortened. The parser labelled each of them `text`, so the block-type
# check above never saw them; between them they produced questions asking
# which pages a term appears on and how the document is licensed.


def test_an_index_the_parser_labelled_prose_is_skipped() -> None:
    """A term and the pages it appears on, over and over."""
    index = passage(
        "Überdeckung 22, 23, 46, 47, 48, 49, 50, 53 Überdeckung aller Übergänge 49 "
        "rundreiseüberdeckung, 35 schlüsselwort, 13 grenzwertanalyse, 30 "
        "grundursachenanalyse, 58, 76 abstrakter testfall, 13, 18 ad-hoc-review, 52 "
        "technisches review 42 test 16, 17 frühes testen 20, 21 fehlerzustand 19, 24",
        language="de",
    )

    assert skipped(index) is not None


def test_a_bibliography_is_skipped() -> None:
    """Names, years and page ranges, which is not a claim about anything."""
    works = passage(
        "- BAKER, Paul; DAI, Zhen Ru; GRABOWSKI, Jens, 2008. Model-Driven Testing. "
        "Springer, pp. 12-48. - ALYAHYA, Sultan, 2020. Crowdsourced Software "
        "Testing: A Systematic Literature Review. In: Journal 61, pp. 1-22. "
        "- Kahler, T. (2008). The Process Therapy Model. Taibi, 1st ed., 340 pp.",
        language="en",
    )

    assert skipped(works) is not None


def test_a_copyright_notice_is_skipped() -> None:
    """Front matter asserts things, and they are about the document.

    It has to be caught by the glyph rather than by the verb check above:
    a real notice is a paragraph of prose that says who holds what, so it
    carries finite verbs and reads as content to everything else here.
    """
    notice = passage(
        "Dieser Lehrplan ist urheberrechtlich geschützt. Das Urheberrecht © 2019 "
        "gehört den Autoren der englischen Originalausgabe. Die Nutzung ist nur "
        "mit ausdrücklicher Zustimmung der Inhaber gestattet, und jede Ausgabe "
        "nennt die Organisation, die sie herausgegeben hat.",
        language="de",
    )

    assert skipped(notice) == "a copyright notice, which is about the document"


def test_a_bullet_list_of_content_is_not_skipped() -> None:
    """The false positive the numeric half of the rule exists to prevent.

    A German bullet list is sparse in finite verbs because its points are
    infinitives - `Evaluieren von Arbeitsergebnissen` asserts nothing a
    parser can count - so claim density alone reads it as a list. It
    carries no numbers, and an index is nothing but numbers.
    """
    objectives = passage(
        "Typische Testziele sind: - Evaluieren von Arbeitsergebnissen wie "
        "Anforderungen, User-Storys und Code - Auslösen von Fehlerwirkungen und "
        "Auffinden von Fehlerzuständen - Sicherstellen der erforderlichen "
        "Überdeckung eines Testobjekts - Verringern des Risikos einer "
        "unzureichenden Softwarequalität - Zusammenarbeit mit Stakeholdern, um "
        "die Erfüllung der Akzeptanzkriterien zu überprüfen",
        language="de",
    )

    assert skipped(objectives) is None


def test_prose_carrying_a_reference_is_not_skipped() -> None:
    """One citation in a sentence does not make the sentence a bibliography."""
    prose = passage(
        "KI kann mit einer breiten Palette von Techniken implementiert werden "
        "(siehe [B02] für weitere Informationen), und die Wahl der Technik "
        "bestimmt, wie das System getestet werden muss.",
        language="de",
    )

    assert skipped(prose) is None
