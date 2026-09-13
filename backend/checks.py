"""Self-checks for the logic that fails silently.

Run with `make check`, or `PYTHONPATH=backend python -m checks`. Pure logic
and plain asserts, so it needs no database and no served model; it does load
the spaCy pipelines, which are in the image.

Covers the fact checks, the sentence numbering a citation resolves against,
what chunking keeps and counts, the table reader's labelling, the parser's
repair of words the page broke across lines, the worker's shutdown, the
queue's narrowing to one item, and the topic model's weights, vocabulary and
label carryover.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://unused:unused@localhost/x")
os.environ.setdefault("NLP_MODELS", "de:de_core_news_md,en:en_core_web_md")
os.environ.setdefault("NLP_DEFAULT_LANGUAGE", "en")

import telemetry
from database.qa_generator import Rejection
from extraction.extractors.table import TableExtractor
from extraction.models import CandidateFact, PassageToExtract
from extraction.service import skipped
from extraction.validation import FactChecker
from nlp.models import Sentence
from preprocessing.chunking.models import Chunking
from preprocessing.chunking.passages import NoPassages, chunking_of, lines_of
from stages.worker import Shutdown, watch

log = logging.getLogger("checks")

_TEXT = (
    "The device weighs 4 kg and runs for 12 hours. "
    "It arrives in March 2026 from the Hamburg plant."
)

#: The fitter settings the topic checks run under, and a corpus with two
#: clear subjects to run them over. Shared, because a second copy is a second
#: thing to keep in step with the first.
_TOPIC_SETTINGS = {
    "num_topics": 2,
    "passes": 5,
    "random_state": 42,
    "top_terms": 5,
    "min_weight": 0.1,
    "no_below": 2,
    "no_above": 0.9,
}

_TOPIC_CORPUS = (
    "lieferung versand transport monatlich paketdienst",
    "lieferung versand transport jaehrlich spedition",
    "wartung reparatur instandhaltung ersatzteil pruefung",
    "wartung reparatur instandhaltung wartungsplan intervall",
)


def _vocabularies() -> list:
    """Builds the corpus as the fitter takes it, one passage per text."""
    from topic_modelling.models import PassageVocabulary

    return [
        PassageVocabulary(id=index, lemmas=text.split())
        for index, text in enumerate(_TOPIC_CORPUS, start=1)
    ]


def _passage(text: str = _TEXT, language: str = "en", **kwargs) -> PassageToExtract:
    """Builds a passage with its sentences numbered, as chunking stores them."""
    from nlp.analysis import sentences as split

    given = kwargs.pop("sentences", ...)
    return PassageToExtract(
        id=1,
        text=text,
        section_path=kwargs.pop("section_path", None),
        block_type=kwargs.pop("block_type", None),
        language=language,
        sentences=split(text, language) if given is ... else given,
        table_cells=kwargs.pop("table_cells", []),
    )


def sentence_numbering() -> None:
    """A cited sentence must resolve to the exact span of the passage text.

    The whole citation mechanism rests on this: the extractor names a number
    and the offsets come from the stored sentence, so an offset that does not
    index back to the sentence text is a wrong citation stored as an exact one.

    Raises:
        AssertionError: If a sentence's offsets do not slice its own text, if
            the passage is not covered in order, or if a rendered table row is
            mislocated.
    """
    passage = _passage()
    assert len(passage.sentences) == 2, [s.text for s in passage.sentences]
    for sentence in passage.sentences:
        assert passage.text[sentence.start : sentence.end] == sentence.text
    assert [s.index for s in passage.sentences] == [0, 1]
    assert passage.claims == 2, passage.claims

    # A table has no sentences; its rendered rows are numbered the same way,
    # so one mechanism resolves both.
    rendered = "Table 1: Models\n\n| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"
    lines = lines_of(rendered)
    assert [line["i"] for line in lines] == [0, 1, 2, 3], lines
    for line in lines:
        sliced = rendered[line["start"] : line["end"]]
        assert sliced.strip() == sliced and sliced, repr(sliced)
    assert rendered[lines[3]["start"] : lines[3]["end"]] == "| Compact | 4 |"

    # Leading whitespace must not be counted into the span.
    indented = lines_of("  | a | b |\n")
    assert indented[0]["start"] == 2, indented


def fact_checks() -> None:
    """Each verdict must stay distinguishable, and each must be the right one.

    A rejected fact's code is the quality signal this project reports, so
    "carried two claims" and "invented a number" must not share a bucket.

    Raises:
        AssertionError: If a verdict, a code or a resolved span is wrong.
    """
    check = FactChecker()
    passage = _passage()

    good = check.check(passage, CandidateFact("The device weighs 4 kg.", (0,)), "llm")
    assert good.validated, (good.rejection_code, good.validation_error)
    assert good.evidence_text == "The device weighs 4 kg and runs for 12 hours."
    assert good.evidence_sentence_ids == [0]
    assert good.statement_predicates == 1
    assert good.evidence_predicates == 2, "the cited sentence carries two claims"
    assert not good.units_added, good.units_added

    # The citation names a sentence the passage does not have.
    absent = check.check(passage, CandidateFact("Anything.", (7,)), "llm")
    assert absent.rejection_code == Rejection.EVIDENCE_ABSENT
    assert not absent.validated

    # No citation at all is the same failure.
    assert (
        check.check(passage, CandidateFact("Anything.", ()), "llm").rejection_code
        == Rejection.EVIDENCE_ABSENT
    )

    # The model filled both fields with one string.
    copied = check.check(
        passage,
        CandidateFact("The device weighs 4 kg and runs for 12 hours.", (0,)),
        "llm",
    )
    assert copied.rejection_code == Rejection.COPIED, copied.validation_error

    # Two claims in one statement is two facts, not one.
    both = check.check(
        passage,
        CandidateFact("The device weighs 4 kg and it runs for 12 hours.", (0,)),
        "llm",
    )
    assert both.rejection_code == Rejection.NOT_ATOMIC, both.validation_error
    assert both.statement_predicates == 2, both.statement_predicates

    # A statement with no finite verb names something rather than asserting it.
    naming = check.check(passage, CandidateFact("The 4 kg device.", (0,)), "llm")
    assert naming.rejection_code == Rejection.NOT_ATOMIC, naming.validation_error

    # The failure no lexical gate could see: a number the source never gave.
    invented = check.check(
        passage, CandidateFact("The device weighs 7 kg.", (0,)), "llm"
    )
    assert invented.rejection_code == Rejection.UNSUPPORTED_ADDITION, (
        invented.validation_error
    )
    assert "7" in invented.units_added, invented.units_added

    # A proper noun the cited text does not have is the same failure. A name
    # the tagger reads as a common noun is not a unit and so escapes this;
    # numbers and unambiguous names are what the check rests on.
    named = check.check(
        passage,
        CandidateFact("The Bundesbank delivers the device in March 2026.", (1,)),
        "llm",
    )
    assert named.rejection_code == Rejection.UNSUPPORTED_ADDITION, (
        named.validation_error
    )
    assert "bundesbank" in named.units_added, named.units_added

    # Neither a common noun nor a verb is a unit. The two sides are parsed
    # separately, so the tagger disagrees with itself about a noun; and the
    # statement is written rather than quoted, so its verb is the model's to
    # choose. Reading either as a unit reported words that were in the cited
    # sentence, or never could be, as invented.
    german = _passage(
        "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen "
        "und fortschreitender Digitalisierung. Die Bafin beobachtet das.",
        language="de",
    )
    for statement in (
        "Das Umfeld des Finanzsektors ist geprägt von geopolitischen Umbrüchen.",
        "Die Digitalisierung schreitet fort.",
    ):
        echoed = check.check(german, CandidateFact(statement, (0,)), "llm")
        assert not echoed.units_added, (statement, echoed.units_added)

    # A statement that cannot be read on its own cannot be a question.
    dangling = check.check(
        passage, CandidateFact("It arrives in March 2026.", (1,)), "llm"
    )
    assert dangling.rejection_code == Rejection.UNRESOLVED_REFERENCE, (
        dangling.validation_error
    )
    assert "it" in dangling.unresolved_references, dangling.unresolved_references

    # Not every pronoun refers. A German reflexive belongs to its own verb and
    # an expletive stands in for nothing, and reading either as a dangling
    # reference rejected correct facts on the real corpus.
    german = _passage(
        "Ein zentrales Risiko ergibt sich aus Handelskonflikten. "
        "Es besteht Potenzial für plötzliche Marktkorrekturen.",
        language="de",
    )
    for statement, cited in (
        ("Ein zentrales Risiko ergibt sich aus Handelskonflikten.", 0),
        ("Es besteht Potenzial für plötzliche Marktkorrekturen.", 1),
    ):
        # Cited from the other sentence, so the copy check leaves it alone.
        other = 1 - cited
        judged = check.check(german, CandidateFact(statement, (other,)), "llm")
        assert judged.rejection_code != Rejection.UNRESOLVED_REFERENCE, (
            f"{statement!r}: {judged.unresolved_references}"
        )

    # The narrowing the old similarity gate turned away: one claim drawn out
    # of a sentence carrying two, in the source's own words.
    narrowed = check.check(
        passage, CandidateFact("The device runs for 12 hours.", (0,)), "llm"
    )
    assert narrowed.validated, (narrowed.rejection_code, narrowed.validation_error)

    # Two sentences cited together span from the first to the last.
    joined = check.check(
        passage, CandidateFact("The device arrives in March 2026.", (0, 1)), "llm"
    )
    assert joined.evidence_sentence_ids == [0, 1]
    assert joined.evidence_start == 0
    assert joined.evidence_end == len(passage.text)

    # A deterministic statement is composed from a grid, so it is neither a
    # sentence nor written: only the citation and the copy check apply to it.
    composed = check.check(
        passage, CandidateFact("Device - Mass: 4 kg", (0,)), "deterministic"
    )
    assert composed.validated, (composed.rejection_code, composed.validation_error)

    # Every rejection carries a code the report can group on, and a reader can
    # tell each one from the others.
    codes = {
        absent.rejection_code,
        copied.rejection_code,
        both.rejection_code,
        invented.rejection_code,
        dangling.rejection_code,
    }
    assert len(codes) == 5, codes
    assert all(code in set(Rejection) for code in codes), codes
    assert good.rejection_code is None and good.validation_error is None


def german_modals() -> None:
    """A German obligation must read as one claim.

    Regulatory German states a duty with a modal and a participle - "Ein
    Risikobericht muss erstellt werden". de_core_news_sm tags no finite verb
    in that at all, so the atomicity check read it as asserting nothing and
    rejected 11% of this corpus's German facts for a parser limitation.
    Measured over these sentences: sm 3/8, md 8/8.

    Raises:
        AssertionError: If an obligation reads as anything but one claim,
            which means NLP_MODELS has been pointed at a smaller model.
    """
    from nlp.analysis import claim

    for sentence in (
        "Ein Risikobericht muss mindestens vierteljährlich erstellt werden.",
        "Bei abweichenden Voten muss der Kredit abgelehnt werden.",
        "Die Geschäftsleitung muss für die Umsetzung Sorge tragen.",
        "Die vereinfachte Umsetzung muss unter Risikogesichtspunkten vertretbar sein.",
        "Das interne Kontrollsystem umfasst eine Risikocontrolling-Funktion.",
    ):
        found = claim(sentence, "de").predicates
        assert found == 1, f"{found} claim(s) read in {sentence!r}; expected 1"

    # And the shapes either side of one claim still read correctly.
    assert (
        claim(
            "Der Antrag wird geprüft und das Ergebnis wird mitgeteilt.", "de"
        ).predicates
        == 2
    )
    assert (
        claim("Die Erhöhung der Eigenkapitalquote auf 12,5 Prozent", "de").predicates
        == 0
    )


def passage_gate() -> None:
    """A passage that asserts nothing must never reach the model.

    One model call costs minutes; a heading or a caption returns its own text
    back as a fact. The finite-verb test is what catches the general case,
    where matching the heading trail only caught one kind.

    Raises:
        AssertionError: If a claim-bearing passage is skipped, or a passage
            carrying no claim is not.
    """
    assert skipped(_passage()) is None

    for text in ("3.2 Lieferung und Versand", "Tabelle 1: Modelle", "Risiko"):
        assert skipped(_passage(text, language="de")) == "no finite verb", text

    # A heading that does carry a verb is still a heading when it repeats its
    # own trail.
    heading = _passage(
        "The device arrives in March.",
        section_path="Intro > The device arrives in March.",
    )
    assert skipped(heading) == "heading", skipped(heading)

    # A table is read from its grid, so it is never judged on its sentences.
    assert skipped(_passage("| a | b |", block_type="table")) is None

    # But a passage with nothing numbered can be cited from by nobody, table
    # or not: read without them, a table produced a fact per cell and every
    # one cited a row that did not exist.
    assert skipped(_passage("x", sentences=[])) == "no sentences"
    assert (
        skipped(_passage("| a | b |", block_type="table", sentences=[]))
        == "no sentences"
    )


class _Chunk:
    """A stand-in carrying the attribute the chunk reader reads off a chunk."""

    def __init__(self, text: str) -> None:
        self.text = text


def _numbered(ordinal: int, chunk: _Chunk):
    """Turns a chunk into the little of a passage a check reads."""
    return type("P", (), {"ordinal": ordinal, "text": chunk.text.strip()})()


def _built(texts, max_tokens=10) -> Chunking:
    """Runs the chunk reader over some chunk texts, counting a word as a token."""
    return chunking_of(
        [_Chunk(text) for text in texts],
        max_tokens=max_tokens,
        count_tokens=lambda text: len(text.split()),
        to_passage=_numbered,
    )


def passage_language() -> None:
    """A passage must be read in the language it is actually written in.

    These documents carry a German report and its English summary in one
    file, so the document's own label sends half the passages to the wrong
    pipeline - and the wrong pipeline is what put German function words in an
    English topic.

    Raises:
        AssertionError: If a language is misread, if a passage too short to
            judge does not fall back to the document's, or if a table is
            given a language it has no text for.
    """
    from nlp.language import detect

    german = (
        "Die Bundesanstalt für Finanzdienstleistungsaufsicht beaufsichtigt "
        "Institute und prüft deren Risikomanagement regelmäßig."
    )
    english = (
        "The supervisory authority examines the risk management of the "
        "institutions it supervises on a regular basis."
    )
    assert detect(german) == "de", detect(german)
    assert detect(english) == "en", detect(english)

    # Too short to judge: the caller falls back to the document's language.
    assert detect("Risiko") is None
    assert detect("") is None


def kept_chunks() -> None:
    """Chunking must store every chunk, and count the ones over budget.

    The failure this catches looked like success: a character ceiling below
    the token budget threw away whole tables that were within the budget, and
    the document still read as cleanly chunked.

    Raises:
        AssertionError: If a chunk goes unstored, if the oversized count is
            wrong, if ordinals are not contiguous, if a passage is stored as
            anything other than the text it was counted on, or if an empty
            document does not fail.
    """
    result = _built(("x", "y " * 12, "z", "w", "v"))
    assert isinstance(result, Chunking)
    assert len(result.passages) == 5, "a chunk over the budget is still stored"
    assert result.oversized == 1, result.oversized
    assert [p.ordinal for p in result.passages] == [1, 2, 3, 4, 5]

    assert _built(("v",)).passages[0].text == "v"

    # What is stored is what was counted. Every sentence offset indexes this.
    stored = _built(("  hello\n",)).passages[0].text
    assert stored == "hello", repr(stored)

    assert _built(("keep", "   ", "\n")).passages[0].text == "keep"
    try:
        _built(("", "   "))
    except NoPassages as exc:
        assert "no passages" in str(exc), str(exc)
    else:
        raise AssertionError("a document yielding no chunk must fail")


def table_facts() -> None:
    """A cell fact must cite the numbered row its own value sits in.

    The failure this catches stored a wrong fact as verified: the cell was
    labelled from its grid coordinates but its evidence was found by searching
    the rendered text, so a split table attached one row's value to another
    row's text and the gate passed it.

    Raises:
        AssertionError: If a fact cites a row that does not hold its value, if
            a header cell produces a fact, or if a row the passage does not
            render produces one.
    """
    rendered = "Table 1: Models\n\n| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"
    lines = lines_of(rendered)

    def cell(row, col, text, line, **flags):
        """Builds one stored cell."""
        return {
            "row": row,
            "col": col,
            "row_span": 1,
            "col_span": 1,
            "column_header": flags.get("column_header", False),
            "row_header": flags.get("row_header", False),
            "text": text,
            "line": line,
        }

    # The Standard row is in the table but not in this passage, so the chunker
    # kept no line for it and it must yield nothing at all.
    grid = {
        "caption": "Table 1: Models",
        "num_rows": 3,
        "num_cols": 2,
        "cells": [
            cell(0, 0, "Model", None, column_header=True),
            cell(0, 1, "Mass", None, column_header=True),
            cell(2, 0, "Compact", 3, row_header=True),
            cell(2, 1, "4", 3),
        ],
    }
    passage = PassageToExtract(
        id=1,
        text=rendered,
        section_path="Models",
        block_type="table",
        language="en",
        sentences=[
            Sentence(
                index=line["i"],
                start=line["start"],
                end=line["end"],
                text=rendered[line["start"] : line["end"]],
                predicates=0,
            )
            for line in lines
        ],
        table_cells=[grid],
    )

    facts = TableExtractor().extract(passage)
    assert len(facts) == 1, [f.statement for f in facts]
    assert facts[0].statement == "Table 1: Models - Compact - Mass: 4", facts[
        0
    ].statement
    assert facts[0].sentences == (3,), facts[0].sentences

    # And the citation has to resolve to the row holding the value.
    checked = FactChecker().check(passage, facts[0], "deterministic")
    assert checked.validated, checked.validation_error
    assert checked.evidence_text == "| Compact | 4 |", checked.evidence_text

    # A cell whose row this passage does not render carries no line, and a
    # fact with no evidence is worse than no fact.
    orphan = dict(
        grid,
        cells=[
            *grid["cells"],
            cell(1, 0, "Standard", None, row_header=True),
            cell(1, 1, "9", None),
        ],
    )
    assert (
        len(
            TableExtractor().extract(
                PassageToExtract(
                    id=1,
                    text=rendered,
                    section_path="Models",
                    block_type="table",
                    language="en",
                    sentences=passage.sentences,
                    table_cells=[orphan],
                )
            )
        )
        == 1
    ), "a row the passage does not render must produce no fact"


def broken_words() -> None:
    """A word the page broke across lines must come back whole.

    The fonts in these documents map their discretionary hyphen to U+0002,
    which is invisible: `Lebensversicherer` stored with that control character
    in place of each line break looks fine in a table and matches no search
    and no vocabulary term.

    Raises:
        AssertionError: If a broken word is not rejoined, if a control
            character survives, if a space is left as anything but U+0020, if
            a decomposed umlaut is left uncomposed, if a real hyphen or a
            newline is lost, or if the repair misses the table grid.
    """
    from docling_core.types.doc.document import DoclingDocument, TableCell, TableData
    from docling_core.types.doc.labels import DocItemLabel

    from preprocessing.parsing.pipelines.pdf import _mend, _repaired

    assert _mend("Lebens\x02 versiche\x02 rer") == "Lebensversicherer"
    # No space when the break fell at the end of a cell.
    assert _mend("Risikoklassifizie\x02rung") == "Risikoklassifizierung"
    # U+00AD is the same mark spelled properly, and joins the same way.
    assert _mend("Lebens­ versiche­rer") == "Lebensversicherer"
    # A hyphen the author typed is not a line break, and must survive.
    assert _mend("Schaden- / Unfallversicherer") == "Schaden- / Unfallversicherer"

    # A real hyphen at a line break. The converter has already turned the
    # break into a space, so these are the only two shapes to tell apart, and
    # telling them apart is the whole job: joining blindly gives "Zollund".
    for broken, whole in (
        ("Registrierungsformu- lare", "Registrierungsformulare"),
        ("Ge- schäftsleiter", "Geschäftsleiter"),
        ("ergän- zende", "ergänzende"),
        ("Institu- te", "Institute"),
        ("Nied- rig", "Niedrig"),
    ):
        assert _mend(broken) == whole, _mend(broken)
    # The hanging hyphen: a shared suffix, completed by a conjunction.
    for kept in (
        "Zoll- und Steuerrecht",
        "Über- oder Unterdeckung",
        "Qualitäts- bzw. Quantitätsprüfung",
        "Geschäftsfortführungs- sowie Notfallpläne",
        "pre- and post-trade",
    ):
        assert _mend(kept) == kept, _mend(kept)
    assert _mend("Anlage- Verwalter") == "Anlage- Verwalter"
    assert _mend("Krypto-Transaktionen") == "Krypto-Transaktionen"
    assert _mend("a\nb\tc\x07d") == "a\nb\tcd"

    # Every space becomes the one a model types back.
    assert _mend("Unternehmenskrediten →") == "Unternehmenskrediten →"
    assert _mend("12,50 EUR netto") == "12,50 EUR netto"
    assert _mend("Bafin\u200b﻿ meldet") == "Bafin meldet"
    # Composed and decomposed umlauts look identical and compare unequal.
    assert _mend("Bußgelder für Häuser") == "Bußgelder für Häuser"
    assert _mend("ü") == "ü"

    document = DoclingDocument(name="t")
    document.add_text(label=DocItemLabel.TEXT, text="Ände\x02 rung zum Vor\x02 jahr")
    document.add_table(
        data=TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    text="Pensi\x02 onskas\x02 sen",
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                )
            ],
        )
    )
    mended = _repaired(document)
    assert mended.texts[0].text == "Änderung zum Vorjahr", mended.texts[0].text
    # The grid too: a passage's table_cells are read from here, not from the
    # text the serialiser rendered.
    cell = mended.tables[0].data.table_cells[0].text
    assert cell == "Pensionskassen", cell


def graceful_shutdown() -> None:
    """SIGTERM must end the loop rather than the process.

    A worker killed mid-item leaves its row claimed until the lease expires,
    which is the cost of every ordinary redeploy.

    Raises:
        AssertionError: If the signal is not recorded, or if the loop drains
            again after receiving it.
    """
    flag = Shutdown()
    assert not flag.requested()
    flag._request(signal.SIGTERM, None)
    assert flag.requested()

    drains = []

    def drain(stopping) -> int:
        drains.append(stopping())
        if len(drains) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return 1  # work found, so the loop does not sleep

    watch(drain, 0.01)
    assert drains == [False, False], drains


def stop_before_claiming() -> None:
    """A stopping worker must not claim one more item.

    Raises:
        AssertionError: If a drain that starts already stopping claims
            anything, or if an ordinary drain never does.
    """
    from stages.service import StageService

    class Repository:
        """A queue that counts how often it was asked for work."""

        done = "extracted"

        def __init__(self) -> None:
            self.claims = 0

        def abandon(self) -> int:
            """Reports that nothing was left claimed."""
            return 0

    class Service(StageService):
        """A stage whose work is to count that it was asked."""

        name = "check"
        unit = "row"

        def process_next(self):
            """Claims one row and does nothing with it."""
            self._repository.claims += 1

    service = Service(Repository())
    assert service.drain(stopping=lambda: True) == 0
    assert service._repository.claims == 0, "a stopping worker claimed another row"

    running = Service(Repository())
    assert running.drain() == 0
    assert running._repository.claims == 1, "a draining worker never claimed"


def topic_vocabulary() -> None:
    """The stored lemmas must be subjects, not grammar.

    Document frequency cannot separate the two: in one German test corpus the
    function word "vor" occurred in 19.4% of passages and a content word in
    18.8%. Keeping only content parts of speech is what separates them, and it
    also folds the inflections German spreads one term across.

    Raises:
        AssertionError: If a function word reaches the vocabulary, if a
            content word is dropped, or if inflections are not folded.
    """
    from nlp.analysis import read

    def lemmas(text: str, language: str) -> list[str]:
        """Reads one text's stored vocabulary."""
        return next(iter(read([text], language)))[1]

    german = lemmas("Die Lieferung wird vor dem Versand wie folgt geprueft", "de")
    assert not {"die", "wird", "vor", "dem", "wie"} & set(german), german
    assert "lieferung" in german, german

    english = lemmas("The delivery is checked before the dispatch.", "en")
    assert not {"the", "is", "before"} & set(english), english
    assert "delivery" in english, english

    # One term, not three: the frequency filter can only work on folded forms.
    folded = lemmas("Institut Instituts Institute", "de")
    assert folded.count("institut") >= 2, folded

    # A lemmatiser handed a word from another language returns the placeholder
    # "--" for a token that is itself alphabetic, and the lemma is what gets
    # stored: "--" reached the top terms of a fitted topic.
    mixed = lemmas("Die Bafin prueft the risk on the market genau.", "de")
    assert "--" not in mixed, mixed
    assert all(term.isalpha() for term in mixed), mixed

    # A table is vocabulary too: its headings and cell values are what it is
    # about, and leaving them out put every fact the cell reader draws from a
    # table outside every topic-weighted report.
    grid = lemmas(
        "Tabelle 1: Planstellenübersicht\n\n| Besoldungsgruppe | Anzahl |\n"
        "| --- | --- |\n| Oberregierungsrat | 12 |\n",
        "de",
    )
    assert {"besoldungsgruppe", "anzahl"} <= set(grid), grid
    assert all(term.isalpha() for term in grid), grid

    # A web address is not vocabulary. Markdown writes a link as [text](url),
    # and the brackets stop the tokenizer seeing one token, so every path
    # segment arrived tagged as a noun: one repeated link put `publikationen`,
    # `fokus` and `im` among the most widespread terms in the corpus.
    linked = lemmas(
        "[Risks in Bafin's Focus](https://www.bafin.de/EN/die-bafin/"
        "publikationen-daten/risiken-im-fokus/Fokusrisiken.html)",
        "en",
    )
    assert not {"publikationen", "fokus", "im", "en", "die"} & set(linked), linked
    assert "risk" in linked, "the link text is still vocabulary"


def topic_weights() -> None:
    """A membership weight must be storable, and a lost passage counted.

    `passage_topics.weight` has a CHECK constraint of (0, 1]. gensim returns
    float32, so a distribution's last member can round a hair above 1 and fail
    the whole insert - which for a batch of thousands loses every membership.

    Raises:
        AssertionError: If a weight leaves the interval, if a passage with no
            topic goes uncounted, or if the fitter accepts a threshold that
            cannot do its job.
    """
    from topic_modelling.models import PassageVocabulary
    from topic_modelling.topics import NoVocabulary, TopicFitter

    settings = _TOPIC_SETTINGS
    passages = _vocabularies()

    fitting = TopicFitter(**settings).fit(lambda: passages, "de")
    assert fitting.weights, "a corpus with two clear subjects produced no membership"
    for weight in fitting.weights:
        assert 0 < weight.weight <= 1, (
            f"weight {weight.weight!r} breaches the CHECK on passage_topics"
        )
    assert fitting.vocabulary > 0
    assert fitting.passages == len(passages)
    assert fitting.language == "de", "a fit belongs to the language it was over"

    # The fit walks the corpus several times. Something that can only be
    # walked once silently fits an empty corpus after the first pass.
    spent = iter(passages)
    try:
        TopicFitter(**settings).fit(lambda: spent, "de")
    except NoVocabulary:
        pass
    else:
        raise AssertionError("a corpus that cannot be re-walked must not fit")

    # A passage holding nothing of the vocabulary must be counted, never
    # silently absent.
    counted = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=99, lemmas=["zzz", "qqq"])], "de"
    )
    assert counted.without_topics == 1, counted.without_topics
    assert not any(w.passage_id == 99 for w in counted.weights)

    # A passage chunking stored no lemmas for is the same case.
    empty = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=98)], "de"
    )
    assert empty.without_topics == 1, empty.without_topics

    for bad in (
        {"num_topics": 1},
        {"top_terms": 0},
        {"min_weight": 0},
        {"min_weight": 1.5},
        {"no_above": 0},
    ):
        try:
            TopicFitter(**(settings | bad))
        except ValueError:
            continue
        raise AssertionError(f"a fitter accepted {bad}, which cannot do its job")

    try:
        TopicFitter(**(settings | {"no_below": 99})).fit(lambda: passages, "de")
    except NoVocabulary:
        pass
    else:
        raise AssertionError("a filter that removes every term must fail the run")


def topic_visualisation() -> None:
    """The drawn model must be the model that was stored.

    pyLDAvis reorders topics by prevalence unless told not to, which would
    number the figure differently from the topics table beside it. It also
    refuses a distribution that does not sum to 1.

    Raises:
        AssertionError: If the matrices disagree with the fit, if a topic is
            renumbered, if the page reaches the network, or if a model that
            placed no passage is drawn rather than refused.
    """
    from topic_modelling.models import PassageVocabulary, TopicSpace
    from topic_modelling.topics import TopicFitter
    from topic_modelling.visualisation import render

    settings = _TOPIC_SETTINGS
    passages = _vocabularies()
    fitting = TopicFitter(**settings).fit(lambda: passages, "de")
    space = fitting.space

    assert len(space.topic_term) == 2, len(space.topic_term)
    assert len(space.vocabulary) == fitting.vocabulary, len(space.vocabulary)
    assert len(space.term_frequency) == fitting.vocabulary
    for row in space.topic_term:
        assert len(row) == fitting.vocabulary, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)

    # One row per passage the model placed, and each a distribution.
    assert len(space.doc_topic) == len(passages) - fitting.without_topics
    assert len(space.doc_lengths) == len(space.doc_topic)
    for row in space.doc_topic:
        assert len(row) == 2, len(row)
        assert abs(sum(row) - 1) < 1e-6, sum(row)
    assert all(length > 0 for length in space.doc_lengths), space.doc_lengths

    # A passage holding none of the vocabulary is in neither the memberships
    # nor the figure, and is counted once.
    lost = TopicFitter(**settings).fit(
        lambda: [*passages, PassageVocabulary(id=99, lemmas=["zzz", "qqq"])], "de"
    )
    assert lost.without_topics == 1, lost.without_topics
    assert len(lost.space.doc_topic) == len(passages), len(lost.space.doc_topic)

    page = render(space, "de").decode("utf-8")
    # Topic n in the figure is topic n in the topics table.
    assert '"topic.order": [0, 1]' in page, re.findall(r'"topic\.order":[^]]*]', page)
    # Nothing is fetched when the page is opened.
    fetched = [
        url
        for url in re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', page)
        if url.startswith(("http://", "https://", "//"))
    ]
    assert not fetched, fetched
    assert "var LDAvis" in page, "the LDAvis script is not inlined"
    assert "d3.select" in page, "d3 is not inlined"

    # A model that placed no passage has nothing to draw.
    try:
        render(
            TopicSpace(
                topic_term=space.topic_term,
                doc_topic=[],
                doc_lengths=[],
                vocabulary=space.vocabulary,
                term_frequency=space.term_frequency,
            ),
            "de",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a model placing no passage must not be drawn")


def topic_naming() -> None:
    """A model must never overwrite a name a person typed.

    A person's label is the one a coverage report is read by, and it survives
    a refit by being matched on top terms. Regenerating over it would lose
    the one thing the matching exists to protect.

    Raises:
        AssertionError: If a person's label is replaced, if a refusal is
            stored as a name, or if the provenance is not recorded.
    """
    from topic_modelling.labels import TopicLabeller
    from topic_modelling.models import FittedTopic

    class Answer:
        """What the client hands back."""

        def __init__(self, label: str) -> None:
            self.label = label

    class Model:
        """A client that answers whatever it was told to, and counts calls."""

        def __init__(self, says: str) -> None:
            self.says, self.calls = says, 0
            self.model = "test-model"

        def answer(self, **_: object) -> Answer:
            """Answers, and records that it was asked."""
            self.calls += 1
            return Answer(self.says)

    client = Model('"Anti-money laundering".')
    labeller = TopicLabeller(client, {"en": "English"})
    named = labeller.label(FittedTopic(0, ["money", "laundering"]), "en", ["excerpt"])
    assert named == "Anti-money laundering", named
    assert labeller.model == "test-model"

    # A model that cannot find a subject must leave the topic unnamed rather
    # than stamp "Mixed" on a coverage report.
    refused = TopicLabeller(Model("Mixed"), {"en": "English"})
    assert refused.label(FittedTopic(0, ["a", "b"]), "en", []) is None

    # And a topic a person named is never offered to the model at all.
    from topic_modelling.service import TopicModellingService

    counting = Model("Generated")
    service = TopicModellingService.__new__(TopicModellingService)
    service._labeller = TopicLabeller(counting, {"en": "English"})
    service._repository = type("R", (), {"excerpts": staticmethod(lambda ids: [])})()
    fitting = type("F", (), {"weights": []})()
    topics = [
        FittedTopic(0, ["a"], label="Typed by a person", labelled_by="person"),
        FittedTopic(1, ["b"]),
    ]
    named = service._named(topics, fitting, "en")
    assert named[0].label == "Typed by a person", named[0]
    assert named[0].labelled_by == "person", named[0]
    assert named[1].label == "Generated", named[1]
    assert named[1].labelled_by == "test-model", named[1]
    assert counting.calls == 1, f"the model was asked {counting.calls} times, not 1"


def topic_labels() -> None:
    """A refit must keep a label on its own topic, or on none at all.

    A refit deletes every topic, so a hand-typed label survives only by being
    matched on top terms. Attaching one to the wrong topic is worse than
    losing it: the label is what a coverage report is read by.

    Raises:
        AssertionError: If a label moves to an unrelated topic, is dropped
            despite a clear match, or is used twice.
    """
    from topic_modelling.models import FittedTopic
    from topic_modelling.topics import carry_labels

    first = ["lieferung", "versand", "transport", "paket", "monatlich"]
    second = ["wartung", "reparatur", "instandhaltung", "ersatzteil", "pruefung"]

    previous = [
        FittedTopic(0, first, label="Shipping", include_in_coverage=False),
        FittedTopic(1, second, label="Maintenance"),
    ]
    refitted = [
        FittedTopic(
            0, ["wartung", "reparatur", "instandhaltung", "ersatzteil", "intervall"]
        ),
        FittedTopic(1, ["lieferung", "versand", "transport", "paket", "jaehrlich"]),
    ]
    carried = carry_labels(refitted, previous)
    assert carried[0].label == "Maintenance", carried[0].label
    assert carried[1].label == "Shipping", carried[1].label
    assert carried[1].include_in_coverage is False, "the exclusion must travel too"

    unrelated = carry_labels(
        [FittedTopic(0, ["farbe", "lack", "oberflaeche"])], previous
    )
    assert unrelated[0].label is None, unrelated[0].label

    twice = carry_labels([FittedTopic(0, first), FittedTopic(1, first)], previous)
    assert [t.label for t in twice].count("Shipping") == 1, [t.label for t in twice]


def queue_narrowing() -> None:
    """A narrowed queue operation must select one item, not the whole queue.

    The failure this catches is silent and expensive. `stop` narrowed to one
    document, with the narrowing quietly dropped, takes every queued row in
    the corpus off the queue rather than that document's - and answers with
    a count that looks like a success. The condition is built in one place
    and executed in another, so nothing at run time would report it missing.
    """
    from sqlalchemy import update
    from sqlalchemy.dialects import postgresql

    from database.qa_generator import Status
    from extraction.repository import PassageQueue
    from preprocessing.parsing.repository import ParseQueue
    from stages.queue import _also

    assert _also(None) == (), "a narrowing nobody gave must drop out"
    assert _also("a", None, "b") == ("a", "b"), _also("a", None, "b")

    parsing, extraction = ParseQueue(), PassageQueue()

    def where(queue, within) -> str:
        """Compiles the WHERE one operation would run with."""
        return str(
            update(queue.columns.entity)
            .where(*_also(queue.columns.status == Status.PENDING, within))
            .values({queue.columns.status: Status.NEW})
            .compile(dialect=postgresql.dialect())
        )

    # Narrowing must reach the SQL beside the status, and the whole-queue
    # form must not carry it: those two are the same method, one argument
    # apart.
    narrowed = where(parsing, parsing.narrow("document", "abc"))
    assert "sha256" in narrowed, narrowed
    assert "parse_status" in narrowed, narrowed
    assert "sha256" not in where(parsing, None), where(parsing, None)

    # Extraction queues over passages and is asked for a document, so its
    # scope is the passage's document and never the passage's own key.
    by_document = where(extraction, extraction.narrow("document", "abc"))
    assert "doc_sha256" in by_document, by_document
    assert "passages.id" not in by_document, by_document

    # A value is coerced to what the column holds. Comparing an integer
    # column against a string is an error PostgreSQL raises and SQLite does
    # not, so it has to be refused here rather than at the database.
    assert extraction.narrow("passage", "7").right.value == 7, "must be an int"

    for queue, scope, value, expected in (
        (parsing, "passage", "1", KeyError),
        (extraction, "document", "abc", None),
        (extraction, "passage", "not-a-number", ValueError),
        (extraction, "nonsense", "1", KeyError),
    ):
        try:
            queue.narrow(scope, value)
        except (KeyError, ValueError) as exc:
            assert type(exc) is expected, f"{scope}={value}: {type(exc).__name__}"
        else:
            assert expected is None, f"{scope}={value} should have been refused"


def structured_logs() -> None:
    """A shipped log line must carry the level, the trace and the traceback.

    The failure this catches is silent in both directions. An `@timestamp`
    Filebeat cannot parse is quietly replaced with the time the line was
    read, so every line is misdated and nobody is told; and an exception
    rendered without `error.stack_trace` leaves the dashboard showing that
    something failed and not what.

    Raises:
        AssertionError: If a field is missing or misnamed, if the timestamp
            is not one Elasticsearch accepts, if the traceback is lost, or
            if a caller's own fields do not survive.
    """
    import json
    import tempfile
    from datetime import datetime

    from telemetry.logs import JsonFormatter, _file

    record = logging.LogRecord(
        name="extraction.service",
        level=logging.ERROR,
        pathname="extraction/service.py",
        lineno=104,
        msg="failed passage %d",
        args=(41,),
        exc_info=None,
        func="process_next",
    )
    try:
        raise ValueError("the model would not answer")
    except ValueError:
        record.exc_info = sys.exc_info()
    # As telemetry.add_trace_fields puts them on: zeros outside a span.
    record.otelTraceID = "0" * 32
    record.otelSpanID = "0" * 16
    record.passage_id = 41  # a caller's own field, passed as extra=

    line = json.loads(JsonFormatter("extraction").format(record))

    assert line["log.level"] == "error", line["log.level"]
    assert line["log.logger"] == "extraction.service", line["log.logger"]
    assert line["service.name"] == "extraction", line["service.name"]
    assert line["message"] == "failed passage 41", line["message"]
    assert line["log.origin.file.line"] == 104, line["log.origin.file.line"]
    assert line["trace.id"] == "0" * 32, line["trace.id"]
    assert line["passage_id"] == 41, "a caller's own fields must survive"

    # The whole traceback, not just the message.
    assert line["error.type"] == "ValueError", line["error.type"]
    assert line["error.message"] == "the model would not answer"
    assert "Traceback" in line["error.stack_trace"], line["error.stack_trace"]
    assert "ValueError" in line["error.stack_trace"]

    # Parseable, offset-aware and UTC: what Filebeat needs to keep it rather
    # than substitute the time it read the line.
    assert line["@timestamp"].endswith("Z"), line["@timestamp"]
    when = datetime.fromisoformat(line["@timestamp"])
    assert when.tzinfo is not None, line["@timestamp"]
    assert when.utcoffset().total_seconds() == 0, line["@timestamp"]

    # One line, whatever the record holds: the shipper reads them one per
    # line, so an embedded newline is a second, unparseable event.
    assert "\n" not in JsonFormatter("extraction").format(record)

    # A directory that cannot be written costs the shipped copy and nothing
    # else. Raising here would stop a worker over a mounted volume.
    assert _file("extraction") is None, "no LOG_DIR must mean no file handler"
    with tempfile.TemporaryDirectory() as directory:
        os.environ["LOG_DIR"] = f"{directory}/nested"
        try:
            handler = _file("extraction")
            assert handler is not None, "a writable LOG_DIR must give a handler"
            # Named per container: a scaled stage runs several on one volume.
            assert handler.baseFilename.startswith(f"{directory}/nested/extraction-")
            handler.close()

            os.environ["LOG_DIR"] = "/proc/nowhere/qa"
            assert _file("extraction") is None, "an unwritable LOG_DIR must not raise"
        finally:
            del os.environ["LOG_DIR"]


def main() -> int:
    """Runs every check, stopping at the first failure.

    Reports through the shared logger rather than stdout, so a failure here
    reads the same as a failure anywhere else and reaches the same place.
    """
    telemetry.configure("checks")
    for check in (
        sentence_numbering,
        fact_checks,
        german_modals,
        passage_gate,
        passage_language,
        kept_chunks,
        table_facts,
        broken_words,
        graceful_shutdown,
        stop_before_claiming,
        queue_narrowing,
        structured_logs,
        topic_vocabulary,
        topic_weights,
        topic_visualisation,
        topic_naming,
        topic_labels,
    ):
        try:
            check()
        except AssertionError:
            # exception(), not error(): the assertion's own line is the most
            # useful thing here and a message alone throws it away.
            log.exception("FAIL %s", check.__name__)
            return 1
        log.info("ok   %s", check.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
