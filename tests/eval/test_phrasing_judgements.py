"""How well whatever answers the three phrasing judgements answers them.

`names_its_source`, `self_contained` and `subject` are judgements about a
question alone - no passage decides any of them. They ride along in the
verifier's reading call today, and until this module existed nothing measured
them: the only ground truth was the model that answered, so a change to the
prompt or to the model moved the numbers with nothing to say which way.

That is the point of this file, and it is deliberately here before anything
is swapped. A judgement with no ground truth cannot be moved off a model
safely, because "the small model disagrees with the large one" and "the small
model is wrong" are the same observation until somebody has labelled the
cases.

Scored one judgement at a time. A single accuracy over a call answering three
things hides the one that got worse, and these three fail differently:
`names_its_source` confuses a source with a party, `self_contained` confuses a
pointing word with a reference, and `subject` is a span two correct readers
disagree about the edges of.

Printed rather than asserted, like every other eval module, with one
exception: a judgement no better than answering the same way every time is a
judgement that is not working, and that is a wiring failure rather than a
matter of taste.

    make test-eval

Skipped without a model, which is every CI run.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import pytest

from evaluation.cases import PHRASING

pytestmark = [pytest.mark.eval, pytest.mark.nlp]


@dataclass
class Score:
    """How one judgement did over the whole set."""

    name: str
    right: int = 0
    total: int = 0
    #: Cases it got wrong, as (case name, expected, answered).
    missed: list[tuple[str, object, object]] | None = None

    def record(self, case: str, expected, answered) -> None:
        """Counts one case."""
        self.total += 1
        if expected == answered:
            self.right += 1
        else:
            self.missed = (self.missed or []) + [(case, expected, answered)]

    @property
    def accuracy(self) -> float:
        """The share it got right."""
        return self.right / self.total if self.total else 0.0

    def report(self) -> str:
        """One line, and every case it got wrong under it."""
        lines = [f"  {self.name:20} {self.right:2}/{self.total:2}  {self.accuracy:.1%}"]
        for case, expected, answered in self.missed or ():
            lines.append(f"      {case}: expected {expected!r}, got {answered!r}")
        return "\n".join(lines)


def majority(judgement: str) -> float:
    """What answering every case the same way would score.

    The floor a judgement has to clear to be doing anything. Four of nineteen
    cases are not self-contained, so a judge that says True every time scores
    0.79 on that one and has measured nothing.
    """
    counts = Counter(case[judgement] for case in PHRASING)
    return max(counts.values()) / len(PHRASING)


@pytest.fixture(scope="module")
def verifier():
    """The real phrasing judge against the served model, or a skip.

    `QUESTIONS_PHRASING_MODEL` when one is named, the verifier's otherwise,
    because that is the fallback the factory applies and this has to measure
    what a run would actually ask.
    """
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is unset; no model to measure")

    from llm.client import Client, ModelUnavailable
    from llm.config import Settings
    from question_generation.config import Settings as QuestionSettings
    from question_generation.phrasing import PhrasingJudge

    settings = Settings.load()
    questions = QuestionSettings.load()
    built = PhrasingJudge(
        Client(
            settings.overridden(questions.phrasing_model or questions.verifier_model)
        )
    )
    try:
        if built.names_its_source("Is this on?") is None:
            raise ModelUnavailable("it abstained on the warm-up")
    except ModelUnavailable as exc:
        pytest.skip(f"the phrasing judge is not answering: {exc}")
    return built


@pytest.fixture(scope="module")
def scored(verifier):
    """Every case put to the pipeline as it now answers these.

    Not to one call any more. `subject` is read off the parse,
    `names_its_source` is a rule wherever a rule settles it and the model
    only for the residue, and `self_contained` is asked only where the parse
    found a pointing word. Scoring what the pipeline does is the point: a
    number for a call nothing makes measures nothing.
    """
    from nlp.analysis import pointing
    from question_generation.gates import cites_source
    from question_generation.gates import subject as read_subject

    source = Score("names_its_source")
    contained = Score("self_contained")
    subject = Score("subject")

    for case in PHRASING:
        question, language = case["question"], case["language"]

        ruled = cites_source(question, language)
        source.record(
            case["name"],
            case["names_its_source"],
            ruled if ruled is not None else verifier.names_its_source(question),
        )

        # Nothing to point at means nothing to rule on, and the gate reads
        # that as self-contained rather than asking.
        pointers = pointing(question, language)
        contained.record(
            case["name"],
            case["self_contained"],
            True if not pointers else bool(verifier.self_contained(question, pointers)),
        )

        # Containment, not equality: `holds` is a word the copied span must
        # carry, and "" is a question that names nothing.
        wanted = case["holds"]
        got = read_subject(question, language)
        subject.record(
            case["name"],
            bool(wanted),
            bool(got) and wanted.casefold() in got.casefold(),
        )
    return source, contained, subject


@pytest.mark.parametrize("judgement", ["names_its_source", "self_contained"])
def test_a_judgement_beats_answering_the_same_way_every_time(scored, judgement) -> None:
    """A judge that ignores its input scores the majority class."""
    found = next(one for one in scored if one.name == judgement)
    floor = majority(judgement)

    print(f"\n{found.report()}\n      (answering the same way every time: {floor:.1%})")

    assert found.accuracy > floor, (
        f"{judgement} scored {found.accuracy:.1%} against a floor of {floor:.1%}: "
        f"it is not reading the question"
    )


def test_every_judgement_is_scored_on_its_own(scored) -> None:
    """The table a swap is compared against. Printed, never a gate."""
    print("\nphrasing judgements, one line each:")
    for one in scored:
        print(one.report())

    assert all(one.total == len(PHRASING) for one in scored), (
        "a judgement was not put every case"
    )


def test_the_german_cases_are_judged_as_well_as_the_english(scored) -> None:
    """A judge that reads German worse is the risk a swap carries.

    Printed rather than asserted: twelve cases against seven is not a
    difference anybody should gate on. It is here because the corpus is
    German and an English-first judge would pass every test above - which
    is not hypothetical. Asked inside the verifier's reading call these
    judgements scored 7/7 in English and 6/12 in German, and that gap is
    why they are asked on their own now.

    Read off `scored` rather than asked again, so the split is over the same
    answers the table above reports.
    """
    wrong = {
        (one.name, case)
        for one in scored
        if one.name != "subject"
        for case, _, _ in one.missed or ()
    }
    by_language: dict[str, list[bool]] = {"de": [], "en": []}
    for case in PHRASING:
        by_language[case["language"]].append(
            ("names_its_source", case["name"]) not in wrong
            and ("self_contained", case["name"]) not in wrong
        )

    for language, results in by_language.items():
        share = sum(results) / len(results) if results else 0.0
        print(
            f"\n{language}: both judgements right on {sum(results)}/{len(results)} = {share:.1%}"
        )

    assert by_language["de"], "no German case was scored"
