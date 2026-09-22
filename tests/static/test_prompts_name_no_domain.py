"""Every prompt is about nothing in particular, and stays that way.

The pipeline's one substantive claim about itself is that it is not bound to
a subject: the parser, the chunker and the topic model work over whatever the
documents say, and the thirteen question types are FORMS - a `comparison` is
askable of a contract, a manual or a report equally. Nothing in the code
decides what a corpus is about.

A prompt is where that claim is lost, and it is lost the same way every time.
Somebody debugs against the corpus at hand, finds a real question that the
gate got wrong, and pastes it in as a worked example - which is exactly what
extraction's own prompt warns about: an example drawn from the corpus at hand
teaches the model to expect it. It happened here. Four of `phrasing.py`'s
examples named the syllabi this deployment was pointed at, down to the
standard numbers, and `prompt_version` 2 recast them.

So the prompts are read as syntax and checked against the vocabulary of the
corpora this project has been aimed at. That list is a REGRESSION GUARD and
not a definition of what a domain is - there is no such list. Point the
pipeline at contracts and add the words a contract is full of; the test earns
its keep by failing when the next person debugging a gate reaches for the
nearest real example.

Read without importing, so the test needs no NLP_MODELS and no model, and a
failure names the prompt and the word.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every module whose strings reach a model. `types.py` carries the thirteen
#: type directives and the shared rule block, `generation.py` the
#: perturbation and the follow-up, `verifier.py` the round trip, and
#: `phrasing.py` the two judgements a rule could not settle. Extraction's are
#: here for the same reason: its worked example is the one the README holds
#: up as deliberately about nothing.
PROMPTS = (
    "backend/question_generation/types.py",
    "backend/question_generation/generation.py",
    "backend/question_generation/verifier.py",
    "backend/question_generation/phrasing.py",
    "backend/extraction/extractors/llm.py",
    "backend/extraction/extractors/bridge.py",
    "backend/extraction/extractors/digest.py",
    "backend/topic_modelling/labels.py",
)

#: The shortest string worth reading as a prompt. Below it a literal is a
#: field description, a log line or a key, and a subject word in one of those
#: reaches no model.
PROMPT_CHARS = 120

#: What the corpora this deployment has been aimed at are about. Lower-case,
#: matched as a substring, and deliberately short: these are the words a
#: person reaching for a real example would bring with them.
#:
#: Two corpora so far - software-testing syllabi and a regulatory set - and
#: the German compounds are why a substring match is right here. `Lehrplan`
#: is the leak whether it arrives as `Lehrplaninhalt` or `Lehrplänen`.
SUBJECTS = (
    # The software-testing syllabi
    "lehrplan",
    "lehrplän",
    "istqb",
    "iso/iec",
    "ieee",
    "ctfl",
    "ctal",
    "testteam",
    "teststufe",
    "testfall",
    "fehlerzustand",
    "überdeckung",
    "skripterstellung",
    "softwaretest",
    "black-box",
    "white-box",
    "wcag",
    # The regulatory set
    "bafin",
    "kryptowert",
    "anlassprüfung",
    "arbeitslosenquote",
    "liquiditätsmanagement",
    "bankensektor",
)


def _prompts(path: Path) -> list[tuple[int, str]]:
    """Every string literal in one module long enough to be a prompt."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value) >= PROMPT_CHARS
    ]


@pytest.mark.parametrize("module", PROMPTS)
def test_no_prompt_names_what_a_corpus_is_about(module: str) -> None:
    """A worked example teaches the model what to expect. Keep it neutral."""
    path = ROOT / module
    assert path.exists(), f"{module} is named here and does not exist"

    guilty = [
        f"{module}:{line} names {subject!r}"
        for line, text in _prompts(path)
        for subject in SUBJECTS
        if subject in text.casefold()
    ]

    assert not guilty, (
        "a prompt names what a corpus is about:\n"
        + "\n".join(guilty)
        + "\n\nA worked example drawn from the corpus at hand teaches the "
        "model to expect it, which is the one thing this pipeline claims not "
        "to do. Recast it into the neutral domain the other examples use - a "
        "support request, a shift plan, a site manager, an edition - and bump "
        "that module's PROMPT_VERSION."
    )


def test_the_guard_reads_the_prompts_it_claims_to() -> None:
    """A path that stopped holding prompts would pass by reading nothing.

    The list above is maintained by hand, so this is what says it is still
    pointed at the strings that reach a model.
    """
    empty = [module for module in PROMPTS if not _prompts(ROOT / module)]

    assert not empty, (
        f"these are named as prompt modules and hold no string over "
        f"{PROMPT_CHARS} characters: {', '.join(empty)}"
    )


#: The lemma lists a new language has to be added to, and the README section
#: that is supposed to name all of them.
LEMMA_LISTS = {
    "backend/question_generation/gates.py": (
        "_DIVISIONS",
        "_DOCUMENTS",
        "_ATTRIBUTIONS",
        "_AGENTS",
    ),
    "backend/nlp/analysis.py": ("_ANAPHORIC",),
}

ONBOARDING = ROOT / "backend/nlp/README.md"


def test_every_lemma_list_is_named_in_the_onboarding_guide() -> None:
    """A list nobody is told about is a gate that silently never fires.

    Most readings here are Universal Dependencies features and need no edit
    for a new language. The handful that are WORDS do, and a language
    missing from one of them gets no error and no log - just a judgement
    nothing makes. So the guide has to name all of them, and this is what
    says it still does.
    """
    guide = ONBOARDING.read_text(encoding="utf-8")
    missing = [
        f"{name} in {module}"
        for module, names in LEMMA_LISTS.items()
        for name in names
        if name not in guide
    ]

    assert not missing, (
        "these lemma lists are not named in "
        "backend/nlp/README.md#adding-a-language:\n  " + "\n  ".join(missing)
    )


def test_the_guide_names_no_list_that_has_been_deleted() -> None:
    """The other direction: a guide naming a list nothing holds is stale."""
    held = {
        name
        for module, names in LEMMA_LISTS.items()
        for name in names
        if name in (ROOT / module).read_text(encoding="utf-8")
    }
    named = {name for names in LEMMA_LISTS.values() for name in names}

    assert named == held, f"named but no longer defined: {sorted(named - held)}"
