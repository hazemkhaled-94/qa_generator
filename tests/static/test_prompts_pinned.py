"""Every prompt this stage writes with, pinned to its version.

"Two prompts are two datasets" is what PROMPT_VERSION is for, and nothing
checked it: a rule could be added to `types.py`, every question written
after it would differ from every question written before, and both would
carry version 8.

So each prompt is pinned by digest. Changing one fails this file, and the
only way to make it pass is to bump PROMPT_VERSION and write the new digest
down - which is the decision the version exists to record.

The digest and not the text, because the prompts are eight hundred lines
and a copy of them here would be a second place for them to drift. Which
one moved is what the per-type entries say; `test_prompts_name_no_domain`
is what reads their content.
"""

from __future__ import annotations

import hashlib

import pytest

from question_generation.generation import _FOLLOW, _PERTURB
from question_generation.types import PROMPT_VERSION, SPECS

#: The version every digest below was taken under.
VERSION = "8"

#: Each type's whole system prompt: the shared rules, the answer form, the
#: directive and its worked examples.
TYPES = {
    "aggregation": "e965582d60522c93",
    "application": "b444eecdbff9cf31",
    "comparison": "dd57b8cb9c8a659b",
    "condition": "ce344dd6b8daaeda",
    "consequence": "e4ed91488e70010e",
    "definition": "d4c14309f6b022d5",
    "entity": "86e2d5ad5310703f",
    "enumeration": "40b0d69e7b0d2472",
    "factoid": "b498acbbf5537e5e",
    "implication": "fd2f98bad0b0bb08",
    "procedure": "83cd9e3744d6aca0",
    "reason": "9d64a2bf5ce43efc",
    "temporal": "c681eaec3d0fb768",
}

#: The two prompts that are not a type's: the perturbation that writes the
#: unanswerable questions, and the one that writes the next turn.
OTHERS = {
    "perturb": "0b3d2a3bdc02bab1",
    "follow": "c3f02405d0cedd9f",
}


def digest(text: str) -> str:
    """The first sixteen hex characters of one prompt's SHA-256."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_the_version_pinned_here_is_the_one_the_code_declares() -> None:
    """Bumping one and not the other leaves the pinning meaningless."""
    assert PROMPT_VERSION == VERSION, (
        f"PROMPT_VERSION is {PROMPT_VERSION} and this file pins {VERSION}. "
        f"Update VERSION and the digests below in the same commit."
    )


@pytest.mark.parametrize("name", sorted(TYPES))
def test_a_type_writes_the_prompt_it_was_pinned_with(name: str) -> None:
    """A rule added to one type changes what that type's questions are."""
    assert digest(SPECS[name].system()) == TYPES[name], (
        f"the {name} prompt changed. Bump PROMPT_VERSION and write the new "
        f"digest here: questions written before and after are two datasets."
    )


@pytest.mark.parametrize(("name", "text"), [("perturb", _PERTURB), ("follow", _FOLLOW)])
def test_the_prompts_that_are_not_a_type_are_pinned_too(name: str, text: str) -> None:
    """The unanswerable share and the follow-ups are written by these."""
    assert digest(text) == OTHERS[name], (
        f"the {name} prompt changed. Bump PROMPT_VERSION and write the new digest here."
    )


def test_every_type_the_mix_can_name_is_pinned() -> None:
    """A type added with no digest would be a prompt nothing watches."""
    assert set(SPECS) == set(TYPES), (
        "a question type was added or removed without pinning its prompt"
    )


def test_the_shared_rules_reach_every_type() -> None:
    """One rule block, so two prompts cannot come to disagree.

    The pinning above would pass if `READS` were dropped from one type and
    that type's digest updated, which is the drift this is about.
    """
    for name, spec in SPECS.items():
        written = spec.system()
        assert "NEVER SAY WHERE THE ANSWER IS" in written, name
        assert "ONE question, ending in a question mark" in written, name
        assert "You write TWO answers" in written, name


def test_a_type_that_spans_is_told_to_span() -> None:
    """The row otherwise carries a spread the question never used."""
    for name, spec in SPECS.items():
        if spec.spans:
            assert "MORE THAN ONE PASSAGE" in spec.system(), name


def test_a_single_passage_type_is_told_to_span_only_when_the_sample_does() -> None:
    """A factoid drawn from two documents is a cross-document factoid."""
    factoid = SPECS["factoid"]

    assert "MORE THAN ONE PASSAGE" not in factoid.system()
    assert "MORE THAN ONE PASSAGE" in factoid.system(spans=True)
