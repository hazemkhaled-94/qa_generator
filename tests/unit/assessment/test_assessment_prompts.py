"""What the judge records about itself, held against what it asks.

A template the catalogue leaves out is a verdict nothing can be read back
to. A template it records under a name another one already has is worse:
the table is unique on (service, version, name), so the second write goes
over the first and the record says the judge asks one thing where it asks
three.
"""

from __future__ import annotations

from assessment.prompts import SERVICE, catalogue
from assessment.templates import PROMPT_VERSION, TEMPLATES


def test_every_template_is_recorded_under_a_name_of_its_own() -> None:
    """All of them, and `relevance` three times under three names."""
    every = catalogue()

    assert len(every) == sum(len(group) for group in TEMPLATES.values())
    assert len({one.name for one in every}) == len(every)
    assert "fact: relevance" in {one.name for one in every}


def test_a_recorded_prompt_carries_both_halves_and_its_version() -> None:
    """The system prompt, the user template and the shape the answer takes.

    The user half is the one a row used not to carry, and it is the half
    that says what was substituted into the call.
    """
    for one in catalogue():
        assert one.version == PROMPT_VERSION, one.name
        assert one.text.strip(), one.name
        assert "{{" in one.user_text, one.name
        assert one.response_schema, one.name


def test_the_service_is_the_stage() -> None:
    """`stored("assessment")` is what the api and the publisher ask for."""
    assert SERVICE == "assessment"
