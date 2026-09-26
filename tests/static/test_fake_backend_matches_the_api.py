"""The fake backend the frontend tests drive, against the real responses.

`tests/frontend/pages.py` answers the frontend with hand-written dicts. That
is what makes those tests fast and what makes them lie: a page reads
`quality["cognitive_level"]`, the fake is edited in the same commit to carry
it, and the suite agrees with itself while the real API returns no such key.

Which is what happened. The field was added to the API and to the page, the
fake was updated to match, 2,673 tests passed, and the page raised
`KeyError: 'cognitive_level'` against a running backend.

Nothing else covers this. tests/contract/test_openapi.py pins the SURFACE -
paths, verbs, status codes, parameter names - and deliberately not the
response bodies, so a field appearing in one place and not the other is
invisible to it.

So: every fake response is checked against the dataclass the API actually
serialises. A field the API gained and the fake did not is the failure this
exists for; a field the fake invented is the other direction of the same
drift.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

#: Each fake response beside the dataclass it stands in for.
#:
#: Six of these when the rule was written, for the three quality reports and
#: the three rows. That left the other ten unguarded, including every shape
#: on the two pages nothing else covers - a document, a passage, a topic and
#: a fit - so the drift this exists to catch could still happen anywhere but
#: in the six places somebody had already been bitten.
#:
#: The shapes are NAMED rather than imported, and that is not a style
#: choice. Three of them live in `api.routes.*`, which imports
#: `api.dependencies` - the composition root, which builds every repository
#: and caches an engine off DATABASE_URL as it is imported. pytest imports
#: every test module while collecting, deselected ones included, so an
#: import here bound the whole session's API to the placeholder DSN in
#: conftest before any fixture could point it at a container. 222 of the
#: integration tests failed on it and `test_the_api_is_wired_to_the_
#: container_and_not_to_a_developers_database` said exactly why.
PINNED: dict[str, tuple[str, str]] = {
    # The rows a listing page draws.
    "document": ("DOCUMENT", "ingestion.models:StoredDocument"),
    "passage": ("PASSAGE", "preprocessing.chunking.models:StoredPassage"),
    "passage_detail": ("PASSAGE_DETAIL", "preprocessing.chunking.models:PassageDetail"),
    "fact": ("FACT", "extraction.models:StoredFact"),
    "topic": ("TOPIC", "topic_modelling.models:StoredTopic"),
    "question": ("QUESTION", "question_generation.models:StoredQuestion"),
    "assessment": ("ASSESSMENT", "assessment.models:StoredAssessment"),
    # What a row was drawn from, which the detail panes read.
    "question_source": ("QUESTION_SOURCE", "question_generation.models:QuestionSource"),
    # The reports behind each page's Analysis fold.
    "fact_quality": ("FACT_QUALITY", "extraction.models:FactQuality"),
    "question_quality": (
        "QUESTION_QUALITY",
        "question_generation.models:QuestionQuality",
    ),
    "assessment_quality": ("ASSESSMENT_QUALITY", "assessment.models:AssessmentQuality"),
    # The state of the topic model, per language and overall.
    "language_fit": ("LANGUAGE_FIT", "topic_modelling.models:LanguageFit"),
    "topic_fit": ("FIT", "topic_modelling.models:TopicFit"),
    # What each phase is configured to do.
    "question_plan": ("PLAN", "api.routes.questions:GenerationPlan"),
    "assessment_plan": ("ASSESSMENT_PLAN", "api.routes.assessment:JudgePlan"),
    # The configuration panel, which every page draws.
    "settings": ("SETTINGS", "api.routes.settings:ServiceSettings"),
}


def shape_of(dotted: str) -> type:
    """The dataclass one name points at, imported when a test asks.

    Inside the test, never at module scope. See the note on PINNED.
    """
    module, _, attribute = dotted.partition(":")
    return getattr(importlib.import_module(module), attribute)


def fields_of(shape: type) -> set[str]:
    """Every field the API serialises for one response."""
    return {one.name for one in dataclasses.fields(shape)}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_fake_response_carries_what_the_api_returns(name: str) -> None:
    """Neither short of the real thing nor carrying what it does not."""
    from tests.frontend import pages

    attribute, dotted = PINNED[name]
    shape = shape_of(dotted)
    faked = set(getattr(pages, attribute))
    real = fields_of(shape)

    assert faked == real, (
        f"{attribute} and {shape.__name__} disagree.\n"
        f"  the API returns and the fake does not: {sorted(real - faked)}\n"
        f"  the fake invents: {sorted(faked - real)}"
    )


def test_a_faked_setting_carries_what_the_panel_is_drawn_from() -> None:
    """The one shape built by a function rather than held as a dict.

    Every control in the configuration panel is derived from these fields,
    so a field the API gained is a control the panel silently stops
    drawing and no page test can see the absence of.
    """
    from tests.frontend import pages

    faked = set(pages.setting("ANY"))
    real = fields_of(shape_of("api.routes.settings:SettingState"))

    assert faked == real, (
        f"pages.setting and SettingState disagree.\n"
        f"  the API returns and the fake does not: {sorted(real - faked)}\n"
        f"  the fake invents: {sorted(faked - real)}"
    )
