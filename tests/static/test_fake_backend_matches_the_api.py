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

import pytest

from api.routes.assessment import JudgePlan
from api.routes.questions import GenerationPlan
from api.routes.settings import ServiceSettings, SettingState
from assessment.models import AssessmentQuality, StoredAssessment
from extraction.models import FactQuality, StoredFact
from ingestion.models import StoredDocument
from preprocessing.chunking.models import PassageDetail, StoredPassage
from question_generation.models import (
    QuestionQuality,
    QuestionSource,
    StoredQuestion,
)
from topic_modelling.models import LanguageFit, StoredTopic, TopicFit

#: Each fake response beside the dataclass it stands in for.
#:
#: Six of these when the rule was written, for the three quality reports and
#: the three rows. That left the other ten unguarded, including every shape
#: on the two pages nothing else covers - a document, a passage, a topic and
#: a fit - so the drift this exists to catch could still happen anywhere but
#: in the six places somebody had already been bitten.
PINNED: dict[str, tuple[str, type]] = {
    # The rows a listing page draws.
    "document": ("DOCUMENT", StoredDocument),
    "passage": ("PASSAGE", StoredPassage),
    "passage_detail": ("PASSAGE_DETAIL", PassageDetail),
    "fact": ("FACT", StoredFact),
    "topic": ("TOPIC", StoredTopic),
    "question": ("QUESTION", StoredQuestion),
    "assessment": ("ASSESSMENT", StoredAssessment),
    # What a row was drawn from, which the detail panes read.
    "question_source": ("QUESTION_SOURCE", QuestionSource),
    # The reports behind each page's Analysis fold.
    "fact_quality": ("FACT_QUALITY", FactQuality),
    "question_quality": ("QUESTION_QUALITY", QuestionQuality),
    "assessment_quality": ("ASSESSMENT_QUALITY", AssessmentQuality),
    # The state of the topic model, per language and overall.
    "language_fit": ("LANGUAGE_FIT", LanguageFit),
    "topic_fit": ("FIT", TopicFit),
    # What each phase is configured to do.
    "question_plan": ("PLAN", GenerationPlan),
    "assessment_plan": ("ASSESSMENT_PLAN", JudgePlan),
    # The configuration panel, which every page draws.
    "settings": ("SETTINGS", ServiceSettings),
}


def fields_of(shape: type) -> set[str]:
    """Every field the API serialises for one response."""
    return {one.name for one in dataclasses.fields(shape)}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_fake_response_carries_what_the_api_returns(name: str) -> None:
    """Neither short of the real thing nor carrying what it does not."""
    from tests.frontend import pages

    attribute, shape = PINNED[name]
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
    real = fields_of(SettingState)

    assert faked == real, (
        f"pages.setting and SettingState disagree.\n"
        f"  the API returns and the fake does not: {sorted(real - faked)}\n"
        f"  the fake invents: {sorted(faked - real)}"
    )
