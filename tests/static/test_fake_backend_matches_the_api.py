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

from extraction.models import FactQuality, StoredFact
from question_generation.models import QuestionQuality, StoredQuestion

#: Each fake response beside the dataclass it stands in for, and how to
#: reach the dict: a listing wraps its rows, a report is the dict itself.
PINNED: dict[str, tuple[str, type, str | None]] = {
    "question_quality": ("QUESTION_QUALITY", QuestionQuality, None),
    "fact_quality": ("FACT_QUALITY", FactQuality, None),
    "question": ("QUESTION", StoredQuestion, None),
    "fact": ("FACT", StoredFact, None),
}


def fields_of(shape: type) -> set[str]:
    """Every field the API serialises for one response."""
    return {one.name for one in dataclasses.fields(shape)}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_fake_response_carries_what_the_api_returns(name: str) -> None:
    """Neither short of the real thing nor carrying what it does not."""
    from tests.frontend import pages

    attribute, shape, _ = PINNED[name]
    faked = set(getattr(pages, attribute))
    real = fields_of(shape)

    assert faked == real, (
        f"{attribute} and {shape.__name__} disagree.\n"
        f"  the API returns and the fake does not: {sorted(real - faked)}\n"
        f"  the fake invents: {sorted(faked - real)}"
    )
