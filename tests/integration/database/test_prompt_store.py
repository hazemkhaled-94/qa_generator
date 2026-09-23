"""Recording a prompt, and reading a version back as the text it named.

Against a real Postgres, because what is under test is the upsert: which
write is a no-op, which rewrites a row, and which is a second row. The
whole point of the table is that a version resolves to exactly one text.
"""

from __future__ import annotations

import pytest

from stages.prompts import Composed, record, stored

pytestmark = pytest.mark.integration


@pytest.fixture
def written(engine, database) -> int:
    """Two prompts of one service, on the empty migrated database."""
    return record(
        "questions",
        [
            Composed("factoid", "8", "Write one question."),
            Composed("verifier: recover", "1", "Answer from the passages."),
        ],
    )


def test_a_version_resolves_to_the_text_it_named(written) -> None:
    """The lookup a question's prompt_version is for."""
    assert written == 2

    found = stored(service="questions", version="8", name="factoid")

    assert [one.text for one in found] == ["Write one question."]


def test_writing_the_same_prompts_again_changes_nothing(written) -> None:
    """Every start-up after the first, which is most of them."""
    again = record("questions", [Composed("factoid", "8", "Write one question.")])

    assert again == 0
    assert len(stored(service="questions")) == 2


def test_a_new_version_is_a_second_row_and_not_a_rewrite(written) -> None:
    """Two prompts are two datasets, so both have to stay readable.

    A question written under 8 must still resolve to 8 after 9 exists -
    that is the whole reason the row outlives the source.
    """
    record("questions", [Composed("factoid", "9", "Write one BETTER question.")])

    assert len(stored(service="questions", name="factoid")) == 2
    assert stored(version="8", name="factoid")[0].text == "Write one question."
    assert stored(version="9", name="factoid")[0].text == "Write one BETTER question."


def test_a_changed_prompt_under_one_version_is_stored_and_reported(
    written, caplog
) -> None:
    """Drift: the row follows what is actually sent, and says so.

    Stored rather than refused, because a worker that will not start over
    an edited prompt is worse than one that says the edit happened. The
    static pinning is what fails a pull request over it.
    """
    with caplog.at_level("WARNING"):
        moved = record("questions", [Composed("factoid", "8", "Write one QUESTION.")])

    assert moved == 1
    assert stored(version="8", name="factoid")[0].text == "Write one QUESTION."
    assert "PROMPT_VERSION is still 8" in caplog.text


def test_two_services_may_share_a_name_at_one_version(written) -> None:
    """Versions are per MODULE, so `1` means different things in two."""
    record("extraction", [Composed("factoid", "8", "Something else entirely.")])

    assert stored(service="questions", version="8", name="factoid")[0].text == (
        "Write one question."
    )
    assert stored(service="extraction", version="8", name="factoid")[0].text == (
        "Something else entirely."
    )


def test_recording_nothing_writes_nothing(engine, database) -> None:
    """A stage with no versioned prompt costs no query."""
    assert record("questions", []) == 0
    assert stored() == []
