"""The configuration panel, on the page that runs the service it configures.

Every control is derived from what the API says about the setting, so what
these cover is the deriving.

The backend is stubbed, refusals included, which is the half no integration
test reaches.
"""

from __future__ import annotations

import pytest
import requests
from pages import SETTINGS, Answers, settings

pytestmark = pytest.mark.frontend

#: Every page and the service its panel configures. One page configures one
#: service, and this is that mapping as the views declare it.
PAGES = [
    ("upload", "ingestion"),
    ("documents", "parsing"),
    ("passages", "chunking"),
    ("facts", "extraction"),
    ("topics", "topics"),
    ("questions", "questions"),
    ("health", "platform"),
]


@pytest.mark.parametrize(("name", "service"), PAGES)
def test_every_page_asks_for_the_service_it_runs(open_view, name, service) -> None:
    """One page configures one service, and never another page's."""
    configured = Answers(**settings())
    view = open_view(name, configured=configured)

    assert view.raised == [], view.raised
    asked = [args[0] for _, args, _ in configured.calls("settings")]
    assert asked == [service], asked


def test_a_number_is_drawn_as_a_number(open_view) -> None:
    """Typed from what the API said it is, not from the page's own list."""
    view = open_view("topics")

    drawn = view.numbers()
    assert "A_COUNT" in drawn
    assert drawn["A_COUNT"].value == 5


def test_a_share_is_drawn_with_the_bounds_it_has(open_view) -> None:
    """A share above one is a gate nothing passes, so the control refuses it."""
    view = open_view("topics")

    assert view.numbers()["A_SHARE"].value == 0.5


def test_a_flag_is_drawn_as_a_checkbox(open_view) -> None:
    """Read the way the backend's own reader reads it."""
    view = open_view("topics")

    assert view.checkboxes()["A_FLAG"].value is True


def test_a_closed_set_is_drawn_as_a_picker(open_view) -> None:
    """A list of two is a picker, not a box somebody can mistype into."""
    view = open_view("topics")

    drawn = view.pickers()
    assert drawn["A_MODE"].value == "fast"
    assert list(drawn["A_MODE"].options) == ["fast", "accurate"]


def test_a_list_with_a_closed_set_is_drawn_as_a_multiselect(open_view) -> None:
    """Each entry is checked against the list, so each entry is picked."""
    view = open_view("topics")

    assert view.multiselects()["A_LIST"].value == ["one", "two"]


def test_a_setting_the_deployment_owns_is_drawn_and_disabled(open_view) -> None:
    """Visible without opening a shell, and not writable from here."""
    view = open_view("topics")

    assert view.texts()["A_POOL"].disabled is True


def test_a_setting_whose_absence_means_something_is_drawn_empty(open_view) -> None:
    """An empty box is how one is turned off, which a number box cannot say."""
    view = open_view("topics")

    assert view.texts()["A_MODEL"].value == ""


def test_a_changed_setting_is_marked(open_view) -> None:
    """A reader should see at a glance that the file no longer decides it."""
    changed = [{**one, "stored": True} for one in SETTINGS["settings"]]
    view = open_view("topics", configured=Answers(**settings(settings=changed)))

    assert "A_COUNT ·" in view.numbers()


def test_saving_sends_the_version_the_panel_was_drawn_from(open_view) -> None:
    """So a save that would land on somebody else's change is refused."""
    configured = Answers(**settings())
    view = open_view("topics", configured=configured)

    view.save_configuration()

    sent = configured.calls("change_settings")
    assert sent, "the save button sent nothing"
    assert sent[0][1][2] == SETTINGS["version"]


def test_saving_sends_every_setting_it_may_write(open_view) -> None:
    """And not the one the deployment owns, which it must not try to write."""
    configured = Answers(**settings())
    view = open_view("topics", configured=configured)

    view.save_configuration()

    written = configured.calls("change_settings")[0][1][1]
    assert "A_COUNT" in written
    assert "A_POOL" not in written


def test_a_refused_change_is_shown_in_the_words_the_api_refused_it(
    open_view,
) -> None:
    """The one thing a person can act on, and the reason the API writes it.

    An HTTPError stringifies as its status line, so a page that showed the
    exception would show `400 Client Error` and nothing about which setting
    or what was wrong with it.
    """
    refused = requests.exceptions.HTTPError("400 Client Error")
    refused.response = _refusal(
        "would_not_load", "topics would not start with that: TOPIC_PASSES=many"
    )
    configured = Answers(settings=lambda service: SETTINGS, change_settings=refused)
    view = open_view("topics", configured=configured)

    after = view.save_configuration()

    assert "TOPIC_PASSES" in after.errors()[0]


def test_a_change_that_stales_something_says_so_and_it_stays(open_view) -> None:
    """A toast disappears; what this says to do costs a corpus-sized run."""
    configured = Answers(
        settings=lambda service: SETTINGS,
        change_settings=lambda service, values, version: {
            "service": service,
            "version": "changed",
            "changed": ["A_COUNT"],
            "cleared": [],
            "stale": ["chunking", "questions"],
            "detail": "1 setting(s) changed. What chunking, questions already "
            "produced was made under the old values; rerun that stage.",
        },
    )
    view = open_view("topics", configured=configured)

    after = view.save_configuration()

    assert "rerun" in after.warnings()[0]


def test_an_unreachable_backend_leaves_the_rest_of_the_page_working(
    open_view,
) -> None:
    """The panel is one section of a page that has a job of its own.

    A page that would not draw because its configuration could not be read
    would be a page where nothing could be looked at either.
    """
    view = open_view(
        "topics",
        configured=Answers(settings=requests.exceptions.ConnectionError("refused")),
    )

    assert view.raised == [], view.raised
    assert "Configuration unavailable" in view.text()


def _refusal(code: str, detail: str):
    """A response carrying the API's refusal body."""

    class Response:
        """Just enough of a response for the panel to read the reason."""

        @staticmethod
        def json() -> dict:
            """The body every deliberate refusal carries."""
            return {"code": code, "detail": detail}

    return Response()
