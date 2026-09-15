"""Page objects over the Streamlit views.

One class per page. It holds the selectors - which element, which key, which
label - so a test says what a reader does and what they see, and a change to
the layout is one edit here rather than one per test.
"""

from __future__ import annotations

from typing import Any, ClassVar

#: One fitted topic, as GET /topics returns it.
TOPIC = {
    "id": 1,
    "language": "de",
    "topic_index": 0,
    "top_terms": ["lieferung", "versand", "transport"],
    "label": None,
    "labelled_by": None,
    "include_in_coverage": True,
    "passages": 40,
    "dominant_passages": 25,
    "mean_weight": 0.42,
    "documents": 3,
    "table_passages": 4,
    "validated_facts": 50,
}

#: One language's entry in GET /topics/fit.
LANGUAGE_FIT = {
    "language": "de",
    "topics": 2,
    "corpus_passages": 100,
    "corpus_vocabulary": 800,
    "passages_without_topics": 0,
    "fitted_at": "2026-09-14T10:00:00+00:00",
    "live_passages": 100,
    "memberships": 60,
}

#: A healthy model of two topics in one language.
FIT = {
    "status": "modelled",
    "error": None,
    "requested_at": "2026-09-14T09:00:00+00:00",
    "topics": 2,
    "languages": [LANGUAGE_FIT],
    "passages_without_language": 0,
}


def topic(**changed: Any) -> dict:
    """One topic, with whatever a test needs changed."""
    return {**TOPIC, **changed}


def language_fit(**changed: Any) -> dict:
    """One language's fit state, with whatever a test needs changed."""
    return {**LANGUAGE_FIT, **changed}


def fit(languages: list[dict] | None = None, **changed: Any) -> dict:
    """The model's state, with whatever a test needs changed."""
    return {
        **FIT,
        "languages": [LANGUAGE_FIT] if languages is None else languages,
        **changed,
    }


class TopicsPage:
    """The Topics view, as a reader sees and works it."""

    #: Every backend call the page makes, and a default answer for each.
    ANSWERS: ClassVar[dict[str, Any]] = {
        "topics": [TOPIC],
        "topic_fit": FIT,
        "topic_visualisation": None,
        "describe_topic": TOPIC,
        "delete_topics": {"topics": 2, "memberships": 60, "labels": 1},
        "stage_action": {"detail": "1 row(s) queued.", "rows": 1},
        "stage_status": {"stage": "topics", "working": False, "rows": {}},
    }

    def __init__(self, app) -> None:
        """Wraps a run of the view."""
        self.app = app

    @classmethod
    def answers(cls, **changed: Any) -> dict:
        """The scripted backend, with whatever a test needs changed."""
        return {**cls.ANSWERS, **changed}

    # ── What the page rendered ───────────────────────────────────────────

    @property
    def raised(self):
        """Whatever the page raised, which should be nothing."""
        return self.app.exception

    def text(self) -> str:
        """Everything the page wrote as prose, as one string."""
        parts = []
        for kind in (
            "markdown",
            "text",
            "info",
            "warning",
            "error",
            "success",
            "caption",
            "title",
            "subheader",
        ):
            parts += [one.value for one in getattr(self.app, kind)]
        return " ".join(str(one) for one in parts)

    def tables(self) -> str:
        """Every dataframe's contents in full, as one string.

        `to_string`, because a DataFrame's repr elides both columns and rows.
        """
        return " ".join(
            one.value.to_string() if hasattr(one.value, "to_string") else str(one.value)
            for one in self.app.dataframe
        )

    def metrics(self) -> dict[str, str]:
        """Every figure on the page, by its label."""
        return {one.label: one.value for one in self.app.metric}

    def explanations(self) -> dict[str, str]:
        """The tooltip behind every figure, by its label."""
        return {one.label: one.help or "" for one in self.app.metric}

    def buttons(self) -> dict[str, Any]:
        """Every control, by its label."""
        return {one.label: one for one in self.app.button}

    def button(self, label: str):
        """One control by label, or None if the page does not offer it."""
        return self.buttons().get(label)

    def embedded(self) -> list[Any]:
        """The figure, which `components.html` renders as an iframe."""
        return list(self.app.get("iframe"))

    def toasts(self) -> list[str]:
        """What the page reported as a transient message."""
        return [str(one.value) for one in self.app.get("toast")]

    # ── What a reader does ───────────────────────────────────────────────

    def press(self, label: str) -> TopicsPage:
        """Presses one control and re-runs the page."""
        found = self.button(label)
        assert found is not None, f"no {label!r} control: {sorted(self.buttons())}"
        return TopicsPage(found.click().run())

    def choose_language(self, language: str) -> TopicsPage:
        """Picks which language's figure to draw."""
        self.app.selectbox("topics-map-language").select(language)
        return TopicsPage(self.app.run())

    def type_label(self, topic_id: int, label: str) -> TopicsPage:
        """Types a name for one topic."""
        self.app.text_input(f"topics-label-{topic_id}").set_value(label)
        return TopicsPage(self.app.run())

    def clear_coverage(self, topic_id: int) -> TopicsPage:
        """Takes one topic out of coverage reporting."""
        self.app.checkbox(f"topics-coverage-{topic_id}").uncheck()
        return TopicsPage(self.app.run())
