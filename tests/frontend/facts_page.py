"""The page object over the Facts view.

It holds the selectors - which element, which key, which label - so a test
says what a reader does and what they see, and a change to the layout is one
edit here rather than one per test.
"""

from __future__ import annotations

from typing import Any, ClassVar

#: One atomic fact, as GET /facts returns it.
FACT = {
    "id": 1,
    "statement": "A standard request is answered within 48 hours.",
    "evidence_text": "Standard requests are answered within 48 hours on working days.",
    "kind": "atomic",
    "extraction_method": "llm",
    "validated": True,
    "rejection_code": None,
    "validation_error": None,
    "statement_predicates": 1,
    "evidence_predicates": 2,
    "units_added": [],
    "unresolved_references": [],
    "passages": [
        {
            "passage_id": 11,
            "doc_sha256": "a" * 64,
            "ordinal": 3,
            "page_from": 2,
            "position": 0,
        }
    ],
}


def resting(passage_id: int, ordinal: int, position: int, **changed: Any) -> dict:
    """One further passage a fact rests on, as GET /facts returns it."""
    return {
        "passage_id": passage_id,
        "doc_sha256": changed.pop("doc_sha256", "b" * 64),
        "ordinal": ordinal,
        "page_from": changed.pop("page_from", None),
        "position": position,
        **changed,
    }


#: What GET /facts/quality returns for one clean atomic corpus.
QUALITY = {
    "total": 1,
    "validated": 1,
    "mean_statement_chars": 46.0,
    "mean_evidence_chars": 63.0,
    "mean_statement_predicates": 1.0,
    "mean_evidence_predicates": 2.0,
    "facts_per_passage": 4.0,
    "rejected": {},
    "kinds": {"atomic": 1},
}


def fact(**changed: Any) -> dict:
    """One fact, with whatever a test needs changed."""
    return {**FACT, **changed}


def quality(**changed: Any) -> dict:
    """The quality report, with whatever a test needs changed."""
    return {**QUALITY, **changed}


def page_of(*facts: dict) -> dict:
    """One page of facts, and the total behind it."""
    return {"total": len(facts), "facts": list(facts)}


class FactsPage:
    """The Facts view, as a reader sees and works it."""

    #: Every backend call the page makes, and a default answer for each.
    ANSWERS: ClassVar[dict[str, Any]] = {
        "document_names": [{"sha256": "a" * 64, "filename": "report.pdf"}],
        "facts": page_of(FACT),
        "fact_quality": QUALITY,
        "stage_status": {"stage": "extraction", "working": False, "rows": {}},
        "stage_action": {"detail": "1 row(s) queued.", "rows": 1},
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
        """Every dataframe's contents in full, as one string."""
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

    def helps(self) -> str:
        """Every tooltip the page carries, as one string."""
        parts = [one.help or "" for one in self.app.metric]
        parts += [one.help or "" for one in self.app.get("radio")]
        parts += [one.help or "" for one in self.app.caption]
        return " ".join(parts)

    def readings(self) -> list[str]:
        """The kind filter's options, as a reader sees them."""
        return list(self.app.radio("facts-kind").options)

    # ── What a reader does ───────────────────────────────────────────────

    def choose_reading(self, label: str) -> FactsPage:
        """Picks which reading of the corpus to look at."""
        self.app.radio("facts-kind").set_value(label)
        return FactsPage(self.app.run())

    def inspect(self, at: int) -> FactsPage:
        """Picks one fact out of the table to see in full."""
        self.app.selectbox[-1].select_index(at)
        return FactsPage(self.app.run())
