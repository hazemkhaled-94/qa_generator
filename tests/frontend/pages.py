"""One page object over every view, and the answers each view needs.

Every page is now the same shape - overview, analysis, one service, search,
filters, the table, and the row a person picked - so one class covers all of
them. It holds the selectors, so a test says what a reader does and what
they see, and a change to the layout is one edit here.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

#: The digest every fixture document carries.
SHA = "a" * 64

#: A queue with something in every state, so a page's figures have something
#: to divide by and every control has a reason to be live.
STATUS: dict[str, Any] = {
    "stage": "any",
    "working": False,
    "rows": {"new": 2, "pending": 1, "in_progress": 1, "failed": 1},
}

#: A queue at rest, with nothing left to do.
IDLE: dict[str, Any] = {"stage": "any", "working": False, "rows": {}}


def status(**rows: int) -> dict:
    """One stage's queue, holding exactly what a test names."""
    return {"stage": "any", "working": False, "rows": rows}


DOCUMENT = {
    "sha256": SHA,
    "filename": "report.pdf",
    "first_seen": "2026-09-01T10:00:00",
    "page_count": 12,
    "title": "Risks in focus",
    "language": "en",
    "parse_status": "parsed",
    "parse_error": None,
    "chunk_status": "chunked",
    "chunk_error": None,
    "extracted_passages": 8,
    "total_passages": 10,
    "oversized": 0,
}

PASSAGE = {
    "id": 11,
    "doc_sha256": SHA,
    "ordinal": 3,
    "text": "Standard requests are answered within 48 hours on working days.",
    "page_from": 2,
    "page_to": 2,
    "section_path": "Service > Response times",
    "block_type": "paragraph",
    "language": "en",
    "doc_item_refs": ["#/texts/7"],
    "bbox": [{"page": 2, "l": 1, "t": 2, "r": 3, "b": 4}],
    "table_count": 0,
    "sentence_count": 1,
}

PASSAGE_DETAIL = {
    "passage": PASSAGE,
    "table_cells": [],
    "sentences": [{"i": 1, "start": 0, "end": 62, "predicates": 2}],
    "extract_status": "extracted",
    "extract_error": None,
}

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
            "doc_sha256": SHA,
            "ordinal": 3,
            "page_from": 2,
            "position": 0,
        }
    ],
    "confidence": 0.7527,
    "gate_scores": [
        {"gate": "duplicate", "value": 0.23, "threshold": 0.95, "margin": 0.7579}
    ],
}

FACT_QUALITY = {
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

FIT = {
    "status": "modelled",
    "error": None,
    "requested_at": "2026-09-14T09:00:00+00:00",
    "topics": 2,
    "languages": [LANGUAGE_FIT],
    "passages_without_language": 0,
}

QUESTION = {
    "id": 1,
    "question_text": "Within how long is a standard request answered?",
    "target_answer": "48 hours",
    "answer_explanation": (
        "A standard support request has to be answered within 48 hours. That "
        "is the ordinary service level, and the slower of the two the "
        "material sets: an urgent request is answered within 4 hours."
    ),
    "answerable": True,
    "difficulty": "easy",
    "passage_scope": "single_passage",
    "document_scope": "single_document",
    "topic_scope": "single_topic",
    "answer_chars": 8,
    "language": "en",
    "status": "accepted",
    "rejected_reason": None,
    "reviewed_verdict": None,
    "created_at": "2026-09-14T11:00:00",
    "facts": 1,
    # The document's title, which is what the listing resolves; the digest
    # is the fallback for a row carrying none.
    "documents": ["Service policy"],
    "topics": ["de #0"],
    "statements": [FACT["statement"]],
    "evidence": [FACT["evidence_text"]],
    "thread_position": 1,
    "follows_id": None,
    "question_type": "factoid",
    "cognitive_level": "recall",
    "answer_form": "value",
    "planned_difficulty": "easy",
    "prompt_version": "8",
    "trace_id": "0bdf4a87ef67fc94b435f23b2824b7e1",
    "span_id": "412b8f4d0b9073d3",
    "attempt": 1,
    "confidence": 0.24,
    "gate_scores": [
        {
            "gate": "near_duplicate",
            "value": 0.71,
            "threshold": 0.93,
            "margin": 0.2366,
        },
        {"gate": "recall", "value": 0.83, "threshold": 0.6, "margin": 0.575},
    ],
}

QUESTION_SOURCE = {
    "fact_id": 1,
    "statement": FACT["statement"],
    "evidence_text": FACT["evidence_text"],
    "validated": True,
    "passage_id": 11,
    "doc_sha256": SHA,
    "ordinal": 3,
}

QUESTION_QUALITY = {
    "total": 1,
    "accepted": 1,
    "draft": 0,
    "unanswerable": 0,
    "followups": 0,
    "topics_covered": 1,
    "topics_in_coverage": 1,
    "mean_question_chars": 47.0,
    "mean_answer_chars": 8.0,
    "rejected": {},
    "difficulty": {"easy": 1},
    "passage_scope": {"single_passage": 1},
    "document_scope": {"single_document": 1},
    "topic_scope": {"single_topic": 1},
    "question_type": {"factoid": 1},
    "cognitive_level": {"recall": 1},
    "answer_form": {"value": 1},
    "planned_difficulty": {"easy": 1},
    "planned_met": 1,
    "cognitive_level_checked": 1,
    # What the accepted questions reach, which the topic counts above
    # cannot say: a topic is covered by its first accepted question.
    "passages_total": 4,
    "passages_with_facts": 3,
    "passages_asked": 2,
    "facts_validated": 6,
    "facts_asked": 3,
}

PLAN = {
    "types": {"factoid": 2, "reason": 1},
    "difficulty": {"easy": 1},
    "followup_types": [],
    "per_topic": 5,
    "unanswerable_share": 0.1,
    "followup_share": 0.2,
    "max_followups": 2,
    "answer_chars": {"value": [2, 80]},
}


def changed(base: dict, **fields: Any) -> dict:
    """One fixture row, with whatever a test needs different."""
    return {**base, **fields}


def resting(passage_id: int, ordinal: int, position: int, **fields: Any) -> dict:
    """One further passage a fact rests on, as GET /facts returns it."""
    return {
        "passage_id": passage_id,
        "doc_sha256": fields.pop("doc_sha256", "b" * 64),
        "ordinal": ordinal,
        "page_from": fields.pop("page_from", None),
        "position": position,
        **fields,
    }


#: One artefact the judge answered about, as GET /assessment returns it.
#: Refused, and refused on a fact the checker KEPT, because that pair is
#: what the page exists to surface.
ASSESSMENT = {
    "id": 1,
    "kind": "fact",
    "artifact_id": 7,
    "status": "assessed",
    "approved": False,
    "judge_model": "ollama_chat/qwen3:14b",
    "prompt_version": "1",
    "run_id": "a-run",
    "trace_id": "a" * 32,
    "span_id": "b" * 16,
    "assessed_at": "2026-09-25T12:00:00Z",
    "error": None,
    "verdict": "accepted",
    "summary": "A reply is due in five days.",
    "metrics": [
        {
            "metric": "hallucination",
            "label": "hallucinated",
            "score": 1.0,
            "approved": False,
            "explanation": "the evidence gives no number of days",
        },
        {
            "metric": "relevance",
            "label": "relevant",
            "score": 1.0,
            "approved": True,
            "explanation": "the sentences are about reply times",
        },
    ],
}

#: What the judge made of the corpus, as GET /assessment/quality returns it.
ASSESSMENT_QUALITY = {
    "total": 3,
    "judged": 2,
    "approved": 1,
    "refused": 1,
    "disagreements": 1,
    "by_kind": {"fact": 1, "topic": 1, "question": 1},
    "approved_by_kind": {"fact": 0, "topic": 1, "question": 0},
    "metrics": [
        {
            "metric": "hallucination",
            "judged": 2,
            "approved": 1,
            "direction": "minimize",
        },
        {"metric": "relevance", "judged": 2, "approved": 2, "direction": "maximize"},
    ],
    "judge_models": ["ollama_chat/qwen3:14b"],
    "outstanding": 1,
}

#: What the evaluation phase is configured to do, as /assessment/plan says.
ASSESSMENT_PLAN = {
    "enabled": True,
    "judge_model": "ollama_chat/qwen3:14b",
    "kinds": ["fact", "topic", "question"],
    "sample": 200,
    "prompt_version": "1",
    "metrics": {
        "fact": ["hallucination", "relevance"],
        "topic": ["summarization", "relevance"],
        "question": ["hallucination", "qa_correctness", "relevance"],
    },
}

#: Every backend call the catalogue pages make, and a default answer for
#: each. One dictionary, because one client serves five pages and a page
#: that started calling something new should not need a new fixture.
CATALOG: dict[str, Any] = {
    "stage_status": lambda *args, **kwargs: STATUS,
    "stage_action": {"detail": "1 row(s) queued.", "rows": 1},
    "documents": {"total": 1, "documents": [DOCUMENT]},
    "document_names": [{"sha256": SHA, "filename": "report.pdf"}],
    "delete": {
        "sha256": SHA,
        "document": True,
        "passages": 10,
        "file": True,
        "parsed": True,
    },
    "passages": {"total": 1, "passages": [PASSAGE]},
    "passage": PASSAGE_DETAIL,
    "passage_types": ["paragraph", "table"],
    "facts": {"total": 1, "facts": [FACT]},
    "fact_quality": FACT_QUALITY,
    "topics": [TOPIC],
    "topic_fit": FIT,
    "topic_visualisation": None,
    "describe_topic": TOPIC,
    "delete_topics": {"topics": 2, "memberships": 60, "labels": 1},
    "questions": {"total": 1, "questions": [QUESTION]},
    "question": {"question": QUESTION, "sources": [QUESTION_SOURCE], "thread": []},
    "question_quality": QUESTION_QUALITY,
    "question_plan": PLAN,
    "decide_question": QUESTION,
    "assessments": {"total": 1, "assessments": [ASSESSMENT]},
    "assessment_quality": ASSESSMENT_QUALITY,
    "assessment_plan": ASSESSMENT_PLAN,
}

#: What the upload page's own client reports, which is the only figure on
#: the one page that has no listing.
UPLOAD_COUNTS = {"documents": 3, "upload_attempts": 5}

#: What the health page's own client reports. Three services, because the
#: three states a service can be in are what the page exists to tell apart:
#: answering, not answering, and serving no port to ask.
HEALTH: dict[str, Any] = {
    "reachable": (True, "Reachable."),
    "services": [
        {
            "name": "argilla",
            "purpose": "Where a person accepts or rejects.",
            "ok": True,
            "detail": "Listening on argilla:6900.",
            "url": "http://localhost:6900",
        },
        {
            "name": "parse-worker",
            "purpose": "Turns a file into a document.",
            "ok": None,
            "detail": "Serves no port.",
            "url": None,
        },
        {
            "name": "redis",
            "purpose": "The lock a stage takes.",
            "ok": False,
            "detail": "redis:6379 refused the connection.",
            "url": None,
        },
    ],
    "components": {
        "database": {"ok": True, "detail": "Connected.", "metrics": {}},
        "ingestion": {
            "ok": True,
            "detail": "Documents held.",
            "metrics": UPLOAD_COUNTS,
        },
    },
}

#: The whole pipeline, as `GET /pipeline` reports it: a corpus part way
#: through, with something for every control to act on.
PIPELINE: dict[str, Any] = {
    "stages": [
        {"stage": "parsing", "working": False, "rows": {"new": 2, "parsed": 3}},
        {"stage": "chunking", "working": False, "rows": {"chunked": 3}},
        {"stage": "extraction", "working": False, "rows": {"failed": 1}},
        {"stage": "topics", "working": False, "rows": {"modelled": 4}},
        {"stage": "questions", "working": False, "rows": {"new": 4}},
        {"stage": "assessment", "working": False, "rows": {}},
    ],
    "orchestration": {
        "available": True,
        "running": None,
        "recent": [],
        "automation": {
            "available": True,
            "on_arrival": False,
            "nightly": False,
            "detail": None,
        },
        "detail": None,
    },
    "working": False,
    "failed": 1,
}


def pipeline(**replaced: Any) -> dict:
    """The pipeline state, with whatever a test needs changed."""
    return {**PIPELINE, **replaced}


def orchestration(**replaced: Any) -> dict:
    """The orchestrator's half of it, with whatever a test needs changed.

    An orchestrator that is not available reports its two triggers
    unavailable too, because that is what `GET /pipeline` answers: there is
    nothing there to have switched anything on.
    """
    built = {**PIPELINE["orchestration"], **replaced}
    if not built["available"] and "automation" not in replaced:
        built["automation"] = {
            "available": False,
            "on_arrival": False,
            "nightly": False,
            "detail": None,
        }
    return built


#: What the pipeline panel answers with, before a test changes anything.
#: The three verbs and the switch answer `(did it, what to say)`.
PIPELINE_CLIENT: dict[str, Any] = {
    "state": PIPELINE,
    "run": (True, "Run abc12345 started."),
    "act": lambda action: (True, f"{action}: 7 row(s) moved."),
    "automate": (True, "Saved."),
}

#: The client each page outside the catalogue holds, and what it answers.
#: `open_view` gives a page every client it reaches for, so a test naming a
#: page never has to know which of these it is.
OUTSIDE: dict[str, dict[str, Any]] = {
    "upload": {"upload_api": {"counts": UPLOAD_COUNTS}},
    "health": {"health_api": HEALTH},
    # Documents carries a second client, unlike the other five listings:
    # the Pipeline panel asks the orchestrator to run every stage, which is
    # not a thing the catalogue can be asked for.
    "documents": {"pipeline_api": PIPELINE_CLIENT},
}

#: The stage each page is allowed to run, and nothing else.
OWNED = {
    "documents": "parsing",
    "passages": "chunking",
    "facts": "extraction",
    "topics": "topics",
    "questions": "questions",
    "assessment": "assessment",
}

#: The pages that list something and let a row be picked.
LISTING = tuple(OWNED)


def answers(**replaced: Any) -> dict:
    """The scripted catalogue backend, with whatever a test needs changed."""
    return {**CATALOG, **replaced}


def setting(name: str, **replaced: Any) -> dict:
    """One setting as /settings/{service} describes it."""
    return {
        "name": name,
        "kind": "integer",
        "help": "What it does, in one line.",
        "value": "5",
        "default": "5",
        "stored": False,
        "optional": False,
        "fixed": False,
        "invalidates": [],
        "changed_at": None,
        "low": None,
        "high": None,
        "choices": [],
        **replaced,
    }


#: What a service is configured by, as a page draws it. One of each control
#: the panel knows how to draw, so a page test covers every branch of it.
SETTINGS: dict[str, Any] = {
    "service": "any",
    "version": "a1b2c3d4e5f6",
    "settings": [
        setting("A_COUNT", low=1),
        setting("A_SHARE", kind="decimal", value="0.5", default="0.5", low=0, high=1),
        setting("A_FLAG", kind="boolean", value="true", default="true"),
        setting(
            "A_MODE",
            kind="text",
            value="fast",
            default="fast",
            choices=["fast", "accurate"],
        ),
        setting(
            "A_LIST",
            kind="csv",
            value="one,two",
            default="one,two",
            choices=["one", "two", "three"],
        ),
        setting("A_NAME", kind="text", value="a name", default="a name"),
        setting("A_MODEL", kind="text", value=None, default=None, optional=True),
        setting("A_POOL", value="5", default="5", fixed=True),
    ],
}


def settings(**replaced: Any) -> dict:
    """The scripted settings backend, with whatever a test needs changed."""
    return {
        "settings": lambda service: {**SETTINGS, **replaced},
        "change_settings": lambda service, values, version: {
            "service": service,
            "version": "changed",
            "changed": sorted(values),
            "cleared": [],
            "stale": [],
            "detail": "1 setting(s) changed.",
        },
    }


class Answers:
    """A backend client that answers from a script, and records the asks.

    Here rather than in conftest.py because a test module cannot safely say
    `from conftest import Answers`: pytest imports every conftest under a
    directory with no `__init__.py` as the bare name `conftest`, so the
    first one loaded owns the name for the process. With `tests/unit` ahead
    of `tests/frontend` in the collection order the four frontend modules
    that imported it resolved to `tests/unit/topics/conftest.py` and failed
    to collect. Alphabetical order was the only thing hiding it.
    """

    def __init__(self, **answers: object) -> None:
        """Initialises the client with one answer per method name."""
        self._answers = answers
        self.asked: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        """Answers whatever the page calls, or raises what it was given."""
        if name not in self._answers:
            raise AttributeError(name)

        def answer(*args, **kwargs):
            """Records the call and gives back the scripted answer."""
            self.asked.append((name, args, kwargs))
            prepared = self._answers[name]
            if isinstance(prepared, Exception):
                raise prepared
            return prepared(*args, **kwargs) if callable(prepared) else prepared

        return answer

    def calls(self, method: str) -> list[tuple[str, tuple, dict]]:
        """Every ask of one method, which is what most tests assert on."""
        return [one for one in self.asked if one[0] == method]


#: Every view that has a page class of its own, by the view it drives. A
#: subclass registers itself here by declaring `view=`, so `open_view` can
#: hand a test the page object for the view it asked for without a table
#: anybody has to remember to update.
PAGES: dict[str, type[View]] = {}


def page_for(name: str) -> type[View]:
    """The page class one view is worked through, or the plain one."""
    return PAGES.get(name, View)


class View:
    """Any view, as a reader sees and works it.

    The base holds what every page has, which is nearly everything: the
    layout is deliberately the same on all of them. A subclass exists only
    where a page has something no other page does, and says so by naming
    the view it drives.
    """

    def __init_subclass__(cls, view: str, **kwargs) -> None:
        """Registers one page class against the view it drives."""
        super().__init_subclass__(**kwargs)
        cls.VIEW = view
        PAGES[view] = cls

    def __init__(self, app, name: str, selected: int | None = None) -> None:
        """Wraps a run of one view, remembering which row is picked."""
        self.app = app
        self.name = name
        self.selected = selected

    @property
    def table_key(self) -> str:
        """What this page's table is keyed with, which a row is picked through.

        Derived rather than tabulated: the table on every page is keyed with
        the page's own name, and the table that listed the six catalogue
        pages left `health` out - so the one test that picks a row there
        wrote session state by hand instead of calling `select`.
        """
        return f"{self.name}-table"

    def _rerun(self) -> View:
        """Runs the script again, with whatever row was picked still picked.

        A dataframe selection is real widget state in a browser and
        survives a rerun. AppTest has no click for one, so it is written
        straight into session state - and written again here, because a
        value Streamlit never registered as a widget's does not survive the
        run that read it.
        """
        if self.selected is not None:
            self.app.session_state[self.table_key] = {
                "selection": {"rows": [self.selected], "columns": []}
            }
        return type(self)(self.app.run(), self.name, self.selected)

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

    def panels(self) -> list[str]:
        """Each section's label, in the order the page drew them."""
        return re.findall(r"<div class='qa-panel-label'>(.*?)</div>", self.html())

    def html(self) -> str:
        """Every panel label, state line and note, as one string.

        These are drawn with `st.html`, which none of the prose accessors
        reach.
        """
        return " ".join(str(one.value) for one in self.app.get("html"))

    def tables(self) -> str:
        """Every table's contents in full, as one string.

        `to_string`, because a DataFrame's repr elides columns and rows.
        """
        return " ".join(
            one.value.to_string() if hasattr(one.value, "to_string") else str(one.value)
            for one in self.app.dataframe
        )

    def stats(self) -> dict[str, str]:
        """Every figure on the page, by its label."""
        return {one.label: one.value for one in self.app.metric}

    def explanations(self) -> dict[str, str]:
        """The tooltip behind every figure, by its label."""
        return {one.label: one.help or "" for one in self.app.metric}

    def buttons(self) -> dict[str, Any]:
        """Every control, by its label."""
        return {one.label: one for one in self.app.button}

    def control_keys(self) -> list[str]:
        """Every control's key, which says what the control does."""
        return [one.key for one in self.app.button]

    def button(self, label: str):
        """One control by label, or None if the page does not offer it."""
        return self.buttons().get(label)

    def progress_bars(self) -> list[Any]:
        """Every progress bar, of which there should be none anywhere."""
        return list(self.app.get("progress"))

    def folds(self) -> list[str]:
        """Every expander's label."""
        return [one.label for one in self.app.get("expander")]

    def embedded(self) -> list[Any]:
        """The figure, which `components.html` renders as an iframe."""
        return list(self.app.get("iframe"))

    def toasts(self) -> list[str]:
        """What the page reported as a transient message."""
        return [str(one.value) for one in self.app.get("toast")]

    def widget_keys(self) -> set[str]:
        """Every input's key, which is how search is told from a filter."""
        found = set()
        for kind in (
            "selectbox",
            "text_input",
            "number_input",
            "checkbox",
            "segmented_control",
            "multiselect",
            "radio",
        ):
            found |= {one.key for one in getattr(self.app, kind) if one.key}
        return found

    # ── The controls, by label and by key ────────────────────────────────

    #: The pickers a value is chosen in, in the order `_widget` searches
    #: them. A key is unique across all of them, so the first hit is it.
    PICKERS: ClassVar[tuple[str, ...]] = (
        "selectbox",
        "segmented_control",
        "multiselect",
        "radio",
    )

    def _widget(self, key: str):
        """One picker by its key, whichever kind of picker it is.

        A test asks what a control offers, not which Streamlit primitive
        draws it - and the primitive has changed under two of these already
        (the assessment tabs are a segmented control and were a selectbox).

        Through the attribute rather than `app.get`, which is keyed on the
        proto's element name: `st.segmented_control` draws a `button_group`
        and `app.get("segmented_control")` is empty.
        """
        for kind in self.PICKERS:
            for one in getattr(self.app, kind):
                if one.key == key:
                    return one
        raise AssertionError(f"no picker keyed {key!r}: {sorted(self.widget_keys())}")

    def options(self, key: str) -> list[Any]:
        """Everything one picker offers, in the order it offers them."""
        return list(self._widget(key).options)

    def chosen(self, key: str) -> Any:
        """What one picker is currently set to."""
        return self._widget(key).value

    def tooltip(self, key: str) -> str:
        """The help behind one picker, which says what choosing does.

        Not `explanations`, which is the tooltip on every FIGURE by label.
        Two names a letter apart for two different things is how a test
        ends up asserting about the wrong one.
        """
        return self._widget(key).help or ""

    def numbers(self) -> dict[str, Any]:
        """Every number box, by its label."""
        return {one.label: one for one in self.app.number_input}

    def texts(self) -> dict[str, Any]:
        """Every text box, by its label."""
        return {one.label: one for one in self.app.text_input}

    def checkboxes(self) -> dict[str, Any]:
        """Every checkbox, by its label."""
        return {one.label: one for one in self.app.checkbox}

    def pickers(self) -> dict[str, Any]:
        """Every dropdown, by its label."""
        return {one.label: one for one in self.app.selectbox}

    def multiselects(self) -> dict[str, Any]:
        """Every multi-choice control, by its label."""
        return {one.label: one for one in self.app.multiselect}

    def ticked(self, prefix: str) -> dict[str, bool]:
        """Whether each checkbox under one key prefix is on.

        By prefix because a page draws one per artefact kind, and what a
        test asserts is the set of them and their states together.
        """
        return {
            one.key: one.value
            for one in self.app.checkbox
            if one.key and one.key.startswith(prefix)
        }

    # ── What the page said, by the voice it said it in ───────────────────

    def errors(self) -> list[str]:
        """Everything drawn as an error."""
        return [str(one.value) for one in self.app.error]

    def warnings(self) -> list[str]:
        """Everything drawn as a warning."""
        return [str(one.value) for one in self.app.warning]

    def notes(self) -> list[str]:
        """Everything drawn as an information note."""
        return [str(one.value) for one in self.app.info]

    def captions(self) -> list[str]:
        """Everything drawn as a caption."""
        return [str(one.value) for one in self.app.caption]

    def downloads(self) -> list[Any]:
        """Every file the page is offering, of which there is usually none."""
        return list(self.app.get("download_button"))

    def uploader(self) -> list[Any]:
        """The file picker, which only the upload page has."""
        return list(self.app.get("file_uploader"))

    def table_with(self, column: str):
        """The one table carrying a named column.

        By column and not by index: the Analysis fold draws its own tables
        above the listing, so `dataframe[0]` is the queue counts on every
        page that has one.
        """
        for frame in self.app.dataframe:
            if column in getattr(frame.value, "columns", ()):
                return frame.value
        raise AssertionError(f"no table on {self.name} carries a {column!r} column")

    # ── What a reader does ───────────────────────────────────────────────

    def press(self, label: str) -> View:
        """Presses one control and re-runs the page."""
        found = self.button(label)
        assert found is not None, f"no {label!r} control: {sorted(self.buttons())}"
        found.click()
        return self._rerun()

    def select(self, row: int = 0) -> View:
        """Picks one row out of the page's table."""
        self.selected = row
        return self._rerun()

    def choose(self, key: str, value: Any) -> View:
        """Picks a value in one of the page's pickers."""
        self.app.selectbox(key).select(value)
        return self._rerun()

    def type_in(self, key: str, value: str) -> View:
        """Types into one of the page's inputs."""
        self.app.text_input(key).set_value(value)
        return self._rerun()

    def uncheck(self, key: str) -> View:
        """Clears one of the page's checkboxes."""
        self.app.checkbox(key).uncheck()
        return self._rerun()

    def save_configuration(self) -> View:
        """Presses Save in the configuration panel every page draws.

        By label and not by position: the stage's own controls are drawn
        above it, and which index it lands on differs from page to page.
        """
        return self.press("Save")


class AssessmentPage(View, view="assessment"):
    """The evaluation phase's page, which is the one with two verdicts a row.

    Everything else about it is the shared layout. What is here is the
    reading of a judgement: which table carries it, and which kinds the
    page would queue for judging.
    """

    def verdicts(self):
        """The table of judgements, which is not the first table on the page."""
        return self.table_with("Judge")

    def judging(self) -> dict[str, bool]:
        """Which artefact kinds are ticked to be judged, by kind."""
        return {
            key.removeprefix("assessment-judge-"): value
            for key, value in self.ticked("assessment-judge-").items()
        }
