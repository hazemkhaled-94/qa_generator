"""System health view."""

from __future__ import annotations

from html import escape
from itertools import batched

import streamlit as st

from lib import backend, config, page

#: Cards per row. Fixed, so a card is the same width however many components
#: the backend reports and a new one wraps instead of squeezing the rest.
_PER_ROW = 4

#: What a service's own figures count, keyed by the service and the name it
#: reports the figure under. Both parts are needed: `documents` is a bucket
#: to the object store and a table to ingestion, and `parsed` is a bucket
#: here and a queue status everywhere else.
#:
#: Queue statuses are explained once in page.STATUS_HELP and shared with the
#: four pipeline pages, so nothing is worded twice.
_COUNTS = {
    ("object_store", "documents"): "Objects in the bucket holding uploaded "
    "files, one per stored document. Should match ingestion's document count.",
    ("object_store", "parsed"): "Objects in the bucket holding converted "
    "documents, one per document parsing has finished.",
    ("object_store", "export"): "Objects in the bucket holding generated "
    "datasets and coverage reports. Nothing writes to it yet.",
    ("ingestion", "documents"): "Documents in the catalogue, whatever stage "
    "each has reached.",
    ("ingestion", "upload_attempts"): "Uploads recorded, accepted and refused "
    "alike. Higher than the document count is normal: re-uploading a file "
    "already held is recorded and stores nothing.",
    ("chunking", "passages"): "Passages chunking has produced across every "
    "document. This is the unit extraction queues over.",
    ("extraction", "facts"): "Facts extraction has written, including those "
    "that failed a check. Rejected facts are kept so the failure rate can be "
    "measured.",
    ("extraction", "validated"): "Facts that passed every check. Only these "
    "are usable for question generation.",
    ("topic_modelling", "memberships"): "Passage-to-topic links held. A "
    "passage belongs to several topics with a weight on each, so this is well "
    "above the passage count.",
    ("topic_modelling", "passages_with_a_topic"): "Passages placed in at "
    "least one topic. Passages below this and the corpus total sit outside "
    "every topic-weighted report.",
}


def card(column, label: str, ok: bool, detail: str) -> None:
    """Renders one status card."""
    pill = "pill-ok" if ok else "pill-no"
    column.html(
        "<div class='qa-card'>"
        f"<div class='qa-card-label'>{escape(label)}</div>"
        f"<div class='qa-card-pill'><span class='pill {pill}'>"
        f"{'ok' if ok else 'failed'}</span></div>"
        f"<div class='qa-card-detail'>{escape(detail)}</div>"
        "</div>"
    )


def view() -> None:
    """Renders the system health page.

    Draws a card per component the backend reports, then its numbers, so a
    new component appears without a change here.
    """
    page.header(
        "System health",
        "Whether every component behind the API is reachable, and what each "
        "one currently holds.",
    )

    client = backend.health_api()
    api_ok, api_message = client.reachable()
    components = client.components()

    page.section(
        "Components",
        "One card per thing the backend depends on. `ok` means it answered; "
        "`failed` means it did not, and the reason is on the card. The "
        "frontend holds one address and reaches everything else through it, "
        "so a failed backend card makes every other card unknowable.",
    )
    cards = [("Backend", api_ok, f"{config.BACKEND_URL} — {api_message}")]
    cards += [
        (name.replace("_", " ").title(), state["ok"], state["detail"])
        for name, state in components.items()
    ]

    for row in batched(cards, _PER_ROW):
        # strict=False: the last row is short, and its cards keep the width
        # the full rows have rather than spreading to fill it.
        for column, (label, ok, detail) in zip(st.columns(_PER_ROW), row, strict=False):
            card(column, label, ok, detail)

    if not components:
        st.warning("The backend is unreachable, so there is nothing to report.")
        return

    for name, state in components.items():
        if not state["metrics"]:
            continue
        st.divider()
        page.section(
            name.replace("_", " ").title(),
            f"What this service holds right now. {state['detail']}",
        )
        page.metrics(
            {
                label.replace("_", " ").title(): (
                    "—" if value is None else f"{value:,}",
                    _explain(name, label),
                )
                for label, value in state["metrics"].items()
            }
        )


def _explain(service: str, label: str) -> str:
    """Says what one of a service's own numbers counts.

    The labels come from the backend rather than from here, so a figure this
    page has no wording for still gets an explanation naming where it came
    from rather than no tooltip at all.
    """
    named = _COUNTS.get((service, label))
    if named is not None:
        return named
    if label in page.STATUS_HELP:
        return f"{service.replace('_', ' ').capitalize()}: {page.STATUS_HELP[label]}"
    return (
        f"Reported by the {service.replace('_', ' ')} service under the name "
        f"`{label}`. This page has no wording for it."
    )


page.render(view)
