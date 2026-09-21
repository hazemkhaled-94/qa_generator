"""System health view. Runs nothing, and configures what the stages share."""

from __future__ import annotations

import streamlit as st

from lib import backend, config, configure, page

#: What a service's own figures count, keyed by the service and the name it
#: reports the figure under. Both parts are needed: `documents` is a bucket
#: to the object store and a table to ingestion, and `parsed` is a bucket
#: here and a queue status everywhere else.
#:
#: Queue statuses are explained once in page.STATUS_HELP and shared with the
#: five pipeline pages, so nothing is worded twice.
_COUNTS = {
    ("object_store", "documents"): "Objects holding uploaded files.",
    ("object_store", "parsed"): "Objects holding converted documents.",
    ("object_store", "export"): "Objects holding datasets and reports.",
    ("ingestion", "documents"): "Documents in the catalogue.",
    ("ingestion", "upload_attempts"): "Uploads recorded, accepted and refused.",
    ("chunking", "passages"): "Passages chunking has produced.",
    ("extraction", "facts"): "Facts written, rejected ones included.",
    ("extraction", "validated"): "Facts that passed every check.",
    ("topic_modelling", "memberships"): "Passage-to-topic links held.",
    ("topic_modelling", "passages_with_a_topic"): "Passages in at least one topic.",
}


#: How a service's state reads. None is not a failure: a worker reads a
#: queue and serves no port, so there is nothing to connect to and a red
#: mark would say something untrue about it.
_STATE = {True: "up", False: "down", None: "no port"}


def _services(client) -> None:
    """Draws every container the deployment runs, with a link to each.

    The whole of compose rather than the browsable half. A page called
    System health that lists only what has a web page cannot answer "is
    Redis up", which is the question somebody has when a stage stops
    claiming rows.
    """
    running = client.services()
    if not running:
        st.caption("The backend is unreachable, so it cannot report the services.")
        return

    up = sum(1 for one in running if one["ok"] is True)
    down = [one for one in running if one["ok"] is False]
    page.stats(
        {
            "Services": (f"{len(running):,}", "Containers this deployment runs."),
            "Up": (f"{up:,}", "Accepted a connection on their port."),
            "Down": (f"{len(down):,}", "Refused one. A worker serves no port."),
        }
    )
    if down:
        st.warning(
            "Not answering: " + ", ".join(one["name"] for one in down),
            icon=":material/error:",
        )

    st.dataframe(
        [
            {
                "Service": one["name"],
                "State": _STATE[one["ok"]],
                "Open": one["url"],
                "What it does": one["purpose"],
            }
            for one in running
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Open": st.column_config.LinkColumn("Open", display_text="open"),
            "What it does": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "A service is *up* when it accepted a connection, which is not the "
        "same as working. What a stage is actually doing is on its own page."
    )


def view() -> None:
    """Renders the system health page.

    Draws a row per component the backend reports, so a new component
    appears without a change here.
    """
    page.header("System health")

    client = backend.health_api()
    api_ok, api_message = client.reachable()
    components = client.components()

    rows = [
        {
            "Component": "Backend",
            "State": "ok" if api_ok else "failed",
            "Detail": f"{config.BACKEND_URL} — {api_message}",
            "Figures": 0,
        }
    ]
    rows += [
        {
            "Component": name.replace("_", " ").title(),
            "State": "ok" if state["ok"] else "failed",
            "Detail": state["detail"],
            "Figures": len(state["metrics"]),
        }
        for name, state in components.items()
    ]

    with page.panel("Overview"):
        page.stats(
            {
                "Components": (f"{len(rows):,}", "Things the backend depends on."),
                "Failed": (
                    f"{sum(1 for one in rows if one['State'] == 'failed'):,}",
                    "Components that did not answer.",
                ),
            }
        )

    with page.panel("Services"):
        _services(client)

    # The model, the tokenizer and the language pipelines the six stages
    # share. A stage naming its own model does it on its own page.
    with page.panel("Platform"):
        configure.panel("platform")

    with page.panel(f"Components · {len(rows):,}"):
        picked = page.table(
            rows,
            key="health-table",
            column_config={"Detail": st.column_config.TextColumn(width="large")},
        )
        if not components:
            st.caption("The backend is unreachable, so there is nothing to report.")

    if picked is None:
        return
    name = list(components)[picked - 1] if picked else None
    if name is None:
        return

    with page.panel(name.replace("_", " ").title()):
        figures = components[name]["metrics"]
        if not figures:
            st.caption("This component reports no figures.")
            return
        st.dataframe(
            [
                {
                    "Figure": label.replace("_", " ").title(),
                    "Count": page.written(value),
                    "What it counts": _explain(name, label),
                }
                for label, value in figures.items()
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "What it counts": st.column_config.TextColumn(width="large")
            },
        )


def _explain(service: str, label: str) -> str:
    """Says what one of a service's own numbers counts."""
    named = _COUNTS.get((service, label))
    if named is not None:
        return named
    if label in page.STATUS_HELP:
        return f"{service.replace('_', ' ').capitalize()}: {page.STATUS_HELP[label]}"
    return f"Reported by {service.replace('_', ' ')} as `{label}`."


page.render(view)
