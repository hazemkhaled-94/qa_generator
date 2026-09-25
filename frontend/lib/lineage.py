"""How one artefact was produced, in the order the pipeline produced it.

One panel, drawn on every page that shows an artefact. It asks
`GET /lineage/{kind}/{id}` and renders the chain: the document, the
passages cut from it, the facts read out of those, the topics they sit in,
and the questions written from them - each step numbered by the stage's
place in the pipeline.

Behind a selection rather than in the first section, like everything else
that is not needed on arrival.
"""

from __future__ import annotations

import requests
import streamlit as st

from lib import backend, config

#: What each step is called on the page, by the stage that produces it.
#: The number is the stage's place in the pipeline and is what orders
#: every surface that shows a chain - the same numbering Phoenix's
#: projects, Argilla's datasets and the Grafana folders carry.
STEPS = {
    "parsing": ("Document", "uploaded, then read into a structured document"),
    "chunking": ("Passages", "the document cut into passages"),
    "extraction": ("Facts", "the claims read out of those passages"),
    "topic_modelling": ("Topics", "the subjects those passages sit in"),
    "question_generation": ("Questions", "written from those facts"),
}

#: Which store answers what, for the caption under the chain. The same
#: division telemetry/README.md states, said where somebody is looking.
WHERE = (
    "**Phoenix** holds the calls behind each step, **Grafana** every line "
    "they wrote, **Argilla** the artefacts a person reviewed, and "
    "**Dagster** which run produced them."
)


def _phoenix(artifact: dict) -> str:
    """Links one artefact to the calls that produced it, where it can."""
    trace_id, span_id = artifact.get("trace_id"), artifact.get("span_id")
    if not trace_id and not span_id:
        return "—"
    if not config.PHOENIX_BASE_URL:
        return f"`{(span_id or trace_id or '')[:12]}`"
    return ", ".join(
        f"[{label}]({config.PHOENIX_BASE_URL}/redirects/{kind}/{one})"
        for label, kind, one in (
            ("span", "spans", span_id),
            ("trace", "traces", trace_id),
        )
        if one
    )


def _verdict(artifact: dict) -> str:
    """What the stage decided, and what the judge made of it."""
    said = artifact.get("verdict") or "—"
    if artifact.get("reason"):
        said = f"{said} · {artifact['reason']}"
    if artifact.get("judge"):
        said = f"{said} · judge: {artifact['judge']}"
    return said


def _gates(chain: dict) -> None:
    """Every gate that read this question, at its fixed position.

    Only the gates that ran: the checker returns on the first failure, so
    a gate after the one that refused a question never read it, and
    showing it as passed would be a claim nobody made.
    """
    if not chain.get("gates_recorded"):
        st.caption(
            "Written before the gate sequence was recorded on the row, so "
            "only the gate that refused it is known. Phoenix has the rest "
            "where the trace is still held."
        )
        return
    gates = chain.get("gates") or []
    if not gates:
        st.caption("No gate read this question.")
        return
    st.table(
        [
            {
                "Gate": f"{one['position']}. {one['name']}",
                "Verdict": "passed" if one["passed"] else "refused",
            }
            for one in gates
        ]
    )
    st.caption(
        f"{len(gates)} of 6 gates read it. The ones after the last are not "
        "listed because they never ran, which is different from passing."
    )


def _step(step: dict) -> None:
    """One stage of the chain, and the artefacts this chain holds at it."""
    title, what = STEPS.get(step["stage"], (step["kind"].title(), step["stage"]))
    shown, total = len(step["artifacts"]), step["total"]
    counted = f"{total:,}" if shown == total else f"{shown:,} of {total:,}"
    st.markdown(f"**{step['position']}. {title}** · {counted} — {what}")
    st.table(
        [
            {
                "Id": one["id"][:12],
                "What it says": one["label"],
                "Verdict": _verdict(one),
                "In Phoenix": _phoenix(one),
            }
            for one in step["artifacts"]
        ]
    )


def panel(kind: str, artifact_id: str | int) -> None:
    """Draws one artefact's chain, inside the fold the page already has.

    Builds its own client, like the configuration panel: the pages that
    draw this do not all hold one.
    """
    client = backend.lineage_api()
    try:
        chain = client.lineage(kind, str(artifact_id))
    except requests.exceptions.RequestException as error:
        st.caption(f"Nothing traced: {error}")
        return

    st.caption(
        f"Everything this {kind} was produced from, in the order the "
        f"pipeline produced it. {WHERE}"
    )
    for step in chain["steps"]:
        _step(step)
    if kind == "question":
        st.markdown("**Gates**")
        _gates(chain)
