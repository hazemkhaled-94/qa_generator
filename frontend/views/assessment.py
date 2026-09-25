"""Assessment view. Runs the evaluation phase, and no other stage.

The last page, for the last phase. Everything the five pages before it
show is the pipeline's own reading of its own work; this one shows a
second opinion held about all of it by a model that wrote none of it.

The number worth the page is `Disagreements`: the artefacts the pipeline
KEPT and the judge refused. Everything else here is context for it.
"""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, configure, page, stage

#: The one stage this page runs.
_ASSESSMENT = stage.Queue("assessment", "Assessment", "artifacts", "assessed")

#: The columns the search box can look in, by the heading each carries,
#: mapped to the name /assessment takes. The explanations are the half
#: nothing else in this application can search: until this phase there
#: were no model opinions stored to search.
_FIELDS = {
    "Artefact and the judge's reasons": "both",
    "The artefact": "artifact",
    "What the judge said": "explanation",
}

#: What each artefact kind is called on the page, and what judging it asks.
_KINDS = {
    "fact": ("Facts", "Does the cited evidence support the statement?"),
    "topic": ("Topics", "Is this a good name, and a subject worth asking about?"),
    "question": ("Questions", "Is the answer right, and does it rest on the facts?"),
}

#: Every metric, by the name phoenix-evals gives it: what to call it here,
#: which label approves, and what the judgement is. Listed whether or not
#: anything failed it, because a metric missing from a list cannot be told
#: apart from one nobody asked.
_METRICS = {
    "hallucination": (
        "Hallucination",
        "factual",
        "Every part of the statement or answer is found in the evidence.",
    ),
    "qa_correctness": (
        "Answer correctness",
        "correct",
        "The expected answer answers the question that was asked.",
    ),
    "relevance": (
        "Relevance",
        "relevant",
        "The evidence bears on what the artefact is about.",
    ),
    "summarization": (
        "Summarisation",
        "good",
        "The label stands in for the terms beneath it.",
    ),
}


def view() -> None:
    """Renders the assessment page."""
    page.header("Assessment")

    client = backend.catalog_api()
    overview, analysis, service = st.container(), st.container(), st.container()

    plan = client.assessment_plan()

    words, field = catalog.search("assessment", _FIELDS)
    chosen, pager = catalog.filters(
        "assessment",
        {
            "kind": (
                "Artefacts",
                list(_KINDS),
                "Which artefact the verdict is about. "
                + " ".join(f"{label}: {asks}" for label, asks in _KINDS.values()),
            ),
            "metric": (
                "Refused by",
                list(_METRICS),
                "Show only the artefacts one named judgement did NOT approve.",
            ),
        },
    )

    only_disagreements = st.session_state.get("assessment-disagreements", False)
    where = {
        "kind": chosen["kind"],
        "metric": chosen["metric"],
        "q": words,
        "field": field,
    }
    total, rows = catalog.paged(
        "assessment",
        pager,
        lambda limit, offset: client.assessments(
            **where,
            disagreements=only_disagreements or None,
            limit=limit,
            offset=offset,
        ),
        "assessments",
    )
    quality = client.assessment_quality(kind=chosen["kind"])
    counts = client.stage_status("assessment")["rows"]

    with overview, page.panel("Overview"):
        if not plan["enabled"]:
            st.info(
                "The evaluation phase is switched off. Set "
                "`ASSESSMENT_ENABLED=true` in `.env` to run it. Nothing else "
                "in the pipeline is affected either way: this phase records "
                "an opinion beside the checker's verdict and never over it."
            )
        page.stats(
            {
                "Judged": (
                    page.share(quality["judged"], quality["total"]),
                    "Artefacts the judge has answered about.",
                ),
                "Approved": (
                    page.share(quality["approved"], quality["judged"]),
                    "Cleared every judgement their kind faces.",
                ),
                "Disagreements": (
                    f"{quality['disagreements']:,}",
                    (
                        "Kept by the pipeline and refused by the judge. The "
                        "only figure here worth acting on, and it is a queue "
                        "for a person rather than a verdict."
                    ),
                ),
                "Judge": (
                    ", ".join(quality["judge_models"]) or "—",
                    (
                        "The model that answered. It should not be the model "
                        "that wrote what it is judging."
                    ),
                ),
            }
        )
        st.checkbox(
            "Show only the disagreements",
            key="assessment-disagreements",
            help="The artefacts the pipeline kept and the judge refused. "
            "Neither side is right by default.",
        )

    with analysis, st.expander("Analysis"):
        page.findings(_checks(quality, counts))
        st.caption(
            "A judgement is never a gate. What decides whether a fact is "
            "kept is the checker, and what decides a question is the gates; "
            "this is a second opinion recorded beside them so the two can "
            "be compared. An LLM judge was measured at chance on the German "
            "half of one corpus, which is why it cannot reject anything."
        )

    with service, page.panel("Assessment — the evaluation phase"):
        stage.service(client, _ASSESSMENT)
        st.caption(
            f"Judging {', '.join(plan['kinds'])} with "
            f"{plan['judge_model'] or 'LLM_MODEL, which is marking its own work'}"
            f" · template version {plan['prompt_version']}"
        )
        configure.panel("assessment")

    with page.panel(f"Assessments · {total:,}"):
        if not rows:
            st.caption("Nothing matches.")
            return
        picked = page.table(
            [
                {
                    "Artefact": _KINDS.get(row["kind"], (row["kind"],))[0],
                    "Id": row["artifact_id"],
                    "Pipeline": row["verdict"] or "—",
                    "Judge": _approval(row["approved"]),
                    "Refused by": ", ".join(
                        _METRICS.get(one, (one,))[0] for one in _refused(row)
                    )
                    or "—",
                    "Disagrees": "yes" if _disagrees(row) else "no",
                    "What it is": row["summary"],
                }
                for row in rows
            ],
            key="assessment-table",
            column_config={
                "What it is": st.column_config.TextColumn(width="large"),
            },
        )

    if picked is not None:
        _detail(rows[picked])


def _approval(approved: bool | None) -> str:
    """How a verdict reads in a cell, with `not yet` kept distinct from no."""
    if approved is None:
        return "not yet"
    return "approved" if approved else "refused"


def _refused(row: dict) -> list[str]:
    """Which metrics did not approve one artefact."""
    return [one["metric"] for one in row.get("metrics", ()) if not one["approved"]]


def _disagrees(row: dict) -> bool:
    """Whether the pipeline kept this and the judge refused it."""
    return row["approved"] is False and row["verdict"] == "accepted"


def _checks(quality: dict, counts: dict[str, int]) -> list[dict]:
    """Builds the fold: the queue, then one row per metric and per kind."""
    rows = page.queue_rows(
        "Artifacts", counts, ("new", "pending", "in_progress", "assessed", "failed")
    )
    measured = {one["metric"]: one for one in quality.get("metrics", ())}
    rows += [
        {
            "Check": name,
            "Value": page.share(measured[code]["approved"], measured[code]["judged"])
            if code in measured
            else "—",
            "Should be": f"all {approves}",
            "State": "—",
            "What it means": f"{what} Phoenix files this as `{code}`.",
        }
        for code, (name, approves, what) in _METRICS.items()
    ]
    rows += [
        {
            "Check": f"{label} approved",
            "Value": page.share(
                quality.get("approved_by_kind", {}).get(code, 0),
                quality.get("by_kind", {}).get(code, 0),
            ),
            "Should be": "—",
            "State": "—",
            "What it means": asks,
        }
        for code, (label, asks) in _KINDS.items()
    ]
    rows.append(
        {
            "Check": "Disagreements",
            "Value": f"{quality['disagreements']:,}",
            "Should be": "—",
            "State": "Attention" if quality["disagreements"] else "OK",
            "What it means": "Kept by the pipeline, refused by the judge. "
            "Push them to Argilla and have somebody look.",
        }
    )
    rows.append(
        {
            "Check": "Waiting to be judged",
            "Value": f"{quality.get('outstanding', 0):,}",
            "Should be": "0 once the phase has run",
            "State": "—",
            "What it means": "Enrolled and not yet answered about, which is "
            "a phase part-way through rather than a judge that refused.",
        }
    )
    return rows


def _detail(row: dict) -> None:
    """Shows every judgement made about one artefact, and why."""
    label = _KINDS.get(row["kind"], (row["kind"], ""))[0]

    with page.panel(f"{label[:-1]} {row['artifact_id']}"):
        page.attributes(
            {
                "Artefact": label,
                "Id": row["artifact_id"],
                "What the pipeline said": row["verdict"],
                "What the judge said": _approval(row["approved"]),
                "Disagrees": _disagrees(row),
                "Judge": row["judge_model"],
                "Template version": row["prompt_version"],
                "Run": row["run_id"],
                "Judged": row["assessed_at"],
                "Could not be judged": row["error"],
                "What it is": row["summary"],
            }
        )

        if row.get("metrics"):
            st.dataframe(
                [
                    {
                        "Judgement": _METRICS.get(one["metric"], (one["metric"],))[0],
                        "Phoenix name": one["metric"],
                        "Answer": one["label"],
                        "Approved": "yes" if one["approved"] else "no",
                        "Why": one["explanation"],
                    }
                    for one in row["metrics"]
                ],
                width="stretch",
                hide_index=True,
                column_config={"Why": st.column_config.TextColumn(width="large")},
            )

        if _disagrees(row):
            st.warning(
                "The pipeline kept this and the judge refused it. Neither "
                "side is right by default - put it in front of somebody: "
                f"`make review-push-{row['kind']}s IDS={row['artifact_id']}`"
            )


page.render(view)
