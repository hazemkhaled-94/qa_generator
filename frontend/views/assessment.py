"""Assessment view. Runs the evaluation phase, and no other stage.

The last page, for the last phase. Everything the five pages before it
show is the pipeline's own reading of its own work; this one shows a
second opinion held about all of it by a model that wrote none of it.

The number worth the page is `Disagreements`: the artefacts the pipeline
KEPT and the judge refused. Everything else here is context for it.
"""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, configure, explain, page, stage

#: The one stage this page runs.
_ASSESSMENT = stage.Queue("assessment", "Assessment", "artifacts", "assessed")

#: The order this phase works in, and what runs each step.
_STEPS = (
    explain.Step(
        "Claim what has not been judged",
        "Facts, topics and questions alike. An unanswerable question is "
        "skipped: it has no answer for a judge to check, so asking would "
        "score the absence of one.",
        ("queue.py",),
        (),
    ),
    explain.Step(
        "Ask one judgement at a time",
        "Three calls for a fact, two for a topic, six for a question — one "
        "per metric rather than one call answering several, because a call "
        "asked for five things answers the small ones in whatever language "
        "the hard one was thinking in.",
        ("service.py", "prompts.py"),
        ("ASSESSMENT_JUDGE_MODEL", "ASSESSMENT_KINDS"),
    ),
    explain.Step(
        "Record the verdict and the reasoning",
        "Both halves. The label is what the figures group on; the "
        "explanation is the half nothing else in this application stores, "
        "and the search box above looks in it.",
        ("repository.py",),
        (),
    ),
    explain.Step(
        "Report the disagreements",
        "The artefacts the pipeline KEPT and the judge refused. That number "
        "is what this phase exists to produce — everything else here is "
        "context for it.",
        ("service.py",),
        (),
    ),
)

#: Everything this page can say about the phase it runs. It has no gates,
#: and saying so is the point: `Service.gates` is empty and the fold's gate
#: tab says why rather than showing an empty table.
_SERVICE = explain.Service(
    what=(
        "A second opinion on work the pipeline already decided about, held "
        "by a model that wrote none of it.\n\n"
        "**This phase is not a gate and cannot become one.** It writes "
        "`assessments.approved`, which no stage reads; whether a fact is "
        "valid stays the checker's and whether a question is accepted stays "
        "the gates'. That is measured, not cautious: over nineteen labelled "
        "cases a served model answered a phrasing judgement 7/7 in English "
        "and 6/12 in German, which is chance, and half this corpus is "
        "German. A phase that could reject rows would have thrown away good "
        "German questions on the strength of a coin.\n\n"
        "What a judge is good for is the **disagreement**, at corpus scale."
    ),
    steps=_STEPS,
)

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
    "refusal": (
        "Refusal",
        "answered",
        "The answer answers, rather than declining to. Caught by nothing else.",
    ),
    "conciseness": (
        "Conciseness",
        "concise",
        "The answer gives what was asked for and stops.",
    ),
    "toxicity": (
        "Toxicity",
        "non-toxic",
        "The pipeline wrote nothing offensive of its own.",
    ),
}

#: The tab strip, as (what a person picks, what /assessment takes). `None`
#: is every kind, and it is first because arriving at a page that has
#: silently narrowed to one third of the corpus is arriving at a lie.
_TABS: dict[str, str | None] = {
    "All artefacts": None,
    "Facts": "fact",
    "Topics": "topic",
    "Questions": "question",
}


def _tabs() -> None:
    """Draws the artefact selector that everything below it answers to."""
    st.segmented_control(
        "Artefact",
        list(_TABS),
        key="assessment-tab",
        default=next(iter(_TABS)),
        label_visibility="collapsed",
        help="Which artefact this page is about. The figures, the analysis, "
        "the search and the table all narrow with it.",
    )


def _looking_at() -> str | None:
    """Which kind the page is showing, or None for all of them.

    Read from session state rather than from the control's return value,
    because the control is drawn inside the Overview panel and this is
    needed before it, to fetch what that panel is going to report.
    """
    return _TABS.get(st.session_state.get("assessment-tab") or "", None)


def _judging(plan: dict) -> list[tuple[str, str]]:
    """Draws the kind checkboxes and returns what they select.

    Separate from the tab above, and the two answer different questions:
    the tab is what a person is LOOKING at, and these are what they are
    asking to JUDGE. Reading one corpus while queueing another is a
    reasonable thing to want, and merging the two controls would make it
    impossible.

    Defaults to what ASSESSMENT_KINDS names, so the page opens agreeing
    with what the worker and the Dagster asset would do on their own.
    """
    configured = set(plan.get("kinds") or _KINDS)
    st.caption("Judge:")
    picked = []
    for column, (kind, (label, _)) in zip(
        st.columns(len(_KINDS)), _KINDS.items(), strict=True
    ):
        metrics = len(plan.get("metrics", {}).get(kind, ()))
        if column.checkbox(
            label,
            key=f"assessment-judge-{kind}",
            value=kind in configured,
            help=f"{metrics} model call(s) per {label.lower().rstrip('s')}.",
        ):
            picked.append(("kind", kind))
    return picked


def _scopes(picked: list[tuple[str, str]]) -> list[tuple[str, str]] | None:
    """What the queue verbs should act on, given what is ticked.

    None is the whole queue, and it is the right answer to two different
    tickings: every kind, which is the same request said the long way and
    is cheaper as one call than three, and none, which is somebody who has
    not chosen and should not be given controls that do nothing.

    The two read differently to a person, which is what the caption beside
    this says, and identically to the queue - so they are one return value
    here and two sentences there.
    """
    return None if len(picked) in (0, len(_KINDS)) else picked


def view() -> None:
    """Renders the assessment page."""
    page.header("Assessment")

    client = backend.catalog_api()
    overview, analysis, service = st.container(), st.container(), st.container()

    plan = client.assessment_plan()

    looking_at = _looking_at()

    words, field = catalog.search("assessment", _FIELDS)
    chosen, pager = catalog.filters(
        "assessment",
        {
            "metric": (
                "Refused by",
                list(_METRICS),
                "Show only the artefacts one named judgement did NOT approve. "
                + " ".join(f"{name}: {what}" for name, _, what in _METRICS.values()),
            ),
        },
    )

    only_disagreements = st.session_state.get("assessment-disagreements", False)
    where = {
        "kind": looking_at,
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
    quality = client.assessment_quality(kind=looking_at)
    counts = client.stage_status("assessment")["rows"]

    with overview, page.panel("Overview"):
        _tabs()
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
        page.findings(_checks(quality, counts, plan, looking_at))
        st.caption(
            "A judgement is never a gate. What decides whether a fact is "
            "kept is the checker, and what decides a question is the gates; "
            "this is a second opinion recorded beside them so the two can "
            "be compared. An LLM judge was measured at chance on the German "
            "half of one corpus, which is why it cannot reject anything."
        )

    with service, page.panel("Assessment — the evaluation phase"):
        judging = _judging(plan)
        stage.service(client, _ASSESSMENT, scopes=_scopes(judging))
        if not judging:
            st.caption(
                "Nothing is ticked, so the controls act on every artefact. "
                "Tick one or more to judge those alone."
            )
        elif len(judging) < len(_KINDS):
            st.caption(
                "The controls act on "
                + " and ".join(_KINDS[kind][0].lower() for _, kind in judging)
                + " only."
            )
        st.caption(
            f"Judging {', '.join(plan['kinds'])} with "
            f"{plan['judge_model'] or 'LLM_MODEL, which is marking its own work'}"
            f" · template version {plan['prompt_version']}"
        )
        explain.panel(_SERVICE, "How the evaluation phase works")
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


def _asked_of(plan: dict, looking_at: str | None) -> set[str]:
    """Which metrics apply to whatever the page is showing.

    Read off the plan rather than listed here, so a metric added to a kind
    appears without an edit. With no tab chosen it is every metric any
    kind is asked, because the table below is then showing all of them.
    """
    metrics = plan.get("metrics") or {}
    if looking_at:
        return set(metrics.get(looking_at, ()))
    return {one for asked in metrics.values() for one in asked}


def _checks(
    quality: dict, counts: dict[str, int], plan: dict, looking_at: str | None
) -> list[dict]:
    """Builds the fold: the queue, then one row per metric and per kind."""
    rows = page.queue_rows(
        "Artifacts", counts, ("new", "pending", "in_progress", "assessed", "failed")
    )
    measured = {one["metric"]: one for one in quality.get("metrics", ())}
    asked = _asked_of(plan, looking_at)
    # Every metric, the ones that do not apply included and marked as
    # such: a check missing from a list cannot be told apart from a check
    # nobody wrote, and `summarization` being absent from the Questions
    # tab is a fact about the design rather than a gap in the run.
    rows += [
        {
            "Check": name,
            "Value": page.share(measured[code]["approved"], measured[code]["judged"])
            if code in measured
            else ("—" if code in asked else "not asked"),
            "Should be": f"all {approves}" if code in asked else "—",
            "State": "—",
            "What it means": (
                f"{what} Phoenix files this as `{code}`."
                if code in asked
                else f"Not asked of this artefact. {what}"
            ),
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
