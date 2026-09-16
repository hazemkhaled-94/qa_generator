"""Facts view. Runs extraction, and no other stage."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

#: The one stage this page runs. It queues over a passage, so a fact picked
#: below is run by the passage it was drawn from.
_EXTRACTION = stage.Queue("extraction", "Extraction", "passages", "extracted")

#: The columns the search box can look in, by the heading each carries in
#: the table below, mapped to the name /facts takes.
_FIELDS = {
    "Statement and evidence": "both",
    "Statement": "statement",
    "Evidence": "evidence",
}

#: The four readings a passage gets, by the value stored on the fact.
_KINDS = {
    "atomic": ("Atomic", "One claim from one sentence, cited by number."),
    "summary": ("Summary", "Two or three sentences standing in for a passage."),
    "outline": ("Outline", "The points one passage makes, one bullet each."),
    "bridge": ("Bridge", "One claim drawn from a group of passages."),
}

#: Every check a fact is put through, by the code it is rejected under:
#: which kinds face it, and what it tests. Listed whether or not anything
#: failed it, because a check missing from a list cannot be told apart from
#: a check nobody wrote.
_CHECKS = {
    "evidence_absent": (
        "Citation resolves",
        "every kind",
        "Every sentence number cited exists in the passage.",
    ),
    "copied": (
        "Not copied",
        "atomic",
        "The statement differs from the evidence it cites.",
    ),
    "asserts_nothing": (
        "Asserts something",
        "atomic, summary, bridge",
        "The statement carries a finite verb.",
    ),
    "not_atomic": (
        "Exactly one claim",
        "atomic, bridge",
        "The statement carries a single predicate.",
    ),
    "unsupported_addition": (
        "Nothing invented",
        "every kind",
        "Every number, name and date appears in the cited sentence.",
    ),
    "unresolved_reference": (
        "No dangling pronouns",
        "atomic, bridge",
        "The statement stands alone, without a pronoun from elsewhere.",
    ),
    "not_condensed": (
        "Shorter than its passage",
        "summary, outline",
        "The digest is well under the length of its passage.",
    ),
    "not_listed": (
        "Two points or more",
        "outline",
        "The outline holds at least two bullet points.",
    ),
    "not_bridging": (
        "Rests on two passages",
        "bridge",
        "The claim names at least two of the passages it was shown.",
    ),
}

#: What the decomposition ratio should clear. A fact carries one claim out
#: of a sentence that usually carries several.
_DECOMPOSITION_FLOOR = 1.2


def view() -> None:
    """Renders the facts page."""
    page.header("Facts")

    client = backend.catalog_api()
    overview, analysis, service = st.container(), st.container(), st.container()

    words, field = catalog.search("facts", _FIELDS)
    chosen, pager = catalog.filters(
        "facts",
        {
            "kind": (
                "Readings",
                list(_KINDS),
                "Which reading of the corpus. "
                + " ".join(f"{label}: {what}" for label, what in _KINDS.values()),
            ),
            "method": (
                "Methods",
                ["llm", "deterministic"],
                (
                    "How the fact was drawn: llm by the served model, "
                    "deterministic by a rule that needs no model."
                ),
            ),
        },
        documents=client.document_names(),
    )

    where = {
        "document": chosen["document"],
        "q": words,
        "field": field,
        "method": chosen["method"],
        "kind": chosen["kind"],
    }
    total, rows = catalog.paged(
        "facts",
        pager,
        lambda limit, offset: client.facts(**where, limit=limit, offset=offset),
        "facts",
    )
    quality = client.fact_quality(**where)
    counts = client.stage_status("extraction")["rows"]

    with overview, page.panel("Overview"):
        page.stats(
            {
                "Facts": (
                    f"{quality['total']:,}",
                    "Facts matching, rejected ones included.",
                ),
                "Passed": (
                    page.share(quality["validated"], quality["total"]),
                    "Cleared every check their kind faces.",
                ),
                "Per passage": (
                    f"{quality['facts_per_passage']:.1f}",
                    "Facts drawn from each passage that yielded any.",
                ),
                "Passages read": (
                    page.share(counts.get("extracted", 0), sum(counts.values())),
                    "Passages extraction has finished reading.",
                ),
            }
        )

    with analysis, st.expander("Analysis"):
        page.findings(_checks(quality, chosen["kind"], counts))

    with service, page.panel("Extraction"):
        stage.service(client, _EXTRACTION)

    with page.panel(f"Facts · {total:,}"):
        if not rows:
            st.caption("Nothing matches.")
            return
        picked = page.table(
            [
                {
                    "Passage": _ordinals(row),
                    "Page": _pages(row),
                    "Kind": _KINDS.get(row["kind"], (row["kind"],))[0],
                    "Method": row["extraction_method"],
                    "Passed": "yes" if row["validated"] else "no",
                    "Failed check": _CHECKS.get(row["rejection_code"], ("—",))[0]
                    if row["rejection_code"]
                    else "—",
                    "Statement": row["statement"],
                    "Cited": row["evidence_text"],
                }
                for row in rows
            ],
            key="facts-table",
            column_config={
                "Statement": st.column_config.TextColumn(width="large"),
                "Cited": st.column_config.TextColumn(width="medium"),
            },
        )

    if picked is not None:
        _detail(client, rows[picked])


def _ordinals(fact: dict) -> str:
    """Where each passage a fact rests on sits in its own document."""
    return ", ".join(str(one["ordinal"]) for one in fact["passages"]) or "—"


def _pages(fact: dict) -> str:
    """The pages a fact rests on, each named once."""
    found = dict.fromkeys(
        one["page_from"] for one in fact["passages"] if one["page_from"] is not None
    )
    return ", ".join(str(one) for one in found) or "—"


def _checks(quality: dict, kind: str | None, counts: dict[str, int]) -> list[dict]:
    """Builds the fold: the extraction queue, then one row per check."""
    total = quality["total"]
    rejected = quality["rejected"]

    rows = page.queue_rows(
        "Passages", counts, ("new", "pending", "in_progress", "extracted", "failed")
    )
    rows += [
        {
            "Check": name,
            "Value": page.share(rejected.get(code, 0), total),
            "Should be": "0",
            "State": "OK" if not rejected.get(code) else "Attention",
            "What it means": f"{tests} Applies to {applies}.",
        }
        for code, (name, applies, tests) in _CHECKS.items()
    ]
    rows.append(_decomposition_row(quality, kind))
    rows += [
        {
            "Check": kind_code,
            "Value": page.share(rejected[kind_code], total),
            "Should be": "0",
            "State": "Attention",
            "What it means": "A rejection code this page has no name for.",
        }
        for kind_code in sorted(set(rejected) - set(_CHECKS))
    ]
    rows += [
        {
            "Check": f"{label} facts",
            "Value": page.share(quality.get("kinds", {}).get(code, 0), total),
            "Should be": "—",
            "State": "—",
            "What it means": what,
        }
        for code, (label, what) in _KINDS.items()
    ]
    rows += [
        {
            "Check": "Statement length",
            "Value": f"{quality['mean_statement_chars']:.0f} characters, mean",
            "Should be": "under the evidence",
            "State": "—",
            "What it means": "Mean length of the claim extraction wrote.",
        },
        {
            "Check": "Claims per statement",
            "Value": f"{quality['mean_statement_predicates']:.2f}",
            "Should be": "near 1.00",
            "State": "—",
            "What it means": "Predicates spaCy found in the mean statement.",
        },
    ]
    return rows


def _decomposition_row(quality: dict, kind: str | None) -> dict[str, str]:
    """Builds the decomposition row, which no single fact can fail.

    Only atomic facts are decomposed: a summary keeps every claim its
    passage carried on purpose, so measuring it over a mixed set says
    nothing.
    """
    if kind != "atomic":
        return {
            "Check": "Decomposition",
            "Value": "—",
            "Should be": f"above {_DECOMPOSITION_FLOOR}×",
            "State": "—",
            "What it means": "Measured over atomic facts. Pick that reading.",
        }
    statement = quality["mean_statement_predicates"]
    ratio = quality["mean_evidence_predicates"] / statement if statement else 0
    return {
        "Check": "Decomposition",
        "Value": _decomposition(quality),
        "Should be": f"above {_DECOMPOSITION_FLOOR}×",
        "State": "OK" if ratio >= _DECOMPOSITION_FLOOR else "Attention",
        "What it means": "Claims the cited sentences carried, per claim kept.",
    }


def _decomposition(quality: dict) -> str:
    """How many claims the cited text carried for each one kept."""
    statement = quality["mean_statement_predicates"]
    if not statement:
        return "—"
    return f"{quality['mean_evidence_predicates'] / statement:.1f}×"


def _detail(client, fact: dict) -> None:
    """Shows everything held about one fact, and extracts its passage again."""
    held = fact["passages"]
    scope = ("passage", str(held[0]["passage_id"]))
    check = _CHECKS.get(fact["rejection_code"])
    label, _ = _KINDS.get(fact["kind"], (fact["kind"], ""))

    with page.panel(f"{label} fact from passage {_ordinals(fact)}"):
        page.attributes(
            {
                "Id": fact["id"],
                "Reading": label,
                "Method": fact["extraction_method"],
                "Passed": fact["validated"],
                "Failed check": check[0] if check else None,
                "Rejection code": fact["rejection_code"],
                "Rejection reason": fact["validation_error"],
                "Claims in statement": fact["statement_predicates"],
                "Claims in evidence": fact["evidence_predicates"],
                "Added to the statement": fact["units_added"],
                "Unresolved references": fact["unresolved_references"],
                "Rests on passages": _ordinals(fact),
                "Statement": fact["statement"],
                "Evidence cited": fact["evidence_text"],
            }
        )

        if len(held) > 1:
            st.dataframe(
                [
                    {
                        "Passage": one["passage_id"],
                        "Position": one["ordinal"],
                        "Page": page.written(one["page_from"]),
                        "Shown": one["position"],
                    }
                    for one in held
                ],
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "Extraction reads a passage and writes all of its facts "
            "together, this one included."
        )
        stage.service(client, _EXTRACTION, scope)


page.render(view)
