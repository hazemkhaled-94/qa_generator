"""Facts view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

#: The columns the search box can look in, by the heading each one carries in
#: the table below, mapped to the name /facts takes.
_FIELDS = {
    "Statement and evidence": "both",
    "Statement": "statement",
    "Evidence": "evidence",
}

#: What each rejection code means, in the words the page uses.
_REASONS = {
    "evidence_absent": "cited a sentence the passage does not have",
    "copied": "repeated its evidence instead of drawing a claim out of it",
    "not_atomic": "carried more or fewer than one claim",
    "unsupported_addition": "asserted something the cited text does not say",
    "unresolved_reference": "left a pronoun the reader cannot resolve",
}


def view() -> None:
    """Renders the facts page."""
    page.header(
        "Facts",
        "What extraction drew out of each passage, with the sentences it "
        "cites. Rejected facts are listed too: the share that failed a check "
        "is how extraction is judged.",
    )
    stage.panel("extraction", "passages", "extracted")

    client = backend.catalog_api()
    bar = catalog.filters(
        "facts",
        client.document_names(),
        types=["llm", "deterministic"],
        type_label="Method",
        fields=_FIELDS,
    )

    where = {
        "document": bar.document,
        "q": bar.search,
        "field": bar.field,
        "method": bar.block_type,
    }
    total, rows = catalog.paged(
        "facts",
        bar.page,
        lambda limit, offset: client.facts(**where, limit=limit, offset=offset),
        "facts",
    )
    if not total:
        st.info(
            f"No facts whose {catalog.field_name(_FIELDS, bar.field)} matches."
            if bar.search
            else "No facts match."
            if bar.document or bar.block_type
            else "No facts yet. Extract from some passages using the strip above."
        )
        return

    quality = client.fact_quality(**where)
    page.metrics(
        {
            "Facts matching": f"{quality['total']:,}",
            "Every check passed": _share(quality["validated"], quality["total"]),
            "Claims per statement": f"{quality['mean_statement_predicates']:.2f}",
            "Decomposition": _decomposition(quality),
            "Statement / evidence": f"{quality['mean_statement_chars']:.0f}"
            f" / {quality['mean_evidence_chars']:.0f} chars",
        }
    )
    _quality(quality)

    st.dataframe(
        [
            {
                "Passage": row["ordinal"],
                "Page": row["page_from"] if row["page_from"] is not None else "—",
                "Method": row["extraction_method"],
                "Passed": "yes" if row["validated"] else "no",
                "Statement": row["statement"],
                "Cited": row["evidence_text"],
                "Why rejected": row["validation_error"] or "",
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Statement": st.column_config.TextColumn(width="large"),
            "Cited": st.column_config.TextColumn(width="medium"),
        },
    )


def _share(part: int, whole: int) -> str:
    """Renders a count as itself and its share of a whole."""
    if not whole:
        return f"{part:,}"
    return f"{part:,} ({part / whole:.0%})"


def _decomposition(quality: dict) -> str:
    """How many claims the cited text carried for each one the statement kept."""
    statement = quality["mean_statement_predicates"]
    if not statement:
        return "—"
    return f"{quality['mean_evidence_predicates'] / statement:.1f}×"


def _quality(quality: dict) -> None:
    """Says what is wrong with the facts, when something is.

    Two failures look like success: a statement carrying every claim its
    sentence carried was restated rather than decomposed, and a statement
    asserting more than its source does reads like a fact and is not one.
    """
    total = quality["total"]
    if not total:
        return

    statement = quality["mean_statement_predicates"]
    evidence = quality["mean_evidence_predicates"]
    if statement and evidence / statement < 1.2:
        st.warning(
            f"Each statement keeps {evidence / statement:.1f} of the claims its "
            "cited sentences carry. A fact carries one claim out of a sentence "
            "that usually carries several, so this should sit well above 1. "
            "Near 1 means the model restated sentences instead of breaking "
            "them up — check EXTRACTION_MODEL and the prompt version."
        )

    invented = quality["rejected"].get("unsupported_addition", 0)
    if invented:
        st.warning(
            f"{invented:,} of {total:,} statements asserted a number, name or "
            "date the sentence they cite does not contain. Those are not facts "
            "about the corpus. A rising share means the model is filling gaps "
            "rather than reading."
        )

    if quality["rejected"]:
        rejected = sum(quality["rejected"].values())
        with st.expander(f"Why {rejected:,} were rejected"):
            st.dataframe(
                [
                    {
                        "Check": code,
                        "Meaning": _REASONS.get(code, ""),
                        "Facts": count,
                    }
                    for code, count in quality["rejected"].items()
                ],
                width="stretch",
                hide_index=True,
                column_config={"Meaning": st.column_config.TextColumn(width="large")},
            )


page.render(view)
