"""Facts view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

_EXTRACTION = stage.Queue("extraction", "Extraction", "passages", "extracted")

#: The columns the search box can look in, by the heading each one carries in
#: the table below, mapped to the name /facts takes.
_FIELDS = {
    "Statement and evidence": "both",
    "Statement": "statement",
    "Evidence": "evidence",
}

#: The four readings a passage gets, by the value stored on the fact: what
#: the reader is looking at, and what it is for.
_KINDS = {
    "atomic": (
        "Atomic",
        (
            "One claim drawn from one sentence, citing it by number. The unit "
            "a question is generated from and scored against."
        ),
    ),
    "summary": (
        "Summary",
        (
            "Two or three sentences standing in for a whole passage. Read "
            "this instead of the passage to know what it covers."
        ),
    ),
    "outline": (
        "Outline",
        (
            "The points one passage makes, one bullet each. The same reading "
            "as the summary, as a list rather than as prose."
        ),
    ),
    "bridge": (
        "Bridge",
        (
            "One claim no single passage states, drawn from a group of "
            "passages the topic model put together."
        ),
    ),
}

#: Every check a fact is put through, by the code it is rejected under: which
#: kinds face it, what it tests, and what a count above zero means. Listed
#: whether or not anything failed it, because a check missing from a list
#: cannot be told apart from a check nobody wrote.
_CHECKS = {
    "evidence_absent": (
        "Citation resolves",
        "every kind",
        ("Every sentence number a fact cites exists in the passage it was drawn from."),
        (
            "The model cited a sentence that is not there, so the claim rests on "
            "nothing and cannot be traced back to the corpus."
        ),
    ),
    "copied": (
        "Not copied",
        "atomic",
        ("The statement differs from the evidence it cites, rather than repeating it."),
        (
            "The model returned the sentence it was given instead of drawing a "
            "claim out of it. These add no information over the passage itself."
        ),
    ),
    "asserts_nothing": (
        "Asserts something",
        "atomic, summary, bridge",
        (
            "The statement carries a finite verb, so it says something rather "
            "than naming something."
        ),
        (
            "The model returned a heading, a label or a noun phrase. It cannot be "
            "true or false, so nothing can be asked about it. An outline is "
            "exempt: a bullet is written as a fragment, and a parser reads a "
            "fragment as having no verb at all."
        ),
    ),
    "not_atomic": (
        "Exactly one claim",
        "atomic, bridge",
        (
            "The statement carries a single predicate, so it can be answered by "
            "a single question."
        ),
        (
            "Statements carrying several claims. A multi-claim statement cannot "
            "be scored right or wrong as one answer. A summary and an outline "
            "are exempt: carrying several claims is what they are for."
        ),
    ),
    "unsupported_addition": (
        "Nothing invented",
        "every kind",
        (
            "Every number, name and date in the statement appears in the "
            "sentence it cites."
        ),
        (
            "The model asserted something its source does not say. These are the "
            "most damaging failures: they read like facts and are not. A rising "
            "share means the model is filling gaps rather than reading."
        ),
    ),
    "unresolved_reference": (
        "No dangling pronouns",
        "atomic, bridge",
        (
            "The statement stands alone, without a pronoun whose referent is "
            "only in the surrounding text."
        ),
        (
            "The statement cannot be read on its own, so a question built from "
            "it would be unanswerable out of context."
        ),
    ),
    "not_condensed": (
        "Shorter than its passage",
        "summary, outline",
        (
            "The digest is well under the length of the passage it stands in "
            "for, so reading it saves the reader something."
        ),
        (
            "The model returned something about as long as the passage. A "
            "summary the length of its source is of no use to anybody."
        ),
    ),
    "not_listed": (
        "Two points or more",
        "outline",
        "The outline holds at least two bullet points.",
        (
            "The model returned one line, which is a label rather than a list. "
            "This is the only shape check an outline gets: its points are "
            "fragments, so nothing can be read off their grammar."
        ),
    ),
    "not_bridging": (
        "Rests on two passages",
        "bridge",
        "The claim names at least two of the passages it was shown.",
        (
            "The model wrote a claim one passage states on its own. That is an "
            "ordinary fact, and the atomic pass has it already."
        ),
    ),
}

#: What the decomposition ratio should clear. A fact carries one claim out of
#: a sentence that usually carries several, so the ratio sits well above 1;
#: at 1 the model restated sentences rather than breaking them up.
_DECOMPOSITION_FLOOR = 1.2


def view() -> None:
    """Renders the facts page."""
    page.header(
        "Facts",
        "What extraction drew out of each passage, with the sentences it "
        "cites. Four readings of the same corpus: atomic claims, a summary "
        "and an outline of each passage, and bridges across passages about "
        "one subject. Rejected facts are listed too: the share that failed a "
        "check is how extraction is judged. Pick a fact from the table to see "
        "it in full and to read its passage again.",
    )

    client = backend.catalog_api()
    page.section(
        "Extraction across the whole corpus",
        "Every passage in the corpus, not only the ones the filters below "
        "select. Extraction queues over passages, so these count passages "
        "rather than facts. They refresh on their own every few seconds.",
    )
    stage.overview(client, [_EXTRACTION], _corpus_figures)

    st.divider()
    bar = catalog.filters(
        "facts",
        client.document_names(),
        types=["llm", "deterministic"],
        type_label="Method",
        fields=_FIELDS,
    )

    kind = _kind_picker()
    where = {
        "document": bar.document,
        "q": bar.search,
        "field": bar.field,
        "method": bar.block_type,
        "kind": kind,
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
            else "No facts yet. Extract from some passages using the Documents page."
        )
        return

    quality = client.fact_quality(**where)
    page.section(
        "Quality of the facts these filters select",
        "Measured over every fact matching the search, the document and the "
        "method chosen above - the whole filtered set, not just the page of "
        "rows below. Rejected facts are counted in every figure here, which "
        "is the point: the share that failed is how extraction is judged.",
    )
    page.metrics(_quality_figures(quality))
    page.metrics(_kind_figures(quality))
    page.metrics(_shape_figures(quality))
    page.findings(
        "Every check, and what it found",
        "A fact is stored whether or not it passed. Each row is one check, "
        "which kinds of fact face it, the number in this filtered set that "
        "failed it, and what a count above zero means for the corpus. A check "
        "no kind in view faces reads 0 because nothing was asked of it, not "
        "because everything passed.",
        _checks(quality, kind),
    )

    st.dataframe(
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
        width="stretch",
        hide_index=True,
        column_config={
            "Statement": st.column_config.TextColumn(width="large"),
            "Cited": st.column_config.TextColumn(width="medium"),
        },
    )

    st.divider()
    chosen = st.selectbox(
        "Fact to inspect",
        rows,
        format_func=lambda r: f"passage {_ordinals(r)} · {r['statement'][:70]}",
        help="Everything below this point applies to this fact and to the "
        "passage(s) it was drawn from.",
    )
    if chosen is not None:
        _detail(client, chosen)


def _ordinals(fact: dict) -> str:
    """Where each passage a fact rests on sits in its own document."""
    return ", ".join(str(one["ordinal"]) for one in fact["passages"]) or "—"


def _pages(fact: dict) -> str:
    """The pages a fact rests on, each named once."""
    found = dict.fromkeys(
        one["page_from"] for one in fact["passages"] if one["page_from"] is not None
    )
    return ", ".join(str(one) for one in found) or "—"


def _kind_picker() -> str | None:
    """Draws the kind filter and returns what the API takes, or None for all."""
    headings = {label: code for code, (label, _) in _KINDS.items()}
    chosen = st.radio(
        "Reading",
        ["All kinds", *headings],
        horizontal=True,
        key="facts-kind",
        help="Which reading of the corpus to look at. Each narrows every "
        "figure and row below to one kind:\n\n"
        + "\n\n".join(f"- **{label}** — {what}" for label, what in _KINDS.values()),
    )
    return headings.get(chosen)


def _kind_figures(quality: dict) -> dict[str, tuple]:
    """Counts each reading of the corpus in the filtered set."""
    held = quality.get("kinds", {})
    total = quality["total"]
    return {
        label: (*page.portion(held.get(code, 0), total), what)
        for code, (label, what) in _KINDS.items()
    }


def _corpus_figures(
    counts: dict[str, dict[str, int]],
) -> dict[str, tuple]:
    """Names the corpus-wide extraction figures at the top of this page."""
    passages = sum(counts["extraction"].values())
    return {
        "Passages in corpus": (
            f"{passages:,}",
            (
                "Every passage chunking has produced. This is the unit "
                "extraction queues over, so it is the size of the queue when "
                "everything has been asked for."
            ),
        ),
        "Read for facts": (
            *page.portion(counts["extraction"].get("extracted", 0), passages),
            (
                "Passages extraction has finished reading, as a share of every "
                "passage. A passage counts as read whether or not the facts it "
                "yielded passed their checks."
            ),
        ),
        "Queued": (
            f"{counts['extraction'].get('pending', 0) + counts['extraction'].get('in_progress', 0):,}",
            "Passages waiting for a worker or held by one right now.",
        ),
        "Not started": (
            f"{counts['extraction'].get('new', 0):,}",
            page.STATUS_HELP["new"],
        ),
        "Failed": (
            f"{counts['extraction'].get('failed', 0):,}",
            page.STATUS_HELP["failed"]
            + " This is the worker failing on a passage, which is not the "
            "same as a fact failing a check - that is the table below.",
        ),
    }


def _quality_figures(quality: dict) -> dict[str, tuple]:
    """Names the headline quality figures for the filtered set."""
    total = quality["total"]
    rejected = sum(quality["rejected"].values())
    return {
        "Facts matching": (
            f"{total:,}",
            (
                "Facts matching every filter above at once, rejected ones "
                "included. This is what the pager counts through."
            ),
        ),
        "Passed every check": (
            *page.portion(quality["validated"], total),
            (
                "Facts that cleared every check their kind faces: the citation "
                "resolved, nothing was asserted the source does not say, and "
                "whatever else that kind is held to. Only these are usable for "
                "question generation."
            ),
        ),
        "Rejected": (
            *page.portion(rejected, total),
            (
                "Facts that failed at least one check. They are kept rather than "
                "discarded so the failure rate can be measured; the table below "
                "says which check each failed."
            ),
        ),
        "Facts per passage": (
            f"{quality['facts_per_passage']:.1f}",
            (
                "Facts drawn from each passage that yielded any, counted over "
                "the kinds in view. With Atomic picked above, a passage "
                "usually carries several claims and a figure near 1 means most "
                "of each passage went unread."
            ),
        ),
        "Decomposition": (
            _decomposition(quality),
            (
                "How many claims the cited sentences carried for each claim the "
                "statement kept. A fact takes one claim out of a sentence that "
                f"usually carries several, so this should sit above "
                f"{_DECOMPOSITION_FLOOR}×. Near 1× means the model restated "
                "sentences instead of breaking them up."
            ),
        ),
    }


def _shape_figures(quality: dict) -> dict[str, tuple]:
    """Names the figures describing the shape of a statement."""
    statement = quality["mean_statement_chars"]
    evidence = quality["mean_evidence_chars"]
    return {
        "Statement length": (
            f"{statement:.0f}",
            "characters, mean",
            (
                "Mean length of the claim extraction wrote. A statement is meant "
                "to be shorter than the sentence it came from, because it "
                "carries one claim out of it."
            ),
        ),
        "Evidence length": (
            f"{evidence:.0f}",
            "characters, mean",
            "Mean length of the passage text a fact cites as its support.",
        ),
        "Statement vs evidence": (
            f"{statement / evidence:.0%}" if evidence else "—",
            (
                "Statement length as a share of the evidence it cites. Near 100% "
                "means the statement is as long as its source, which usually "
                "means it was copied rather than distilled."
            ),
        ),
        "Claims per statement": (
            f"{quality['mean_statement_predicates']:.2f}",
            (
                "Predicates spaCy found in the mean statement. A fact should "
                "carry exactly one, so this sits near 1.00; above it means "
                "statements are compound and cannot be scored as one answer."
            ),
        ),
        "Claims per evidence": (
            f"{quality['mean_evidence_predicates']:.2f}",
            (
                "Predicates in the mean cited sentence. This is the numerator of "
                "the decomposition ratio: it is how much there was to draw out."
            ),
        ),
    }


def _checks(quality: dict, kind: str | None) -> list[dict[str, str]]:
    """Builds one row per check, whether or not anything failed it."""
    total = quality["total"]
    rejected = quality["rejected"]
    rows = [
        {
            "Check": name,
            "Applies to": applies,
            "Failed": page.share(rejected.get(code, 0), total),
            "Should be": "0",
            "State": "OK" if not rejected.get(code) else "Attention",
            "What it means": tests if not rejected.get(code) else consequence,
        }
        for code, (name, applies, tests, consequence) in _CHECKS.items()
    ]
    rows.append(_decomposition_row(quality, kind))

    unknown = set(rejected) - set(_CHECKS)
    rows += [
        {
            "Check": code,
            "Applies to": "unknown",
            "Failed": page.share(rejected[code], total),
            "Should be": "0",
            "State": "Attention",
            "What it means": "A rejection code this page has no description "
            "for. It was added to the checks in the backend without being "
            "added here.",
        }
        for code in sorted(unknown)
    ]
    return rows


def _decomposition_row(quality: dict, kind: str | None) -> dict[str, str]:
    """Builds the decomposition row, which no single fact can fail.

    Only atomic facts are decomposed: a summary keeps every claim its passage
    carried on purpose, so measuring it over a mixed set says nothing.
    """
    if kind != "atomic":
        return {
            "Check": "Decomposition",
            "Applies to": "atomic",
            "Failed": "—",
            "Should be": f"above {_DECOMPOSITION_FLOOR}×",
            "State": "—",
            "What it means": "Only atomic facts are decomposed; a summary "
            "keeps the claims its passage carried on purpose. Pick Atomic "
            "above to measure it.",
        }

    statement = quality["mean_statement_predicates"]
    ratio = quality["mean_evidence_predicates"] / statement if statement else 0
    return {
        "Check": "Decomposition",
        "Applies to": "atomic",
        "Failed": _decomposition(quality),
        "Should be": f"above {_DECOMPOSITION_FLOOR}×",
        "State": "OK" if ratio >= _DECOMPOSITION_FLOOR else "Attention",
        "What it means": "Each statement takes one claim out of a sentence "
        "carrying several."
        if ratio >= _DECOMPOSITION_FLOOR
        else "Statements keep almost every claim their sentence carried, so "
        "the model restated rather than decomposed. Check EXTRACTION_MODEL "
        "and the prompt version.",
    }


def _decomposition(quality: dict) -> str:
    """How many claims the cited text carried for each one the statement kept."""
    statement = quality["mean_statement_predicates"]
    if not statement:
        return "—"
    return f"{quality['mean_evidence_predicates'] / statement:.1f}×"


def _detail(client, fact: dict) -> None:
    """Shows one fact in full and offers to read its passage again."""
    held = fact["passages"]
    scope = ("passage", str(held[0]["passage_id"]))
    check = _CHECKS.get(fact["rejection_code"])
    label, what = _KINDS.get(fact["kind"], (fact["kind"], "An unrecognised kind."))

    page.section(
        f"{label} fact from passage {_ordinals(fact)}"
        if len(held) == 1
        else f"{label} fact across passages {_ordinals(fact)}",
        "This fact as it is stored, and the passage(s) it came from.",
    )
    page.metrics(
        {
            "Reading": (label, what),
            "Passed every check": (
                "yes" if fact["validated"] else "no",
                (
                    "Whether this fact cleared every check its kind faces. "
                    "Only facts reading yes are usable for question "
                    "generation."
                ),
            ),
            "Failed check": (
                check[0] if check else "—",
                check[2] if check else "This fact failed no check.",
            ),
            "Claims in statement": (
                f"{fact['statement_predicates']:,}",
                (
                    "Predicates spaCy found in the statement. An atomic fact "
                    "and a bridge carry exactly one; a summary and an outline "
                    "carry as many as they need."
                ),
            ),
            "Claims in evidence": (
                f"{fact['evidence_predicates']:,}",
                (
                    "Predicates in the passage text this fact cites. More than "
                    "the statement is what decomposition looks like."
                ),
            ),
            "Method": (
                fact["extraction_method"],
                (
                    "How this fact was drawn: `llm` by the served model, "
                    "`deterministic` by a rule that needs no model."
                ),
            ),
        }
    )

    st.markdown(
        "**Statement**",
        help="What extraction wrote. This is what a question would be generated from.",
    )
    st.text(fact["statement"])
    st.markdown(
        "**Evidence cited**",
        help="The passage text this claim says it rests on. A claim asserting "
        "anything not present here failed the `Nothing invented` check. A "
        "summary and an outline cite their whole passage; a bridge cites a "
        "span in each of the passages below, one per line.",
    )
    st.text(fact["evidence_text"])

    if len(held) > 1:
        _across(held)

    if fact["validation_error"]:
        st.error(f"Rejection reason: {fact['validation_error']}")
    for label, values, explanation in (
        (
            "Added to the statement",
            fact["units_added"],
            (
                "Numbers, names or dates in the statement that the cited "
                "sentence does not contain."
            ),
        ),
        (
            "Unresolved references",
            fact["unresolved_references"],
            (
                "Pronouns in the statement whose referent is only in the "
                "surrounding text."
            ),
        ),
    ):
        if values:
            st.caption(f"{label}: {', '.join(values)}", help=explanation)

    page.section(
        "Read this passage again",
        "Queues extraction over the one passage this fact came from, "
        "replacing every fact drawn from it - this one included. A single "
        "fact cannot be re-read on its own: extraction reads a passage and "
        "writes all of its facts together. Bridge facts are not replaced by "
        "this; they are written by their own pass over the topics. " + stage.COLOUR_KEY,
    )
    stage.controls(client, _EXTRACTION, scope)


def _across(held: list[dict]) -> None:
    """Names the passages a fact resting on several was drawn from."""
    st.caption(
        "Rests on passages: "
        + ", ".join(f"{one['ordinal']} (id {one['passage_id']})" for one in held),
        help="Every passage this claim was drawn from, in the order the model "
        "was shown them, each by its position in its own document. The "
        "evidence above carries one span per passage, in this order; the "
        "controls below apply to the first.",
    )


page.render(view)
