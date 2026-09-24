"""Questions view. Runs question generation, and no other stage."""

from __future__ import annotations

import json

import streamlit as st

from lib import backend, catalog, config, configure, page, stage

#: What an .xlsx is on the wire, which the download button labels the file
#: with so a browser hands it to a spreadsheet rather than saving it blind.
_WORKBOOK = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

#: The one stage this page runs. It queues over topics, which the Topics
#: page fits; nothing here can fit them.
_QUESTIONS = stage.Queue("questions", "Question generation", "topics", "generated")

#: The columns the search box can look in, by the heading each carries in
#: the table below, mapped to the name /questions takes.
_FIELDS = {
    "Question and answer": "both",
    "Question": "question",
    "Answer": "answer",
}

#: Every gate a question is put through, by the code it is rejected under,
#: and what the gate tests. Listed whether or not anything failed it, in the
#: order the checker applies them: cheapest first.
_GATES = {
    "malformed": "The question is a question, in the language of its facts.",
    "answer_too_short": "The answer clears the floor its form is held to.",
    "answer_too_long": "The answer is within the ceiling its form is held to.",
    "wrong_form": "A value names a thing, an explanation explains one.",
    "wrong_type": "An entity question asks after a party, an enumeration a set.",
    "restates_question": "The answer adds a content word the question did not have.",
    "explanation_unusable": "The long answer reads, rests on the passages, and "
    "says more than the key.",
    "asks_nothing_new": "A follow-up reaches a fact the question before it did not.",
    "off_thread": "A follow-up stays on the material the turn before it used.",
    "leaks_source": "The question does not say which document holds the answer.",
    "off_topic": "An unanswerable question is about what the material covers.",
    "duplicate": "The question is far enough from every question accepted.",
    "answerable_after_all": "A question written to have no answer has none.",
    "compound": "The question uses one interrogative, so it asks one thing.",
    "unanchored": "Somebody who never read the passage could tell what is asked.",
    "not_recoverable": "A second model got the answer out of the cited passages.",
    "answer_incomplete": "The answer names most of what that model found there.",
    "answerable_elsewhere": "No passage it leaves uncited answers it either.",
    "source_changed": "Every fact the question rests on still passes its checks.",
}

#: The three criteria a question is classified by, as the wider value of
#: each: what a chatbot has to reach across to answer it.
_CRITERIA = {
    "passage_scope": ("Passages", "multi_passage", "Needs more than one passage."),
    "document_scope": ("Documents", "cross_document", "Needs two documents."),
    "topic_scope": ("Subjects", "multi_topic", "Bridges two subjects."),
}

#: What each difficulty band is. Five things make a question harder: the
#: three criteria, a long answer, and being a follow-up.
_DIFFICULTY = {
    "easy": "None or one of the five things that make a question harder.",
    "medium": "Two of them.",
    "hard": "Three or more of them.",
}

#: Every kind of question that can be asked, and what each asks for.
_TYPES = {
    "factoid": "One checkable value.",
    "definition": "What a named thing or status is.",
    "entity": "Who does, decides, owns or must be told something.",
    "enumeration": "Which things belong to a named set.",
    "condition": "When, or under what circumstances, something applies.",
    "reason": "Why something is required or done.",
    "procedure": "How something is done, or in what order.",
    "consequence": "What happens when something is or is not done.",
    "comparison": "How two named things differ.",
    "aggregation": "A total no single fact states.",
    "temporal": "What changed between two periods.",
    "implication": "What must be true when two stated things both hold.",
    "application": "Which stated rule governs a case the material omits.",
}

#: How much a question asks of whoever answers it, declared by its kind.
#: A different axis from the band: that one says how far the answer is
#: spread and so how hard it is to find.
_LEVELS = {
    "recall": "The answer is a value stated in one place.",
    "understand": "The answer restates what the material means.",
    "apply": "The answer maps a stated rule onto a case.",
    "analyse": "The answer is not stated anywhere and has to be worked out.",
}

#: What each answer form is, which the length bounds are applied against.
_FORMS = {
    "value": "A short noun phrase: a value, an amount, a date, a name.",
    "list": "Several short items.",
    "explanation": "One or two sentences of prose.",
}

#: What a healthy unanswerable share looks like.
_UNANSWERABLE_FLOOR = 0.05

#: The turn filter, by the value `follows` takes.
_TURNS = {"Opening": False, "Follow-up": True}


def _per(count: int, over: int) -> str:
    """One figure per another, or a dash when there is nothing to divide by."""
    return f"{count / over:.1f}" if over else "-"


def _scope(value: str | None, single: str) -> str:
    """How many a scope column says, or a dash where it says nothing.

    The columns are nullable, and read as "anything but single" a question
    that was never given one shows as spanning two.
    """
    if not value:
        return "—"
    return "1" if value == single else "2+"


def view() -> None:
    """Renders the questions page."""
    page.header("Questions")

    client = backend.catalog_api()
    overview, analysis, service = st.container(), st.container(), st.container()

    words, field = catalog.search("questions", _FIELDS)
    chosen, pager = catalog.filters(
        "questions",
        {
            "status": (
                "States",
                ["accepted", "rejected", "draft"],
                "Whether the gates, or a person, accepted it.",
            ),
            "question_type": ("Kinds", list(_TYPES), "What the question asks for."),
            "cognitive_level": (
                "Levels",
                list(_LEVELS),
                "What it asks of whoever answers it.",
            ),
            "answer_form": ("Shapes", list(_FORMS), "The shape the answer takes."),
            "difficulty": (
                "Difficulties",
                list(_DIFFICULTY),
                "The band the question turned out to be.",
            ),
            "follows": ("Turns", list(_TURNS), "Whether it follows another."),
        },
        documents=client.document_names(),
    )

    where = {
        "document": chosen["document"],
        "q": words,
        "field": field,
        "status": chosen["status"],
        "question_type": chosen["question_type"],
        "cognitive_level": chosen["cognitive_level"],
        "answer_form": chosen["answer_form"],
        "difficulty": chosen["difficulty"],
        "follows": _TURNS.get(chosen["follows"] or ""),
    }
    total, rows = catalog.paged(
        "questions",
        pager,
        lambda limit, offset: client.questions(**where, limit=limit, offset=offset),
        "questions",
    )
    quality = client.question_quality(**where)
    counts = client.stage_status("questions")["rows"]

    with overview, page.panel("Overview"):
        page.stats(
            {
                "Questions": (
                    f"{quality['total']:,}",
                    "Questions matching, rejected ones included.",
                ),
                "Accepted": (
                    page.share(quality["accepted"], quality["total"]),
                    "Cleared every gate, or a person said so.",
                ),
                "Unanswerable": (
                    page.share(quality["unanswerable"], quality["total"]),
                    "Written to have no answer in the corpus.",
                ),
                "Subjects covered": (
                    page.share(
                        quality["topics_covered"], quality["topics_in_coverage"]
                    ),
                    "Topics counted in coverage that generation has finished.",
                ),
                # One figure rather than three, because the page holds
                # five and this is one thing: how much of the material the
                # accepted questions actually reach. A topic is covered by
                # its FIRST accepted question, so the count above can read
                # 38 of 38 while a third of the corpus was never asked
                # about. The Analysis fold carries the parts.
                "Corpus reached": (
                    page.share(
                        quality["passages_asked"], quality["passages_with_facts"]
                    ),
                    (
                        "Passages an accepted question rests on, of those a "
                        "validated fact rests on. A passage no fact rests on "
                        "cannot be asked about, so it is not counted here."
                    ),
                ),
            }
        )

    with analysis, st.expander("Analysis"):
        page.findings(_analysis(quality, counts, client.question_plan()))

    with service, page.panel("Question generation"):
        stage.service(client, _QUESTIONS)
        configure.panel("questions")

    with page.panel(f"Questions · {total:,}"):
        if not rows:
            st.caption("Nothing matches.")
            return
        picked = page.table(
            [
                {
                    "State": row["status"],
                    "Failed gate": row["rejected_reason"] or "—",
                    "Kind": row["question_type"] or "—",
                    "Answer shape": row["answer_form"] or "—",
                    "Answerable": "yes" if row["answerable"] else "no",
                    "Difficulty": row["difficulty"] or "—",
                    "Passages": _scope(row["passage_scope"], "single_passage"),
                    "Documents": _scope(row["document_scope"], "single_document"),
                    "Turn": row["thread_position"],
                    "Question": row["question_text"],
                    "Answer": row["target_answer"] or "—",
                }
                for row in rows
            ],
            key="questions-table",
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Answer": st.column_config.TextColumn(width="medium"),
            },
        )
        _export(client, where, total)

    if picked is not None:
        _detail(client, rows[picked])


def _export(client, where: dict, total: int) -> None:
    """Offers the filtered questions as a workbook, built when asked for.

    Two steps rather than one, and the reason is what `st.download_button`
    does: it wants the bytes up front, so a page holding one would build a
    workbook of every question on every rerun - on each keystroke in the
    search box, for a set this page is usually showing all of. So the
    first press builds it and the second saves it.

    Kept against the filter it was built from. A workbook offered after the
    filter moved under it is the wrong set under a name that looks right,
    which is worse than no button.
    """
    asked = json.dumps(where, sort_keys=True, default=str)
    built = st.session_state.get("questions-workbook")

    if built and built[0] == asked:
        st.download_button(
            f"Download {total:,} question(s) · .xlsx",
            data=built[1],
            file_name="questions.xlsx",
            mime=_WORKBOOK,
            key="questions-download",
            type="primary",
            width="stretch",
        )
        return

    if st.button("Build a workbook", key="questions-build", width="stretch"):
        with st.spinner(f"Writing {total:,} question(s)…"):
            st.session_state["questions-workbook"] = (
                asked,
                client.question_workbook(**where),
            )
        st.rerun()
    st.caption(
        "One sheet of questions, one of the facts each cites, one of counts — "
        "everything the filter above selects, not just this page."
    )


def _analysis(quality: dict, counts: dict[str, int], plan: dict) -> list[dict]:
    """Builds the fold: the queue, the gates, the kinds and the spread."""
    total = quality["total"]
    rejected = quality["rejected"]

    rows = page.queue_rows(
        "Topics", counts, ("new", "pending", "in_progress", "generated", "failed")
    )
    rows += [
        {
            "Check": code,
            "Value": page.share(rejected.get(code, 0), total),
            "Should be": "0",
            "State": "OK" if not rejected.get(code) else "Attention",
            "What it means": tests,
        }
        for code, tests in _GATES.items()
    ]

    # Coverage, which is about the SET and not about any question in it.
    # A run can clear every gate and still have asked about a third of the
    # corpus, and nothing on the row level would say so.
    asked, askable = quality["passages_asked"], quality["passages_with_facts"]
    cited, validated = quality["facts_asked"], quality["facts_validated"]
    rows += [
        {
            "Check": "Passages asked about",
            "Value": page.share(asked, askable),
            "Should be": "as high as the material allows",
            "State": "OK" if asked else "Attention",
            "What it means": (
                f"{asked:,} of the {askable:,} passages a validated fact "
                f"rests on. The other {max(askable - asked, 0):,} are what "
                f"another run would reach first."
            ),
        },
        {
            "Check": "Facts asked about",
            "Value": page.share(cited, validated),
            "Should be": "as high as the material allows",
            "State": "OK" if cited else "Attention",
            "What it means": (
                f"{cited:,} of {validated:,} validated facts are cited by an "
                f"accepted question, at {_per(quality['accepted'], cited)} "
                f"questions per fact and "
                f"{_per(quality['accepted'], asked)} per passage."
            ),
        },
    ]

    # Not a gate: no single question fails for being the only answerable
    # one, the set does.
    share = quality["unanswerable"] / total if total else 0
    rows.append(
        {
            "Check": "Some are unanswerable",
            "Value": f"{share:.0%}",
            "Should be": f"above {_UNANSWERABLE_FLOOR:.0%}",
            "State": "OK" if share >= _UNANSWERABLE_FLOOR else "Attention",
            "What it means": "Enough questions have no answer to test whether "
            "a chatbot admits it.",
        }
    )
    rows += [
        {
            "Check": code,
            "Value": page.share(rejected[code], total),
            "Should be": "0",
            "State": "Attention",
            "What it means": "A rejection code this page has no name for.",
        }
        for code in sorted(set(rejected) - set(_GATES))
    ]

    weights = plan.get("types", {})
    asked = sum(weights.values()) or 1
    written = quality["question_type"]
    rows += [
        {
            "Check": f"Kind: {name}",
            "Value": page.share(written.get(name, 0), sum(written.values())),
            "Should be": f"{weights.get(name, 0) / asked:.0%} asked for",
            "State": "—",
            "What it means": what,
        }
        for name, what in _TYPES.items()
        if weights.get(name) or written.get(name)
    ]
    levels = quality["cognitive_level"]
    rows += [
        {
            "Check": f"Level: {name}",
            "Value": page.share(levels.get(name, 0), sum(levels.values())),
            "Should be": "—",
            "State": "—",
            "What it means": what,
        }
        for name, what in _LEVELS.items()
        if levels.get(name)
    ]
    rows += [
        {
            "Check": label,
            "Value": page.share(
                quality[column].get(wider, 0), sum(quality[column].values())
            ),
            "Should be": "—",
            "State": "—",
            "What it means": what,
        }
        for column, (label, wider, what) in _CRITERIA.items()
    ]
    rows += [
        {
            "Check": band.capitalize(),
            "Value": page.share(
                quality["difficulty"].get(band, 0), sum(quality["difficulty"].values())
            ),
            "Should be": "—",
            "State": "—",
            "What it means": what,
        }
        for band, what in _DIFFICULTY.items()
    ]
    rows.append(
        {
            "Check": "Band as planned",
            "Value": page.share(quality["planned_met"], total),
            "Should be": "—",
            "State": "—",
            "What it means": "Questions whose band is the one the plan asked for.",
        }
    )
    return rows


def _detail(client, question: dict) -> None:
    """Shows everything held about one question, and decides its fate."""
    detail = client.question(question["id"])

    with page.panel(f"Question {question['id']}"):
        page.attributes(
            {
                "Id": question["id"],
                "State": question["status"],
                "Failed gate": question["rejected_reason"],
                "Kind": question["question_type"],
                "Level": question["cognitive_level"],
                "Answer shape": question["answer_form"],
                "Answerable": question["answerable"],
                "Difficulty": question["difficulty"],
                "Planned difficulty": question["planned_difficulty"],
                "Passages": question["passage_scope"],
                "Documents": question["document_scope"],
                "Subjects": question["topic_scope"],
                "Answer characters": question["answer_chars"],
                "Language": question["language"],
                "Turn": question["thread_position"],
                "Follows": question["follows_id"],
                "Facts cited": question["facts"],
                "From documents": [one[:12] + "…" for one in question["documents"]],
                "From topics": question["topics"],
                "Written": question["created_at"],
                "Question": question["question_text"],
                "Target answer": question["target_answer"],
                "Explanation": question["answer_explanation"],
            }
        )

        st.caption("Facts cited")
        st.dataframe(
            [
                {
                    "Passage": source["ordinal"],
                    "Fact still holds": "yes" if source["validated"] else "no",
                    "Statement": source["statement"],
                    "Cited": source["evidence_text"],
                }
                for source in detail["sources"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "Statement": st.column_config.TextColumn(width="large"),
                "Cited": st.column_config.TextColumn(width="medium"),
            },
        )
        if any(not source["validated"] for source in detail["sources"]):
            st.warning("A fact this question rests on no longer passes its checks.")

        if len(detail.get("thread") or []) > 1:
            st.caption("The thread this sits in")
            st.dataframe(
                [
                    {
                        "Turn": turn["thread_position"],
                        "State": turn["status"],
                        "Question": turn["question_text"],
                        "Answer": turn["target_answer"] or "—",
                    }
                    for turn in detail["thread"]
                ],
                width="stretch",
                hide_index=True,
                column_config={"Question": st.column_config.TextColumn(width="large")},
            )

        _trace(question)
        _prompt(client, question)
        _verdict(client, question)


def _trace(question: dict) -> None:
    """Links to the calls that produced this question, and their verdicts.

    Two links, because they answer different questions. The **span** is
    the gate decision, and it is what the per-gate annotations hang off,
    so it opens on why this question was accepted or refused. The
    **trace** is the whole topic: the writer call with its prompt, the
    phrasing judgements, the verifier.

    `/redirects/` rather than a project URL. Phoenix resolves either id
    from the hex alone, so nothing here has to know its internal project
    ids or survive it renumbering them.
    """
    trace_id, span_id = question.get("trace_id"), question.get("span_id")
    if not trace_id and not span_id:
        st.caption(
            "Written before the trace was recorded on the row, so the calls "
            "that produced it cannot be found from here."
        )
        return

    if not config.PHOENIX_BASE_URL:
        st.caption(f"Trace `{trace_id}`, span `{span_id}` — Phoenix is not linked.")
        return

    links = [
        f"[{label}]({config.PHOENIX_BASE_URL}/redirects/{kind}/{one})"
        for label, kind, one in (
            ("the gate decision", "spans", span_id),
            ("the whole topic's trace", "traces", trace_id),
        )
        if one
    ]
    st.caption("In Phoenix: " + ", and ".join(links) + ".")


def _prompt(client, question: dict) -> None:
    """Shows the prompt that wrote this question, as it was sent.

    The question a person asks while reading a bad question is whether the
    prompt caused it, and the answer used to be a git checkout: the row
    carries a version and the text lived in the source at whatever commit
    that version was current. It is recorded now, so a version resolves
    here.

    Folded away, because it is several thousand characters and it is not
    what somebody came to this panel for.
    """
    version, kind = question.get("prompt_version"), question.get("question_type")
    if not version:
        st.caption(
            "Written before the prompt version was recorded, so which prompt "
            "wrote it is not known."
        )
        return

    try:
        found = client.prompts(service="questions", version=version, name=kind)
    except Exception:  # noqa: BLE001 - the page says so and renders the rest
        st.caption("The prompts could not be read.")
        return

    if not found:
        st.caption(
            f"Version {version} is on this row and no prompt is recorded under "
            f"it. A stage records its prompts when it starts, so a version "
            f"nothing has run since is one nothing has written down."
        )
        return

    written = found[0]
    with st.expander(f"The prompt that wrote this · {kind} v{version}"):
        st.caption(
            f"`{written['digest']}`, first recorded "
            f"{written['first_seen_at'][:10]}. Read-only: a prompt is changed "
            f"in the source, and what is here is the record of what was sent."
        )
        st.code(written["text"], language="text", wrap_lines=True)


def _verdict(client, question: dict) -> None:
    """Draws the two controls that decide one question's fate."""
    accepted = question["status"] == "accepted"
    rejected = question["status"] == "rejected"
    accept, reject, *_ = st.columns([1.2, 1.2, 4.0], vertical_alignment="center")

    if accept.button(
        "Accept",
        key=f"accept-{question['id']}",
        type="primary",
        disabled=accepted,
        width="stretch",
        help="Puts this question in the benchmark and clears the gate that rejected it."
        if not accepted
        else "Already accepted.",
    ):
        client.decide_question(question["id"], "accepted")
        st.toast(f"Question {question['id']} accepted.")
        st.rerun()

    if reject.button(
        "Reject",
        key=f"reject-{question['id']}",
        disabled=rejected,
        width="stretch",
        help="Takes it out of the benchmark and keeps the row."
        if not rejected
        else "Already rejected.",
    ):
        client.decide_question(question["id"], "rejected")
        st.toast(f"Question {question['id']} rejected.")
        st.rerun()


page.render(view)
