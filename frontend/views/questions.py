"""Questions view. Runs question generation, and no other stage."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

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
#: and what the gate tests. Listed whether or not anything failed it.
_GATES = {
    "malformed": "The question is a question, in the language of its facts.",
    "answer_too_short": "The answer clears the floor its form is held to.",
    "answer_too_long": "The answer is within the ceiling its form is held to.",
    "wrong_form": "A value names a thing, an explanation explains one.",
    "wrong_type": "The question asks for the kind of thing its type asks for.",
    "unanchored": "Somebody who never read the passage could tell what is asked.",
    "leaks_source": "The question does not say which document holds the answer.",
    "duplicate": "The question is far enough from every question accepted.",
    "not_recoverable": "A second model got the answer out of the cited passages.",
    "answerable_after_all": "A question written to have no answer has none.",
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
    "medium": "Two or three of them.",
    "hard": "Four or five of them.",
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
            }
        )

    with analysis, st.expander("Analysis"):
        page.findings(_analysis(quality, counts, client.question_plan()))

    with service, page.panel("Question generation"):
        stage.service(client, _QUESTIONS)

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
                    "Passages": "1"
                    if row["passage_scope"] == "single_passage"
                    else "2+",
                    "Documents": "1"
                    if row["document_scope"] == "single_document"
                    else "2+",
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

    if picked is not None:
        _detail(client, rows[picked])


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

        _verdict(client, question)


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
