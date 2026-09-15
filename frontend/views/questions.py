"""Questions view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

_QUESTIONS = stage.Queue("questions", "Question generation", "topics", "generated")

#: The columns the search box can look in, by the heading each one carries in
#: the table below, mapped to the name /questions takes.
_FIELDS = {
    "Question and answer": "both",
    "Question": "question",
    "Answer": "answer",
}

#: Every gate a question is put through, by the code it is rejected under:
#: what the gate tests, and what a count above zero means. Listed whether or
#: not anything failed it, because a gate missing from a list cannot be told
#: apart from a gate nobody wrote.
_GATES = {
    "malformed": (
        "Well formed",
        (
            "The question is a question, in the language of its facts, with a "
            "target answer when it claims to have one."
        ),
        (
            "The model returned something that is not a usable question: no "
            "question mark, no target answer, the wrong language, or the fact "
            "handed straight back. Cheap to detect and cheap to fix in the "
            "prompt."
        ),
    ),
    "leaks_source": (
        "Does not name its source",
        (
            "The question asks what it wants to know without saying which "
            "document, report or section holds the answer."
        ),
        (
            "Questions that cite their own source: `Laut den 'Risiken im Fokus "
            "2026', ...`, `Gemäß der MaRisk, ...`. Nobody asks a service desk "
            "a question while telling it which file to open, and a question "
            "carrying its own source has already done the retrieving it was "
            "written to measure. A high share means the writer is treating the "
            "passage as something to cite rather than as context."
        ),
    ),
    "restates_fact": (
        "Asks more than its fact",
        (
            "The question names something its fact does not - the institution, "
            "the document, the period - so a person searching could have asked "
            "it."
        ),
        (
            "The question is its own fact with one part replaced by a question "
            "word: `Geopolitische Konflikte schüren Unsicherheit` asked as `Was "
            "schüren geopolitische Konflikte?`. Nobody searching a corpus types "
            "that. A high share means the writer is ignoring its prompt - check "
            "QUESTIONS_FACT_SAMPLE and the prompt version."
        ),
    ),
    "unanchored": (
        "Names a subject",
        ("A reader who has never seen the passage can tell what is being asked about."),
        (
            "Questions that name nothing at all: `What specific components are "
            "included?`, or `the requirements` without saying which. The "
            "opposite failure to naming a source, and the two are easy to "
            "confuse - a question is supposed to name its subject and supposed "
            "not to name its document. The verifier judges this on the call it "
            "was already making, so it costs nothing, but it is a model's "
            "opinion and not a measurement."
        ),
    ),
    "answer_too_short": (
        "Answer worth scoring",
        (
            "The target answer is long enough to mark a chatbot right or wrong "
            "against, as QUESTIONS_MIN_ANSWER_CHARS defines long enough."
        ),
        (
            "Answers too thin to score: `7`, `8%`, `Nein`. A blunt measure, and "
            "the cost is real - at 15 characters it also refuses `70%` and "
            "`2025`, which are the most unambiguously scoreable answers there "
            "are. Lower QUESTIONS_MIN_ANSWER_CHARS to keep bare values."
        ),
    ),
    "duplicate": (
        "Not already asked",
        (
            "The question is far enough from every question already accepted "
            "to be worth asking separately."
        ),
        (
            "Near-copies of questions already in the set. A benchmark that "
            "asks the same thing twice weights that thing twice. A high share "
            "means the topic has fewer distinct facts than "
            "QUESTIONS_PER_TOPIC asks for."
        ),
    ),
    "not_recoverable": (
        "Answer is recoverable",
        (
            "A second model, shown only the cited passages, got the target "
            "answer back out of them."
        ),
        (
            "The answer cannot be found in the evidence the question cites, so "
            "no chatbot could be expected to find it either. These are the "
            "most damaging failures: the question looks answerable and is "
            "not, and marking a chatbot wrong on one is marking it wrong for "
            "being right."
        ),
    ),
    "answerable_after_all": (
        "Unanswerable really is",
        (
            "A question written to have no answer does not turn out to have "
            "one in the corpus."
        ),
        (
            "Questions meant to test whether a chatbot admits ignorance, which "
            "the corpus answers anyway. Scoring a chatbot down for answering "
            "one is scoring it down for reading."
        ),
    ),
    "source_changed": (
        "Evidence still holds",
        (
            "Every fact the question rests on still passes its own checks, and "
            "is still spread the way the question's difficulty says."
        ),
        (
            "The facts moved after the question was written - re-judged and "
            "now rejected, or a citation deleted by a re-extraction. Found by "
            "`make questions-reverify`, which calls no model."
        ),
    ),
}

#: The three criteria a question is classified by, and what each value means.
#: Each says something different about what a chatbot has to do, and each is
#: read off the facts the question cites rather than judged.
_CRITERIA = {
    "passage_scope": (
        "Passages",
        {
            "single_passage": ("One", "Answerable from a single passage."),
            "multi_passage": ("Two or more", "Needs more than one passage."),
        },
    ),
    "document_scope": (
        "Documents",
        {
            "single_document": ("One", "Answerable within a single document."),
            "cross_document": (
                "Two or more",
                (
                    "Needs passages from two different documents. The scope a "
                    "retriever cannot fake: no one chunk holds the answer."
                ),
            ),
        },
    ),
    "topic_scope": (
        "Subjects",
        {
            "single_topic": ("One", "Stays inside one subject."),
            "multi_topic": (
                "Two or more",
                (
                    "Bridges two subjects. Available because a passage usually "
                    "sits in several topics above the weight floor, and the corpus "
                    "itself is what says the two meet."
                ),
            ),
        },
    ),
}

#: What each difficulty band means. Derived from the three criteria above, a
#: long answer and being a follow-up: each is worth a point and the band is
#: the total, so nobody has to agree with a judgement to use it.
_DIFFICULTY = {
    "easy": "None or one of the five things that make a question harder.",
    "medium": "Two or three of them.",
    "hard": "Four or five of them.",
}

#: What a healthy unanswerable share looks like. Too few and nothing tests
#: whether a chatbot admits ignorance; too many and the benchmark is mostly
#: about refusing.
_UNANSWERABLE_FLOOR = 0.05


def view() -> None:
    """Renders the questions page."""
    page.header(
        "Questions",
        "The test questions written from the verified facts, with the answer "
        "each is scored against. Questions a gate rejected are listed too: the "
        "share that failed is how generation is judged. Pick one from the "
        "table to read the facts behind it and to accept or reject it.",
    )

    client = backend.catalog_api()
    page.section(
        "Generation across the whole corpus",
        "Every topic in the corpus, not only the ones the filters below "
        "select. Generation queues over topics, so these count topics rather "
        "than questions. They refresh on their own every few seconds.",
    )
    stage.overview(client, [_QUESTIONS], _corpus_figures)
    _queue_controls(client)

    st.divider()
    bar = catalog.filters(
        "questions",
        client.document_names(),
        types=["accepted", "rejected", "draft"],
        type_label="State",
        fields=_FIELDS,
    )

    where = {
        "document": bar.document,
        "q": bar.search,
        "field": bar.field,
        "status": bar.block_type,
    }
    total, rows = catalog.paged(
        "questions",
        bar.page,
        lambda limit, offset: client.questions(**where, limit=limit, offset=offset),
        "questions",
    )
    if not total:
        st.info(
            f"No questions whose {catalog.field_name(_FIELDS, bar.field)} matches."
            if bar.search
            else "No questions match."
            if bar.document or bar.block_type
            else "No questions yet. Start generation above, once the topics "
            "are fitted and the facts extracted."
        )
        return

    quality = client.question_quality(**where)
    page.section(
        "Quality of the questions these filters select",
        "Measured over every question matching the search, the document and "
        "the state chosen above - the whole filtered set, not just the page of "
        "rows below. Rejected questions are counted in every figure here, "
        "which is the point: the share that failed is how generation is "
        "judged.",
    )
    page.metrics(_quality_figures(quality))
    page.metrics(_criteria_figures(quality))
    page.metrics(_difficulty_figures(quality))
    page.findings(
        "Every gate, and what it found",
        "A question is stored whether or not it passed. Each row is one gate, "
        "the number of questions in this filtered set it rejected, and what a "
        "count above zero means for the benchmark. A row reading OK rejected "
        "nothing.",
        _gates(quality),
    )

    st.dataframe(
        [
            {
                "State": row["status"],
                "Failed gate": _GATES.get(row["rejected_reason"], ("—",))[0]
                if row["rejected_reason"]
                else "—",
                "Answerable": "yes" if row["answerable"] else "no",
                "Difficulty": row["difficulty"] or "—",
                "Passages": "1" if row["passage_scope"] == "single_passage" else "2+",
                "Documents": "1"
                if row["document_scope"] == "single_document"
                else "2+",
                "Subjects": "1" if row["topic_scope"] == "single_topic" else "2+",
                "Turn": row["thread_position"],
                "Question": row["question_text"],
                "Answer": row["target_answer"] or "—",
                "Cites": row["facts"],
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Question": st.column_config.TextColumn(width="large"),
            "Answer": st.column_config.TextColumn(width="medium"),
        },
    )

    st.divider()
    chosen = st.selectbox(
        "Question to inspect",
        rows,
        format_func=lambda r: f"{r['status']} · {r['question_text'][:70]}",
        help="Everything below this point applies to this question and to the "
        "facts it was written from.",
    )
    if chosen is not None:
        _detail(client, chosen)


def _corpus_figures(counts: dict[str, dict[str, int]]) -> dict[str, tuple]:
    """Names the corpus-wide generation figures at the top of this page."""
    rows = counts["questions"]
    topics = sum(rows.values())
    return {
        "Topics in corpus": (
            f"{topics:,}",
            (
                "Every topic the model fitted. This is the unit generation "
                "queues over, so it is the size of the queue when everything "
                "has been asked for. Fitting the topics again returns all of "
                "them to `new`."
            ),
        ),
        "Questions written": (
            *page.portion(rows.get("generated", 0), topics),
            (
                "Topics generation has finished, as a share of every topic. A "
                "topic counts as finished whether or not the questions it "
                "yielded passed their gates, and a topic taken out of coverage "
                "finishes with nothing written."
            ),
        ),
        "Queued": (
            f"{rows.get('pending', 0) + rows.get('in_progress', 0):,}",
            "Topics waiting for a worker or held by one right now.",
        ),
        "Not started": (f"{rows.get('new', 0):,}", page.STATUS_HELP["new"]),
        "Failed": (
            f"{rows.get('failed', 0):,}",
            page.STATUS_HELP["failed"]
            + " This is the worker failing on a topic, which is not the same "
            "as a question failing a gate - that is the table below.",
        ),
    }


def _quality_figures(quality: dict) -> dict[str, tuple]:
    """Names the headline quality figures for the filtered set."""
    total = quality["total"]
    rejected = sum(quality["rejected"].values())
    return {
        "Questions matching": (
            f"{total:,}",
            (
                "Questions matching every filter above at once, rejected ones "
                "included. This is what the pager counts through."
            ),
        ),
        "Passed every gate": (
            *page.portion(quality["accepted"], total),
            (
                "Questions that are well formed, that no earlier question "
                "already asks, and whose answer a second model got back out of "
                "the cited passages. Only these belong in the benchmark."
            ),
        ),
        "Rejected": (
            *page.portion(rejected, total),
            (
                "Questions that failed at least one gate. They are kept rather "
                "than discarded so the failure rate can be measured; the table "
                "below says which gate each failed."
            ),
        ),
        "Unanswerable": (
            *page.portion(quality["unanswerable"], total),
            (
                "Questions written to have no answer in the corpus, which test "
                "whether a chatbot admits ignorance instead of inventing "
                "something. QUESTIONS_UNANSWERABLE_SHARE sets how many are "
                "attempted; this is how many survived their gates."
            ),
        ),
        "Subjects covered": (
            *page.portion(quality["topics_covered"], quality["topics_in_coverage"]),
            (
                "Topics counted in coverage that generation has finished. A "
                "topic taken out of coverage on the Topics page is in neither "
                "number, which is how a topic that is not a subject stops "
                "being counted as a gap."
            ),
        ),
    }


def _criteria_figures(quality: dict) -> dict[str, tuple]:
    """Names the three criteria, each as the share that is more than one."""
    figures = {}
    for column, (label, values) in _CRITERIA.items():
        counts = quality[column]
        total = sum(counts.values())
        wider = next(name for name in values if name.startswith(("multi", "cross")))
        figures[label] = (
            *page.portion(counts.get(wider, 0), total),
            values[wider][1] + " This is the share of the set that needs that.",
        )
    return {
        **figures,
        "Follow-ups": (
            *page.portion(quality["followups"], quality["total"]),
            (
                "Questions asked after another in a thread. These may lean on "
                "the conversation - `And for urgent requests?` - so a chatbot "
                "answering one has to carry the thread. The phrasing gate is "
                "not applied to them, because not standing alone is the point."
            ),
        ),
        "Answer length": (
            f"{quality['mean_answer_chars']:.0f}",
            "characters, mean",
            (
                "Mean length of the target answers. Above "
                "QUESTIONS_LONG_ANSWER_CHARS an answer counts towards the "
                "difficulty band: a chatbot has to produce more of the right "
                "thing and a grader has more to disagree about."
            ),
        ),
    }


def _difficulty_figures(quality: dict) -> dict[str, tuple]:
    """Names the difficulty bands, and the two figures that read beside them."""
    bands = quality["difficulty"]
    total = sum(bands.values())
    figures = {
        band.capitalize(): (
            *page.portion(bands.get(band, 0), total),
            explanation
            + " Derived from the three criteria above, a long answer and being "
            "a follow-up; each is worth a point and the band is the total.",
        )
        for band, explanation in _DIFFICULTY.items()
    }
    return {
        **figures,
        "Question length": (
            f"{quality['mean_question_chars']:.0f}",
            "characters, mean",
            (
                "Mean length of the questions in this set. A long mean usually "
                "means the model is restating its facts rather than asking "
                "about them."
            ),
        ),
        "Not yet judged": (
            f"{quality['draft']:,}",
            (
                "Questions stored without a verdict. The gates accept or "
                "reject every question they see, so this should be 0; anything "
                "here was written by something that did not run them."
            ),
        ),
    }


def _gates(quality: dict) -> list[dict[str, str]]:
    """Builds one row per gate, whether or not anything failed it."""
    total = quality["total"]
    rejected = quality["rejected"]
    rows = [
        {
            "Check": name,
            "Failed": page.share(rejected.get(code, 0), total),
            "Should be": "0",
            "State": "OK" if not rejected.get(code) else "Attention",
            "What it means": tests if not rejected.get(code) else consequence,
        }
        for code, (name, tests, consequence) in _GATES.items()
    ]

    # Not a gate: no single question fails for being the only answerable one,
    # the set does. A benchmark with no unanswerable questions measures
    # nothing about a chatbot's willingness to say it does not know.
    share = quality["unanswerable"] / total if total else 0
    rows.append(
        {
            "Check": "Some are unanswerable",
            "Failed": f"{share:.0%}",
            "Should be": f"above {_UNANSWERABLE_FLOOR:.0%}",
            "State": "OK" if share >= _UNANSWERABLE_FLOOR else "Attention",
            "What it means": "Enough questions have no answer in the corpus to "
            "test whether a chatbot admits it."
            if share >= _UNANSWERABLE_FLOOR
            else "Almost nothing here tests whether a chatbot says it does not "
            "know. Raise QUESTIONS_UNANSWERABLE_SHARE, or check whether the "
            "unanswerable ones are being rejected as answerable after all.",
        }
    )

    unknown = set(rejected) - set(_GATES)
    rows += [
        {
            "Check": code,
            "Failed": page.share(rejected[code], total),
            "Should be": "0",
            "State": "Attention",
            "What it means": "A rejection code this page has no description "
            "for. It was added to the gates in the backend without being "
            "added here.",
        }
        for code in sorted(unknown)
    ]
    return rows


def _queue_controls(client) -> None:
    """Draws the four queue verbs, which act on every topic at once."""
    counts = client.stage_status("questions")["rows"]
    waiting = counts.get("new", 0)
    queued = counts.get("pending", 0)
    failed = counts.get("failed", 0)
    total = sum(counts.values())

    page.section(
        "Write the questions",
        "These four act on every topic at once. " + stage.COLOUR_KEY,
    )
    st.caption(
        "A question's subject is a topic, so generation queues over topics "
        "rather than over documents. It writes only about the facts no "
        "accepted question already rests on, which is what makes Redo cheap: "
        "after the topics are fitted again every topic reads `new`, and a "
        "second run writes only what the first did not."
    )
    stage.corpus_controls(
        client,
        "questions",
        {
            "start": (
                "Start",
                bool(waiting),
                f"Queue the {waiting:,} topic(s) generation has never been "
                "asked to do. A worker picks them up on its next poll."
                if waiting
                else "Every topic has already been asked for. Use Redo to "
                "write again for topics that are finished.",
            ),
            "rerun": (
                "Redo",
                bool(total),
                "Queue every topic again, finished ones included. Facts an "
                "accepted question already rests on are skipped, so this tops "
                "up rather than starting over."
                if total
                else "There are no topics yet. Fit them on the Topics page.",
            ),
            "stop": (
                "Stop",
                bool(queued),
                f"Take the {queued:,} queued topic(s) back off the queue. "
                "Whatever a worker holds right now still finishes."
                if queued
                else "Nothing is queued.",
            ),
            "retry": (
                "Retry",
                bool(failed),
                f"Queue the {failed:,} failed topic(s) again, clearing the "
                "error recorded against each."
                if failed
                else "Nothing has failed.",
            ),
        },
    )


def _detail(client, question: dict) -> None:
    """Shows one question in full, its facts, and the accept/reject controls."""
    detail = client.question(question["id"])
    gate = _GATES.get(question["rejected_reason"])

    page.section(
        "The question, and what it rests on",
        "This question as it is stored, and every fact it was written from.",
    )
    page.metrics(
        {
            "State": (
                question["status"],
                (
                    "`accepted` cleared every gate, `rejected` did not, and "
                    "`draft` has not been judged. A person can overrule the "
                    "gates with the controls below."
                ),
            ),
            "Failed gate": (
                gate[0] if gate else "—",
                gate[1] if gate else "This question failed no gate.",
            ),
            "Answerable": (
                "yes" if question["answerable"] else "no",
                (
                    "`no` means the question was written to have no answer in "
                    "the corpus, and a chatbot is scored on saying so rather "
                    "than on what it answers."
                ),
            ),
            "Difficulty": (
                question["difficulty"] or "—",
                _DIFFICULTY.get(
                    question["difficulty"] or "",
                    "Read off how far the facts behind this question are spread.",
                ),
            ),
            "Language": (
                question["language"],
                (
                    "The language the question is written in, which is the "
                    "language of the facts it came from."
                ),
            ),
        }
    )

    st.markdown("**Question**", help="What would be put to the chatbot, word for word.")
    st.text(question["question_text"])
    st.markdown(
        "**Target answer**",
        help="What the chatbot's answer is scored against. Absent for an "
        "unanswerable question, which is scored on whether the chatbot "
        "recognises it has no answer.",
    )
    st.text(question["target_answer"] or "— none; this question has no answer")

    if question["documents"]:
        st.caption(
            "Documents: " + ", ".join(one[:12] + "…" for one in question["documents"]),
            help="The documents the cited facts came from. Two of them is what "
            "makes this a cross-document question.",
        )
    if question["topics"]:
        st.caption(
            "Topics: " + ", ".join(question["topics"]),
            help="The subject each cited passage counts towards, which is how "
            "coverage is measured.",
        )

    page.section(
        "Facts cited",
        "Every fact this question was written from. The verifier was shown "
        "these passages and nothing else, which is what the recoverability "
        "gate means by an answer being in the corpus.",
    )
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
        st.warning(
            "A fact this question rests on no longer passes its own checks. "
            "`make questions-reverify` rejects every question in that state."
        )

    if len(detail.get("thread") or []) > 1:
        page.section(
            "The conversation this sits in",
            "A follow-up may lean on what was asked before it, so it is scored "
            "with the thread replayed rather than on its own. The phrasing gate "
            "is not applied to one for the same reason.",
        )
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
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
            },
        )

    _verdict(client, question)


def _verdict(client, question: dict) -> None:
    """Draws the two controls that decide one question's fate."""
    page.section(
        "Accept or reject this question",
        "Overrules the gates for this one question. Nothing is deleted either "
        "way: a rejected question keeps its row, because the share that was "
        "thrown away is the evidence behind the coverage report. " + stage.COLOUR_KEY,
    )
    accepted = question["status"] == "accepted"
    accept, reject, _ = st.columns(stage.VERDICT, vertical_alignment="center")

    if accept.button(
        "Accept",
        key=stage.key_for("accept", "question", str(question["id"])),
        disabled=accepted,
        width="stretch",
        help="Puts this question in the benchmark and clears whichever gate "
        "rejected it."
        if not accepted
        else "This question is already accepted.",
    ):
        client.decide_question(question["id"], "accepted")
        st.toast(f"Question {question['id']} accepted.")
        st.rerun()

    if reject.button(
        "Reject",
        key=stage.key_for("reject", "question", str(question["id"])),
        disabled=question["status"] == "rejected",
        width="stretch",
        help="Takes this question out of the benchmark and keeps the row, so "
        "it still counts towards the drop rate."
        if question["status"] != "rejected"
        else "This question is already rejected.",
    ):
        client.decide_question(question["id"], "rejected")
        st.toast(f"Question {question['id']} rejected.")
        st.rerun()


page.render(view)
