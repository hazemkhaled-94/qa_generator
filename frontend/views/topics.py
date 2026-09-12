"""Topics view."""

from __future__ import annotations

import streamlit as st

from lib import backend, page, stage


def view() -> None:
    """Renders the topics page."""
    page.header(
        "Topics",
        "Clusters over the corpus's own vocabulary — one "
        "model per language, because a single model across both spends topics "
        "on telling the languages apart. A fit is all-or-nothing: a topic "
        "cannot be rediscovered on its own. Refit after adding documents.",
    )

    client = backend.catalog_api()
    # The same strip as every other stage, with two differences it passes in:
    # Start asks for a fit rather than queueing rows, and there is nothing to
    # redo because a refit and a first fit are the same request.
    stage.panel(
        "topics",
        "fits",
        "modelled",
        start=lambda api: api.discover_topics(),
        rerun=False,
    )

    fit = client.topic_fit()
    _staleness(fit)

    topics = client.topics()
    if topics:
        page.metrics(
            {
                "Topics": f"{len(topics):,}",
                "Memberships": f"{sum(t['passages'] for t in topics):,}",
                "Passages placed": f"{sum(t['dominant_passages'] for t in topics):,}",
                "Counted in coverage": f"{sum(1 for t in topics if t['include_in_coverage']):,}",
            }
        )
        st.dataframe(
            [
                {
                    "Lang": topic["language"] or "—",
                    "#": topic["topic_index"],
                    "Label": topic["label"] or "—",
                    "Top terms": ", ".join(topic["top_terms"]),
                    "Passages": topic["passages"],
                    "Dominant in": topic["dominant_passages"],
                    "Documents": topic["documents"],
                    "Mean weight": f"{topic['mean_weight']:.2f}",
                    "In coverage": "yes" if topic["include_in_coverage"] else "no",
                }
                for topic in topics
            ],
            width="stretch",
            hide_index=True,
            column_config={"Top terms": st.column_config.TextColumn(width="large")},
        )
        _naming(client, topics)
        _provenance(fit)
    else:
        st.info("No topics yet. Chunk some documents, then press **Start** above.")

    st.divider()
    _removal(client, bool(topics))


def _naming(client, topics: list[dict]) -> None:
    """Lets a person name one topic and say whether it counts for coverage.

    Outside the polling strip: a fragment redrawing every few seconds would
    take the box away mid-sentence.
    """
    st.divider()
    st.html("<div class='qa-section'>Name a topic</div>")
    st.caption(
        "A name survives a refit by being matched on top terms, so a topic "
        "whose terms move too far loses it. Nothing else stores one."
    )

    chosen = st.selectbox(
        "Topic",
        topics,
        format_func=lambda t: (
            f"{t['language']} #{t['topic_index']}: {', '.join(t['top_terms'][:6])}"
        ),
        key="topics-chosen",
    )
    name, coverage, apply, *_ = st.columns([3.2, 1.6, 1.2, 1.4])
    label = name.text_input(
        "Label",
        value=chosen["label"] or "",
        key=f"topics-label-{chosen['id']}",
        placeholder="a short name for this subject",
    )
    included = coverage.checkbox(
        "In coverage",
        value=chosen["include_in_coverage"],
        key=f"topics-coverage-{chosen['id']}",
        help="Clear this for a topic too diffuse to be a meaningful coverage "
        "partition. It keeps its passages either way.",
    )
    apply.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if apply.button("Save", key="topics-save", type="primary", width="stretch"):
        client.describe_topic(chosen["id"], label.strip() or None, included)
        st.toast(f"Topic {chosen['language']} #{chosen['topic_index']} saved.")
        st.rerun()


def _removal(client, has_topics: bool) -> None:
    """Offers the deletion, behind a confirmation.

    Drawn whether or not there are topics: with none, it is what clears a
    fit request that failed.

    Outside the polling fragment on purpose: a fragment that redraws every
    few seconds would take the confirmation away mid-decision.
    """
    remove, *_ = st.columns(stage.CONTROLS)
    if remove.button(
        "Delete topics",
        key="topics-delete",
        width="stretch",
        help="Remove every topic and membership, and any fit still queued.",
    ):
        st.session_state["topics-armed"] = True
        st.rerun()

    if not st.session_state.get("topics-armed"):
        return

    st.error(
        "Delete every topic and every passage-topic membership? Passages, facts "
        "and questions all stay, each holding one fewer topic. Any label you "
        "assigned goes with the topic it was on — nothing else stores it."
        if has_topics
        else "There are no topics. Deleting clears any fit still queued or failed."
    )
    confirm, cancel, *_ = st.columns(stage.CONTROLS)
    if confirm.button(
        "Yes, delete the topics",
        key="topics-confirm",
        type="primary",
        width="stretch",
    ):
        removed = client.delete_topics()
        del st.session_state["topics-armed"]
        st.success(
            f"Removed {removed['topics']} topic(s) and "
            f"{removed['memberships']} membership(s)."
        )
        st.rerun()
    if cancel.button("Cancel", key="topics-cancel", width="stretch"):
        del st.session_state["topics-armed"]
        st.rerun()


def _staleness(fit: dict) -> None:
    """Warns, per language, when the stored topics have fallen behind.

    Re-chunking a document deletes its passages, and the membership rows go
    with them. Nothing else reports that: the topics stay `modelled` with the
    passage count of a corpus that no longer exists.
    """
    for one in fit["languages"]:
        if not one["topics"]:
            continue
        if not one["memberships"]:
            st.error(
                f"**{one['language']}** — {one['topics']} topic(s) hold no "
                "passages at all. The passages they were fitted over have been "
                "re-chunked, which deleted every membership. Everything "
                "topic-weighted is empty for this language until you fit again."
            )
        elif one["corpus_passages"] != one["live_passages"]:
            st.warning(
                f"**{one['language']}** — fitted over "
                f"{one['corpus_passages']:,} passages; the corpus now holds "
                f"{one['live_passages']:,}. Fit again to take the difference in."
            )
    if fit["passages_without_language"]:
        st.warning(
            f"{fit['passages_without_language']:,} passage(s) carry no language "
            "and so belong to no model. They are too short for the detector to "
            "judge; they and everything drawn from them are outside every "
            "topic-weighted report."
        )


def _provenance(fit: dict) -> None:
    """Says what the stored topics were fitted over, and when."""
    if fit["error"]:
        st.warning(f"The last fit failed: {fit['error']}")
    if not fit["fitted_at"]:
        return

    parts = [f"Fitted {fit['fitted_at'][:16].replace('T', ' ')}"]
    if fit["corpus_passages"] is not None:
        parts.append(f"over {fit['corpus_passages']:,} passages")
    if fit["corpus_vocabulary"] is not None:
        parts.append(f"on a {fit['corpus_vocabulary']:,}-term vocabulary")
    st.caption(" · ".join(parts) + ".")

    unplaced = fit["passages_without_topics"]
    if unplaced:
        st.warning(
            f"{unplaced:,} passage(s) were placed in no topic — every one of "
            "their terms was filtered out of the vocabulary. They, and "
            "anything drawn from them, are outside every topic-weighted "
            "report. Lower TOPIC_NO_BELOW to bring them in."
        )


page.render(view)
