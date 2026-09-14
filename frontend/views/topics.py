"""Topics view."""

from __future__ import annotations

from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from lib import backend, page, stage

#: Room for the pyLDAvis figure, which is drawn at a fixed size.
_MAP_HEIGHT = 850


def _per_passage(topic: dict) -> float:
    """Validated facts per passage this topic owns."""
    owned = topic["dominant_passages"]
    return topic["validated_facts"] / owned if owned else 0.0


def view() -> None:
    """Renders the topics page."""
    page.header(
        "Topics",
        "Clusters over the corpus's own vocabulary — one model per language, "
        "because a single model across both spends topics on telling the "
        "languages apart. Pick a topic from the table to see its own figures "
        "and to name it.",
    )

    client = backend.catalog_api()
    fit = client.topic_fit()
    topics = client.topics()

    page.section(
        "The stored topic model",
        "What the topics currently in the database were fitted over, and "
        "whether they still describe the corpus as it is now. A fit is "
        "all-or-nothing, so these describe one model rather than a history "
        "of runs.",
    )
    page.metrics(_model_figures(fit, topics))
    page.findings(
        "Model health",
        "Each row is one thing that has to hold for the topics to describe "
        "the corpus. A row reading OK needs nothing done. These are checks "
        "on the stored model, not on any fit that may be running.",
        _health(fit, topics),
    )

    if not topics:
        st.info(
            "No topics yet. Chunk some documents, then run a fit with the button below."
        )
    else:
        st.divider()
        page.section(
            "Every topic in the model",
            "One row per topic per language. Topics are numbered within "
            "their own language model, so `de #3` and `en #3` are unrelated.",
        )
        st.dataframe(
            [
                {
                    "Language": topic["language"] or "—",
                    "#": topic["topic_index"],
                    "Label": topic["label"] or "—",
                    "Named by": topic["labelled_by"] or "—",
                    "Top terms": ", ".join(topic["top_terms"]),
                    "Passages": topic["passages"],
                    "Dominant in": topic["dominant_passages"],
                    "Documents": topic["documents"],
                    "Tables": page.share(
                        topic["table_passages"], topic["dominant_passages"]
                    ),
                    "Facts each": f"{_per_passage(topic):.1f}",
                    "Mean weight": f"{topic['mean_weight']:.2f}",
                    "In coverage": "yes" if topic["include_in_coverage"] else "no",
                }
                for topic in topics
            ],
            width="stretch",
            hide_index=True,
            column_config={"Top terms": st.column_config.TextColumn(width="large")},
        )

        st.divider()
        _map(client, fit)

        st.divider()
        chosen = st.selectbox(
            "Topic to work on",
            topics,
            format_func=lambda t: (
                f"{t['language']} #{t['topic_index']}: {', '.join(t['top_terms'][:6])}"
            ),
            key="topics-chosen",
            help="Everything below this point applies to this topic alone.",
        )
        _detail(client, chosen, topics)

    st.divider()
    _fit_controls(client, fit, bool(topics))
    _removal(client, bool(topics))


def _model_figures(fit: dict, topics: list[dict]) -> dict[str, tuple]:
    """Names the figures describing the stored model as a whole."""
    languages = fit["languages"]
    memberships = sum(one["memberships"] for one in languages)
    live = sum(one["live_passages"] for one in languages)
    placed = sum(topic["dominant_passages"] for topic in topics)
    fitted = [one["fitted_at"] for one in languages if one["fitted_at"]]
    return {
        "Topics": (
            f"{len(topics):,}",
            (
                "Topics currently stored, across every language model. "
                "TOPIC_COUNT sets how many each fit produces per language."
            ),
        ),
        "Language models": (
            f"{len(languages):,}",
            (
                "One model is fitted per language found in the corpus. A "
                "language with no model has no topics at all, and everything "
                "topic-weighted is empty for it."
            ),
        ),
        "Memberships": (
            f"{memberships:,}",
            (
                "Passage-to-topic links held right now. A passage belongs to "
                "several topics with a weight on each, so this is well above the "
                "passage count. Zero while topics exist means the passages they "
                "were fitted over have been deleted."
            ),
        ),
        "Passages placed": (
            *page.portion(placed, live),
            (
                "Passages that have some topic as their highest weight, as a "
                "share of the passages the corpus holds in a modelled language. "
                "This is the closest thing to coverage."
            ),
        ),
        "Model age": (
            *_age(min(fitted) if fitted else None),
            (
                "How long ago the oldest of the stored language models was "
                "fitted, with the date beneath. Adding documents does not make "
                "one topic stale; it makes all of them stale, because every "
                "topic is fitted jointly over one vocabulary."
            ),
        ),
    }


def _age(fitted_at: str | None) -> tuple[str, str]:
    """Reads a fit timestamp as an age, with the date beneath it."""
    if not fitted_at:
        return "never", "no fit has succeeded"
    when = datetime.fromisoformat(fitted_at)
    days = (datetime.now(tz=when.tzinfo) - when).days
    return (
        "today" if days < 1 else f"{days:,}d",
        when.date().isoformat(),
    )


def _health(fit: dict, topics: list[dict]) -> list[dict[str, str]]:
    """Builds one row per thing that has to hold for the topics to be usable."""
    rows = []
    for one in fit["languages"]:
        if not one["topics"]:
            continue
        name = one["language"]
        if not one["memberships"]:
            state, meaning = (
                "Attention",
                (
                    f"{one['topics']} topic(s) hold no passages. The passages "
                    "they were fitted over have been re-chunked, which deleted "
                    "every membership. Everything topic-weighted is empty for "
                    "this language until you fit again."
                ),
            )
        elif one["corpus_passages"] != one["live_passages"]:
            state, meaning = (
                "Attention",
                (
                    "The corpus has changed since this model was fitted. The "
                    "topics still describe the older corpus; fit again to take "
                    "the difference in."
                ),
            )
        else:
            state, meaning = (
                "OK",
                (
                    "The passages this model was fitted over are the passages "
                    "the corpus holds now."
                ),
            )
        rows.append(
            {
                "Check": f"{name} — corpus unchanged since the fit",
                "Failed": f"{one['live_passages']:,} passages now",
                "Should be": f"{(one['corpus_passages'] or 0):,} at fit time",
                "State": state,
                "What it means": meaning,
            }
        )

        unplaced = one["passages_without_topics"] or 0
        rows.append(
            {
                "Check": f"{name} — every passage placed in a topic",
                "Failed": f"{unplaced:,} unplaced",
                "Should be": "0",
                "State": "OK" if not unplaced else "Attention",
                "What it means": "Every passage of this language carried at "
                "least one term the vocabulary kept."
                if not unplaced
                else "Every term of these passages was filtered out of the "
                "vocabulary, so they sit in no topic. They, and anything "
                "drawn from them, are outside every topic-weighted report. "
                "Lower TOPIC_NO_BELOW to bring them in.",
            }
        )

    without = fit["passages_without_language"]
    rows.append(
        {
            "Check": "Every passage has a language",
            "Failed": f"{without:,} without one",
            "Should be": "0",
            "State": "OK" if not without else "Attention",
            "What it means": "Every passage was assigned a language, so each "
            "belongs to one of the models above."
            if not without
            else "These passages are too short for the detector to judge, so "
            "they belong to no model. They and everything drawn from them "
            "are outside every topic-weighted report.",
        }
    )

    outside = sum(1 for topic in topics if not topic["include_in_coverage"])
    if topics:
        rows.append(
            {
                "Check": "Topics counted in coverage",
                "Failed": f"{outside:,} excluded",
                "Should be": "a deliberate choice",
                "State": "OK",
                "What it means": "Topics excluded by hand, because they were "
                "too diffuse to be a meaningful coverage partition. They keep "
                "their passages either way. This is a decision, not a fault.",
            }
        )

    if fit["error"]:
        rows.append(
            {
                "Check": "The last fit succeeded",
                "Failed": fit["error"][:120],
                "Should be": "no error",
                "State": "Attention",
                "What it means": "The last fit that was asked for did not "
                "finish. The topics above, if any, are from an earlier fit. "
                "Retry below.",
            }
        )
    return rows


def _map(client, fit: dict) -> None:
    """Draws one language model as a pyLDAvis figure."""
    page.section(
        "Topic map",
        "The whole model at once, from pyLDAvis. The table above says how "
        "big each topic is; this says how far apart they are, and which "
        "terms separate them rather than merely appearing in them.",
    )
    languages = [one["language"] for one in fit["languages"] if one["topics"]]
    if not languages:
        return

    language = st.selectbox(
        "Language model to draw",
        languages,
        key="topics-map-language",
        help="One figure per language, because one model is fitted per "
        "language. Topic numbers match the table above.",
    )
    drawn = client.topic_visualisation(language)
    if drawn is None:
        st.info(
            f"No map for {language} yet. It is drawn during a fit, so topics "
            "modelled before this page existed have none — fit the model "
            "again with the button below to draw one."
        )
        return

    st.caption(
        "Left: every topic as a circle, sized by the share of the corpus it "
        "holds and placed so that topics using similar terms sit close "
        "together. Overlapping circles are topics that have not separated. "
        "Right: the terms of whichever topic you select — the pale bar is "
        "how often the term appears in the whole corpus, the dark bar how "
        "often inside the topic. A dark bar nearly as long as its pale one "
        "is a term that belongs to this topic rather than to everything.",
        help="Slide λ towards 0 to rank terms by how exclusive they are to "
        "the topic, and towards 1 to rank them by raw frequency. λ = 0.6 is "
        "usually the most readable. The Top terms column in the table above "
        "is the λ = 1 ordering.",
    )
    components.html(drawn, height=_MAP_HEIGHT, scrolling=True)


def _detail(client, chosen: dict, topics: list[dict]) -> None:
    """Shows one topic's own figures and lets a person name it."""
    total_memberships = sum(topic["passages"] for topic in topics)

    page.section(
        f"Topic {chosen['language']} #{chosen['topic_index']}",
        "Measurements of this topic alone, within its own language model.",
    )
    page.metrics(
        {
            "Passages": (
                f"{chosen['passages']:,}",
                (
                    "Passages holding this topic with any weight at all. A "
                    "passage belongs to several topics, so these overlap."
                ),
            ),
            "Dominant in": (
                *page.portion(chosen["dominant_passages"], chosen["passages"]),
                (
                    "Passages whose highest weight is this topic, as a share of "
                    "the passages that hold it. This is the closest thing to "
                    "`passages about this topic`."
                ),
            ),
            "Documents": (
                f"{chosen['documents']:,}",
                (
                    "Documents at least one of whose passages holds this topic. "
                    "A topic appearing in one document only is usually that "
                    "document's vocabulary rather than a subject."
                ),
            ),
            "Mean weight": (
                f"{chosen['mean_weight']:.2f}",
                (
                    "Mean weight across the passages holding it. A low mean over "
                    "many passages means a diffuse topic: present everywhere, "
                    "about nothing in particular."
                ),
            ),
            "Share of memberships": (
                *page.portion(chosen["passages"], total_memberships),
                (
                    "This topic's passages as a share of every membership in the "
                    "model. Far above the even split means one topic is "
                    "absorbing the corpus."
                ),
            ),
        }
    )
    page.metrics(
        {
            "Tables": (
                *page.portion(chosen["table_passages"], chosen["dominant_passages"]),
                (
                    "Passages it owns that are tables rather than prose. A topic "
                    "that is mostly tables is usually about a document's "
                    "apparatus - a budget annex, a figure's source line - rather "
                    "than about a subject. Whether that is worth keeping is a "
                    "judgement about this corpus, which is why it is a figure "
                    "here and not a rule in the code."
                ),
            ),
            "Facts each": (
                f"{_per_passage(chosen):.1f}",
                f"{chosen['validated_facts']:,} in total",
                (
                    "Validated facts per passage it owns. A question can only be "
                    "asked from a fact, so a topic well below the others yields "
                    "few questions however large it looks."
                ),
            ),
        }
    )
    st.caption(
        "Top terms: " + ", ".join(chosen["top_terms"]),
        help="The terms with the highest weight in this topic, in order. "
        "They are what a label is matched on when a refit tries to carry the "
        "label over.",
    )

    page.section(
        "Name this topic",
        "A name survives a refit by being matched on top terms, so a topic "
        "whose terms move too far loses it. Nothing else stores one.",
    )
    name, coverage, apply, *_ = st.columns([3.2, 1.6, 1.2, 1.4])
    label = name.text_input(
        "Label",
        value=chosen["label"] or "",
        key=f"topics-label-{chosen['id']}",
        placeholder="a short name for this subject",
        help="A short human name for the subject these terms describe. It is "
        "shown wherever the topic is, in place of its number.",
    )
    included = coverage.checkbox(
        "In coverage",
        value=chosen["include_in_coverage"],
        key=f"topics-coverage-{chosen['id']}",
        help="Clear this for a topic too diffuse to be a meaningful coverage "
        "partition, or for one that describes a document's apparatus rather "
        "than a subject. Tables and Facts each above are what to read it off. "
        "Nothing in the code decides this: what counts as a subject belongs to "
        "the corpus, not to the pipeline. The topic keeps its passages either "
        "way.",
    )
    apply.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if apply.button(
        "Save",
        key="topics-save",
        type="primary",
        width="stretch",
        help="Records the label and the coverage choice against this topic.",
    ):
        client.describe_topic(chosen["id"], label.strip() or None, included)
        st.toast(f"Topic {chosen['language']} #{chosen['topic_index']} saved.")
        st.rerun()


def _fit_controls(client, fit: dict, has_topics: bool) -> None:
    """Draws the corpus-wide fit controls, outside any polling fragment."""
    counts = client.stage_status("topics")["rows"]
    queued = counts.get("pending", 0)
    running = queued + counts.get("in_progress", 0)

    page.section(
        "Fit the model",
        "These three act on the whole corpus, and they are the only controls "
        "on this page that cannot be per item. " + stage.COLOUR_KEY,
    )
    st.caption(
        "Every topic is fitted jointly over one vocabulary, so a single topic "
        "cannot be started, stopped or refitted on its own: a new document "
        "does not make one topic stale, it makes all of them stale. A fit is "
        "all-or-nothing, and every existing topic is replaced when one "
        "succeeds. Each fit also redraws the topic map above."
    )
    stage.corpus_controls(
        client,
        "topics",
        {
            "discover": (
                "Fit the corpus",
                not running,
                "Queues a fit over every passage in the corpus, one model "
                "per language. Every existing topic and membership is "
                "replaced when it succeeds, and left alone when it does not. "
                "Labels are carried over where the top terms still match."
                if not running
                else "A fit is already queued or running.",
            ),
            "stop": (
                "Stop",
                bool(queued),
                f"Takes the {queued:,} queued fit(s) back off the queue. A "
                "fit already in progress finishes."
                if queued
                else "No fit is queued.",
            ),
            "retry": (
                "Retry",
                bool(counts.get("failed", 0)),
                "Returns a failed fit to the queue."
                if counts.get("failed", 0)
                else "No fit has failed.",
            ),
        },
    )
    _provenance(fit, has_topics)


def _provenance(fit: dict, has_topics: bool) -> None:
    """Says what the stored topics were fitted over, and when."""
    if not has_topics:
        return
    parts = []
    for one in fit["languages"]:
        if not one["fitted_at"]:
            continue
        detail = (
            f"**{one['language']}** fitted {one['fitted_at'][:16].replace('T', ' ')}"
        )
        if one["corpus_passages"] is not None:
            detail += f" over {one['corpus_passages']:,} passages"
        if one["corpus_vocabulary"] is not None:
            detail += f" on a {one['corpus_vocabulary']:,}-term vocabulary"
        parts.append(detail)
    if parts:
        st.caption(
            " · ".join(parts) + ".",
            help="What each stored language model was fitted over. Compare "
            "the passage count with the live one in Model health above.",
        )


def _removal(client, has_topics: bool) -> None:
    """Offers the deletion, behind a confirmation."""
    st.html(
        "<div class='qa-danger-zone'>"
        "<div class='qa-danger-title'>Delete</div>"
        "<div class='qa-danger-detail'>Removes every topic at once. There is "
        "no per-topic delete, for the same reason there is no per-topic fit: "
        "a topic is one column of a model fitted jointly, so removing one "
        "would leave the rest describing weights that no longer sum.</div>"
        "</div>"
    )

    remove, *_ = st.columns(stage.DANGER)
    if remove.button(
        "Delete all topics",
        key="danger-topics-delete",
        width="stretch",
        help="Removes every topic and every passage-topic membership, and "
        "any fit still queued. Passages, facts and questions all stay. Any "
        "label you assigned goes with the topic it was on.",
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
    confirm, cancel, *_ = st.columns(stage.DANGER)
    if confirm.button(
        "Yes, delete the topics",
        key="danger-confirm-topics",
        width="stretch",
    ):
        removed = client.delete_topics()
        del st.session_state["topics-armed"]
        st.success(
            f"Removed {removed['topics']} topic(s), "
            f"{removed['memberships']} membership(s) and "
            f"{removed['labels']} label(s)."
        )
        st.rerun()
    if cancel.button("Cancel", key="topics-cancel", width="stretch"):
        del st.session_state["topics-armed"]
        st.rerun()


page.render(view)
