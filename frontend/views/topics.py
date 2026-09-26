"""Topics view. Runs topic modelling, and no other stage."""

from __future__ import annotations

from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from lib import backend, catalog, configure, explain, lineage, page, stage

#: The one stage this page runs. A fit is all-or-nothing over one
#: vocabulary, so it has no per-topic form.
_TOPICS = stage.Queue("topics", "Topic modelling", "fits", "modelled")

#: What fails a fit. Nothing here judges one topic - a fit succeeds whole or
#: not at all - so these are raised rather than fitted through.
_GATES = (
    explain.Gate(
        "no language",
        "list languages",
        "Some passage carries a language to fit a model for.",
        "nothing",
        "rule",
    ),
    explain.Gate(
        "empty vocabulary",
        "filter",
        "Some term survives the rare-and-common filter.",
        "nothing",
        "rule",
    ),
    explain.Gate(
        "every term is uninformative",
        "filter",
        "Some surviving term has a non-zero idf, without which every topic "
        "would come out NaN.",
        "nothing",
        "rule",
    ),
)

#: The order this service works in, and what runs each step.
_STEPS = (
    explain.Step(
        "Claim the request",
        "The oldest outstanding fit. A fit is always the whole corpus, so "
        "there is no per-topic form of any of this.",
        ("repository.py",),
        (),
    ),
    explain.Step(
        "List the languages",
        "One model per language, because a mixed corpus fitted as one model "
        "spends its topics telling the languages apart — `delivery` and "
        "`Lieferung` are the same subject and share no characters. The cost "
        "is that topic numbers only mean something within a language.",
        ("service.py",),
        (),
    ),
    explain.Step(
        "Build and filter the vocabulary",
        "Read from the lemmas chunking already stored, not recomputed. Terms "
        "in too few passages or too many of them distinguish nothing and are "
        "dropped.",
        ("service.py",),
        ("TOPIC_NO_BELOW", "TOPIC_NO_ABOVE"),
    ),
    explain.Step(
        "Fit",
        "Non-negative matrix factorisation over the passage-by-term matrix, "
        "weighted by tf-idf. Every component is a topic. Nothing is "
        "pretrained and nothing is embedded, so no word list, ontology or "
        "domain model has to be supplied.",
        ("service.py",),
        ("TOPIC_NUM_TOPICS", "TOPIC_PASSES", "TOPIC_RANDOM_STATE"),
    ),
    explain.Step(
        "Read the topics out",
        "The highest-weighted terms of each component. This is also the only "
        "signature a label is matched on across a refit.",
        ("service.py",),
        ("TOPIC_TOP_TERMS",),
    ),
    explain.Step(
        "Score every passage",
        "Each passage is scored against the fitted model, keeping every "
        "membership above the floor. A passage above none of them is "
        "unplaced, which is reported rather than hidden.",
        ("service.py",),
        ("TOPIC_MIN_WEIGHT",),
    ),
    explain.Step(
        "Name and carry the labels over",
        "A served model names each topic from its terms. Labels from the "
        "previous fit are carried across where the top terms still overlap, "
        "each used at most once, so a refit does not rename the corpus.",
        ("service.py",),
        ("TOPIC_MODEL", "TOPIC_LABEL_MIN_FACT_SHARE"),
    ),
)

#: Everything this page can say about the service it runs.
_SERVICE = explain.Service(
    what=(
        "Finds what the corpus is about: the groups of words that tend to "
        "appear together, one model per language, over the corpus's own "
        "vocabulary. **A fit is always the whole corpus** — the components "
        "are estimated jointly, so adding a document makes all of them "
        "stale, and one ask replaces the whole model. That is why this stage "
        "has no per-item controls and no Start."
    ),
    steps=_STEPS,
    gates=_GATES,
)

#: Room for the pyLDAvis figure, which is drawn at a fixed size.
_MAP_HEIGHT = 850

#: What the search box can look in.
_FIELDS = {"Label and terms": "both", "Label": "label", "Terms": "terms"}


def view() -> None:
    """Renders the topics page."""
    page.header("Topics")

    client = backend.catalog_api()
    fit = client.topic_fit()
    topics = client.topics()

    with page.panel("Overview"):
        page.stats(_overview(fit, topics))

    with st.expander("Analysis"):
        page.findings(_health(fit, topics))
        _map(client, fit)

    with page.panel("Topic modelling"):
        stage.fit(client, _TOPICS)
        explain.panel(_SERVICE, "How topic modelling works")
        configure.panel("topics")

    words, field = catalog.search("topics", _FIELDS)
    chosen, pager = catalog.filters(
        "topics",
        {
            "language": (
                "Languages",
                sorted({one["language"] for one in topics if one["language"]}),
                "One language model.",
            ),
            "coverage": (
                "Coverages",
                ["counted", "excluded"],
                "Whether the topic counts towards coverage.",
            ),
        },
    )

    # Filtered here rather than by the backend: /topics returns every topic
    # at once, because one fit produces a list a person can read.
    selected = [one for one in topics if _matches(one, words, field, chosen)]
    total, rows = catalog.paged(
        "topics",
        pager,
        lambda limit, offset: {
            "total": len(selected),
            "topics": selected[offset : offset + limit],
        },
        "topics",
    )

    with page.panel(f"Topics · {total:,}"):
        if not rows:
            st.caption("Nothing matches." if topics else "No topics. Fit the model.")
        else:
            picked = page.table(
                [
                    {
                        "Language": topic["language"] or "—",
                        "#": topic["topic_index"],
                        "Label": topic["label"] or "—",
                        "Top terms": ", ".join(topic["top_terms"]),
                        "Passages": topic["passages"],
                        "Dominant in": topic["dominant_passages"],
                        "Documents": topic["documents"],
                        "Facts each": f"{_per_passage(topic):.1f}",
                        "In coverage": "yes" if topic["include_in_coverage"] else "no",
                    }
                    for topic in rows
                ],
                key="topics-table",
                column_config={"Top terms": st.column_config.TextColumn(width="large")},
            )
            if picked is not None:
                _detail(client, rows[picked], topics)

    _removal(client, bool(topics))


def _matches(topic: dict, words: str, field: str, chosen: dict) -> bool:
    """Reports whether one topic passes the search and the filters."""
    if chosen["language"] and topic["language"] != chosen["language"]:
        return False
    if chosen["coverage"] and topic["include_in_coverage"] != (
        chosen["coverage"] == "counted"
    ):
        return False
    if not words:
        return True
    label = (topic["label"] or "").lower()
    terms = " ".join(topic["top_terms"]).lower()
    looked = {"label": label, "terms": terms, "both": f"{label} {terms}"}[field]
    return words.lower() in looked


def _per_passage(topic: dict) -> float:
    """Validated facts per passage this topic owns."""
    owned = topic["dominant_passages"]
    return topic["validated_facts"] / owned if owned else 0.0


def _overview(fit: dict, topics: list[dict]) -> dict[str, tuple[str, str]]:
    """Names the figures describing the stored model as a whole."""
    languages = fit["languages"]
    live = sum(one["live_passages"] for one in languages)
    placed = sum(topic["dominant_passages"] for topic in topics)
    fitted = [one["fitted_at"] for one in languages if one["fitted_at"]]
    return {
        "Topics": (f"{len(topics):,}", "Topics stored, across every language."),
        "Language models": (f"{len(languages):,}", "One model per language found."),
        "Passages placed": (
            page.share(placed, live),
            "Passages whose highest weight is some topic.",
        ),
        "Fitted": (_age(min(fitted) if fitted else None), "Age of the oldest model."),
    }


def _age(fitted_at: str | None) -> str:
    """Reads a fit timestamp as an age."""
    if not fitted_at:
        return "never"
    when = datetime.fromisoformat(fitted_at)
    days = (datetime.now(tz=when.tzinfo) - when).days
    return "today" if days < 1 else f"{days:,}d"


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
                f"{one['topics']} topic(s) hold no passages. Fit again.",
            )
        elif one["corpus_passages"] != one["live_passages"]:
            state, meaning = (
                "Attention",
                "The corpus has changed since this model was fitted.",
            )
        else:
            state, meaning = ("OK", "Fitted over the passages the corpus holds now.")
        rows.append(
            {
                "Check": f"{name} — corpus unchanged since the fit",
                "Value": f"{one['live_passages']:,} passages now",
                "Should be": f"{(one['corpus_passages'] or 0):,} at fit time",
                "State": state,
                "What it means": meaning,
            }
        )

        unplaced = one["passages_without_topics"] or 0
        rows.append(
            {
                "Check": f"{name} — every passage placed in a topic",
                "Value": f"{unplaced:,} unplaced",
                "Should be": "0",
                "State": "OK" if not unplaced else "Attention",
                "What it means": "Every passage reached a topic above the weight floor."
                if not unplaced
                else "These sit in no topic. Lower TOPIC_NO_BELOW or TOPIC_MIN_WEIGHT.",
            }
        )

    without = fit["passages_without_language"]
    rows.append(
        {
            "Check": "Every passage has a language",
            "Value": f"{without:,} without one",
            "Should be": "0",
            "State": "OK" if not without else "Attention",
            "What it means": "Every passage belongs to one of the models above."
            if not without
            else "These are too short for the detector, so they belong to no model.",
        }
    )

    if topics:
        outside = sum(1 for topic in topics if not topic["include_in_coverage"])
        rows.append(
            {
                "Check": "Topics counted in coverage",
                "Value": f"{outside:,} excluded",
                "Should be": "a deliberate choice",
                "State": "OK",
                "What it means": "Topics excluded by hand. They keep their "
                "passages either way.",
            }
        )

    if fit["error"]:
        rows.append(
            {
                "Check": "The last fit succeeded",
                "Value": fit["error"][:120],
                "Should be": "no error",
                "State": "Attention",
                "What it means": "The last fit did not finish. Retry above.",
            }
        )
    return rows


@st.cache_data(show_spinner=False, max_entries=4)
def _figure(_client, language: str | None, fitted_at: str | None) -> str | None:
    """Fetches one language's figure, remembering the last few.

    A pyLDAvis page inlines d3, the LDAvis script and the whole term-topic
    matrix, so it is megabytes rather than kilobytes - and this is drawn
    inside the Analysis fold, which runs its body whichever way it is
    folded. Uncached it was fetched from the API on every rerun of the
    script, which on Streamlit is every click on the page, whether or not
    anybody had opened the fold. The same reason the Documents page caches
    a file rather than pulling it again to redraw the viewer beside it.

    `fitted_at` is in the key and is not read: a fit is the only thing that
    redraws a figure, and it is the only thing that moves that timestamp,
    so a refit misses the cache and anything else hits it. `_client` is
    underscored so Streamlit leaves it out of the key.
    """
    return _client.topic_visualisation(language)


def _map(client, fit: dict) -> None:
    """Draws one language model as a pyLDAvis figure."""
    drawn_at = {
        one["language"]: one["fitted_at"] for one in fit["languages"] if one["topics"]
    }
    if not drawn_at:
        return

    language = st.selectbox(
        "Topic map",
        list(drawn_at),
        key="topics-map-language",
        help="One figure per language model. Topic numbers match the table.",
    )
    drawn = _figure(client, language, drawn_at[language])
    if drawn is None:
        st.caption(f"No map for {language}. A fit draws one.")
        return
    components.html(drawn, height=_MAP_HEIGHT, scrolling=True)


def _detail(client, chosen: dict, topics: list[dict]) -> None:
    """Shows everything held about one topic, and lets a person name it."""
    memberships = sum(topic["passages"] for topic in topics)

    with page.panel(f"Topic {chosen['language']} #{chosen['topic_index']}"):
        page.attributes(
            {
                "Id": chosen["id"],
                "Language": chosen["language"],
                "Number": chosen["topic_index"],
                "Label": chosen["label"],
                "Named by": chosen["labelled_by"],
                "In coverage": chosen["include_in_coverage"],
                "Passages": chosen["passages"],
                "Dominant in": chosen["dominant_passages"],
                "Share of memberships": page.share(chosen["passages"], memberships),
                "Documents": chosen["documents"],
                "Table passages": chosen["table_passages"],
                "Mean weight": chosen["mean_weight"],
                "Validated facts": chosen["validated_facts"],
                "Facts each": f"{_per_passage(chosen):.1f}",
                "Top terms": chosen["top_terms"],
            }
        )

        name, coverage, apply, *_ = st.columns([3.2, 1.6, 1.2, 1.4])
        label = name.text_input(
            "Label",
            value=chosen["label"] or "",
            key=f"topics-label-{chosen['id']}",
            placeholder="a short name for this subject",
            help="Shown wherever the topic is, in place of its number.",
        )
        included = coverage.checkbox(
            "In coverage",
            value=chosen["include_in_coverage"],
            key=f"topics-coverage-{chosen['id']}",
            help="Whether this topic counts as a subject to be covered.",
        )
        apply.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if apply.button(
            "Save",
            key="topics-save",
            type="primary",
            width="stretch",
            help="Records the label and the coverage choice.",
        ):
            client.describe_topic(chosen["id"], label.strip() or None, included)
            st.toast(f"Topic {chosen['language']} #{chosen['topic_index']} saved.")
            st.rerun()

        with st.expander("How this was produced"):
            lineage.panel("topic", chosen["id"])


def _removal(client, has_topics: bool) -> None:
    """Offers the deletion, behind a confirmation."""
    chosen = stage.removal(
        "topics",
        "Removes every topic at once. There is no per-topic delete: a topic "
        "is one column of a model fitted jointly. The rows are archived "
        "rather than destroyed, and nothing here puts them back.",
        {
            "all": (
                "Delete all topics",
                (
                    "Removes every topic, membership and label, and any "
                    "queued fit. Passages, facts and questions stay."
                ),
                "Delete every topic and membership?"
                if has_topics
                else "There are no topics. This clears any queued or failed fit.",
            )
        },
    )
    if chosen is None:
        return
    removed = client.delete_topics()
    st.toast(
        f"Removed {removed['topics']} topic(s), "
        f"{removed['memberships']} membership(s) and "
        f"{removed['labels']} label(s)."
    )
    st.rerun()


page.render(view)
