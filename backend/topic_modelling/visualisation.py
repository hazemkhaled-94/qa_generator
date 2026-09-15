"""Rendering a fitted model as a pyLDAvis page.

Built during a fit and stored, from the whole term-topic matrix rather than
the top terms the database keeps.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import numpy as np
import pyLDAvis
from pyLDAvis import urls

from topic_modelling.models import TopicSpace

log = logging.getLogger(__name__)

#: Terms listed per topic in the bar chart.
_TERMS_SHOWN = 30


def _real_pcoa(distances):
    """Places the topics in two dimensions, dropping any imaginary part.

    ponytail: pyLDAvis eigendecomposes a symmetric matrix with `np.linalg.eig`,
    which returns complex values that json.dumps then refuses. The upgrade is
    `eigh`, which is the routine for a symmetric matrix.
    """
    return np.real(pyLDAvis.js_PCoA(distances))


@cache
def _asset(path: str) -> str:
    """Reads one of pyLDAvis's bundled browser assets."""
    return Path(path).read_text(encoding="utf-8")


def render(space: TopicSpace, language: str) -> bytes:
    """Renders one language's model as a self-contained HTML page.

    d3, the LDAvis script and its stylesheet are inlined, so the page reaches
    no network when it is opened. Topics keep the numbers they were fitted
    with.

    Args:
        space: The whole fitted model.
        language: ISO 639-1 code, used in the log line and the refusal.

    Returns:
        The page, UTF-8 encoded.

    Raises:
        ValueError: If the model placed no passage in any topic.
    """
    if not space.doc_topic:
        raise ValueError(
            f"no {language} passage carries a topic, so there is nothing to draw"
        )

    # The relevance ranking takes a log of a term weight, and the
    # factorisation leaves that at exactly zero for a term absent from a topic.
    with np.errstate(divide="ignore"):
        prepared = pyLDAvis.prepare(
            topic_term_dists=space.topic_term,
            doc_topic_dists=space.doc_topic,
            doc_lengths=space.doc_lengths,
            vocab=space.vocabulary,
            term_frequency=space.term_frequency,
            R=min(_TERMS_SHOWN, len(space.vocabulary)),
            mds=_real_pcoa,
            # Numbered as the topics table numbers them.
            sort_topics=False,
            start_index=0,
        )
    log.info(
        "%s: drew %d topic(s) over %d passage(s), %d term(s)",
        language,
        len(space.topic_term),
        len(space.doc_topic),
        len(space.vocabulary),
    )
    # An empty stylesheet in place of the CDN one the template would link.
    page = pyLDAvis.prepared_data_to_html(prepared, ldavis_css_url="data:text/css,")
    return _inlined(page).encode("utf-8")


def _inlined(html: str) -> str:
    """Puts d3 and the LDAvis script ahead of the page that asks for them."""
    return (
        f"<style>{_asset(urls.LDAVIS_CSS_LOCAL)}</style>"
        f"<script>{_asset(urls.D3_LOCAL)}</script>"
        f"<script>{_asset(urls.LDAVIS_LOCAL)}</script>"
        f"{html}"
    )
