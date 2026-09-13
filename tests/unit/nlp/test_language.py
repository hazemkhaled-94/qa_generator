"""Reading the language a passage is written in.

These documents carry a German report and its English summary in one file,
so the document's own label is not what a passage is read under.
"""

from __future__ import annotations

import pytest

GERMAN = (
    "Die Bundesanstalt für Finanzdienstleistungsaufsicht beaufsichtigt "
    "Institute und prüft deren Risikomanagement regelmäßig."
)
ENGLISH = (
    "The supervisory authority examines the risk management of the "
    "institutions it supervises on a regular basis."
)

#: An English list of links to a German site.
LINKS = (
    "1. [Risks arising from significant corrections on the international "
    "financial markets](https://www.bafin.de/EN/die-bafin/"
    "publikationen-daten/risiken-im-fokus/Fokusrisiken_2026/RIF1/rif_1.html)\n"
    "2. [Risks arising from corporate loan defaults](https://www.bafin.de/EN/"
    "die-bafin/publikationen-daten/risiken-im-fokus/Fokusrisiken_2026/RIF2/"
    "rif_2.html)"
)


@pytest.mark.parametrize(
    ("text", "expected"), [(GERMAN, "de"), (ENGLISH, "en"), (LINKS, "en")]
)
def test_prose_is_read_in_its_own_language(text: str, expected: str) -> None:
    """The addresses in a link do not decide the language of its text."""
    from nlp.language import detect

    assert detect(text) == expected, detect(text)


@pytest.mark.parametrize(
    "text",
    [
        "Risiko",
        "",
        "[Digitalisation](https://www.bafin.de/EN/die-bafin/x.html)",
    ],
)
def test_too_short_to_judge_is_not_guessed(text: str) -> None:
    """Nothing is returned, and the caller falls back to the document's."""
    from nlp.language import detect

    assert detect(text) is None
