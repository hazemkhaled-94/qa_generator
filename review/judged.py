"""What the LLM judge said, attached to the rows a person is shown.

A reviewer told nothing sees a wall of statements. `review/datasets.py`
already argues that the machine's own verdict belongs on the record -
"the checker refused this one for unsupported_addition" is what makes a
review answer the question it is for - and the evaluation phase produces a
second machine verdict about the same rows.

So the judge's opinion rides along with the checker's. The valuable case
is the one where they differ: an artefact the pipeline kept and the judge
refused is either a check that let something through or a judge that is
wrong, and a person is the only thing that settles it. The record says so
in as many words.

Anchoring is the risk and it is the lesser one, for the reason that file
gives: a sample nobody can interpret gets abandoned. Both machine verdicts
are shown, neither is presented as right, and the guidelines say the
disagreement is the point.

Reads through `assessment.repository`, which already knows which foreign
key belongs to which kind. A second copy of that here is a second place
for it to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from assessment.repository import AssessmentCatalog

#: What a record's `judge` metadata says when the phase has not run. A
#: value rather than an absent key, because Argilla's terms metadata
#: filters on what is there: `not_judged` is filterable and a missing
#: field is not, and "which of these has nobody judged" is a question a
#: reviewer asks.
NOT_JUDGED = "not_judged"


@dataclass(frozen=True)
class Opinion:
    """The judge's verdict about one artefact, as a reviewer is shown it.

    Attributes:
        verdict: `approved`, `refused` or `not_judged`, for the metadata
            filter.
        disagrees: Whether the pipeline kept this and the judge refused it.
        detail: The judgements in full, one per line, for the field a
            reviewer reads.
    """

    verdict: str
    disagrees: bool
    detail: str


#: An artefact nobody has judged. One instance, because it carries nothing.
UNJUDGED = Opinion(
    verdict=NOT_JUDGED,
    disagrees=False,
    detail=(
        "Nobody has judged this. The evaluation phase is off, or it has "
        "not reached this row."
    ),
)


def catalog() -> AssessmentCatalog:
    """The verdicts, read from the same database the rest of `review/` is.

    A function rather than an instance, so importing this module opens no
    connection: `review.run` builds one when a push is asked for, and the
    unit tests never do.
    """
    return AssessmentCatalog()


def opinions(judge: object | None, kind: str, ids: list[int]) -> dict[int, Opinion]:
    """The judge's verdict on some artefacts, by artefact id.

    One query for the whole push rather than one per record. Absent for an
    artefact the phase has not reached, which the caller reads as
    :data:`UNJUDGED`.

    `judge` of None is a deployment with no assessment catalogue wired -
    the evaluation phase switched off, or a caller that does not want the
    lookup. Every record then carries :data:`UNJUDGED`, which is what a
    reviewer should see when nothing has judged anything.
    """
    if judge is None or not ids:
        return {}
    found = judge.verdicts_for(kind, ids)  # pyright: ignore[reportAttributeAccessIssue]
    return {
        artifact_id: Opinion(
            verdict=_verdict(one.approved),
            disagrees=one.disagrees,
            detail=_detail(one),
        )
        for artifact_id, one in found.items()
    }


def _verdict(approved: bool | None) -> str:
    """How one verdict reads as a metadata term."""
    if approved is None:
        return NOT_JUDGED
    return "approved" if approved else "refused"


def _detail(one) -> str:
    """Every judgement about one artefact, as the reviewer reads it.

    The explanation and not just the label. A reviewer deciding whether
    the judge is right needs to see its reasoning, and that is the whole
    reason every template asks for one in the same call.
    """
    if not one.metrics:
        return UNJUDGED.detail
    lines = [
        f"{metric.metric}: {metric.label} — {metric.explanation}"
        for metric in one.metrics
    ]
    if one.disagrees:
        lines.append("")
        lines.append(
            "The pipeline KEPT this and the judge refused it. Neither is "
            "right by default — that is what you are deciding."
        )
    return "\n".join(lines)
