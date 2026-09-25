"""The stages in order, and the name each tool files them under.

Four tools hold the same seven stages and all four sort their own names
alphabetically, which is not the order a document moves: Argilla listed
`facts, questions, topic-labels`, Phoenix `assessment-, extraction-,
questions-, topics-`, and Dagster's asset list `assessments` first. A
number in front is what makes each tool's own ordering the pipeline's.

The number is the stage's place in the pipeline and means the same thing
in every store, so `4-extraction` in Phoenix, `4-facts` in Argilla and
`4_extraction` in Dagster are the same step seen from three sides.

Here because telemetry is the layer every process already imports and the
one that owns the vocabulary the four stores share - `run.id`, the project
name, the field names. It holds no logic of its own and imports nothing.
"""

from __future__ import annotations

#: Every stage, in the order a document moves through them. `ingestion` is
#: first and owns no queue; `assessment` is last and nothing reads it.
STAGES: tuple[str, ...] = (
    "ingestion",
    "parsing",
    "chunking",
    "extraction",
    "topic_modelling",
    "question_generation",
    "assessment",
)

#: Where each stage sits, from 1.
POSITION: dict[str, int] = {stage: index + 1 for index, stage in enumerate(STAGES)}


def numbered(name: str, stage: str | None = None, separator: str = "-") -> str:
    """Puts a stage's place in the pipeline in front of a name.

    Args:
        name: What the tool calls the thing.
        stage: Which stage it belongs to, when that is not `name` itself.
        separator: What joins the number to the name.

    Returns:
        The name with its position in front, or unchanged where the stage
        is not one of the seven.
    """
    at = POSITION.get(stage or name)
    return f"{at}{separator}{name}" if at else name


def project(service: str, run: str) -> str:
    """What Phoenix files one run of one service under.

    A service that is not a stage keeps its own name, so a host command
    that names a run is still filed under something readable.
    """
    return f"{numbered(service)}-{run}"
