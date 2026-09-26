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

#: What Dagster calls each stage's asset. Ingestion has none: an upload
#: writes a row and parsing is what is then asked to run.
#:
#: Here rather than read off `orchestration`, which the frontend cannot
#: import - it is a code location in an image of its own.
#: `tests/static/test_tool_names.py` is what keeps the two equal.
ASSET: dict[str, str] = {
    "parsing": "parsed_documents",
    "chunking": "passages",
    "extraction": "facts",
    "topic_modelling": "topics",
    "question_generation": "questions",
    "assessment": "assessments",
}

#: What Argilla calls each stage's dataset, before the number goes in
#: front. Only the three a person is asked to judge; `dataset_name` in
#: `review.datasets` is what builds the full name.
#:
#: Here for the same reason as ASSET: no container carries the Argilla
#: client, so the frontend cannot import `review` to name the dataset it
#: is pointing somebody at.
DATASET: dict[str, str] = {
    "extraction": "facts",
    "topic_modelling": "topic-labels",
    "question_generation": "questions",
}

#: Which stage a route prefix belongs to, for the two that answer to
#: something other than their own name. `stages/cli.py` hands `queue_main`'s
#: name straight to telemetry as the service, and that name is the prefix:
#: question generation passed `questions`, `project` below could not place
#: it, and its spans went to a Phoenix project called `questions` while
#: every other stage carried its number.
#:
#: The prefix is the key because that is the side a caller holds.
STAGE_OF: dict[str, str] = {
    "topics": "topic_modelling",
    "questions": "question_generation",
}


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


def project(service: str, run: str | None = None) -> str:
    """What Phoenix files one service's spans under.

    A service that is not a stage keeps its own name, so a host command
    that names a run is still filed under something readable. A stage that
    named itself by its ROUTE is resolved through STAGE_OF first, which is
    what keeps question generation's project numbered with the rest.

    `run` is the NAME somebody chose, not the id every run has. Absent -
    which is every run nobody named - the project is the stage alone and
    successive runs append to it.

    Filing every run under a project of its own was the original design and
    it does not survive contact with a worker: `run_id` mints a uuid per
    PROCESS, `restart: unless-stopped` makes a process per restart, and one
    deployment reached five hundred projects, almost all of them holding a
    handful of spans nobody could find again. The comparison that design
    existed for is still there and is now the thing you ask for by name:

        make questions RUN_ID=a-gpt-4.1     ->  6-question_generation-a-gpt-4.1
        make questions                      ->  6-question_generation

    Runs stay separable inside a shared project either way, because
    `run.id` is a resource attribute on every span. What changed is that
    telling them apart is a filter rather than a hunt through a sidebar.
    """
    stage = STAGE_OF.get(service, service)
    return f"{numbered(stage)}-{run}" if run else numbered(stage)
