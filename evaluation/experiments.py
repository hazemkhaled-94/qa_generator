"""Scoring the golden cases into Phoenix, so two runs can be compared.

`make test-eval` already measures a served model against these cases, and
prints the numbers. Printing is the problem: a prompt change, a model
change or a quantisation change moves them, and what somebody wants to
know is whether it moved them up. Scrollback does not answer that.

So the same cases become a Phoenix dataset, and a run becomes an
experiment against it. Phoenix keeps the dataset versioned and every
experiment beside it, which makes "is this prompt better than last
week's" a question with an answer.

What is scored is what the pipeline already believes. The evaluators here
call the project's own checker and its own gates rather than an LLM judge:
the checker is what decides whether a fact is kept in production, so a
golden set scored by anything else measures something the pipeline does
not use.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation import cases
from evaluation.config import Settings

log = logging.getLogger(__name__)

#: The three datasets, by the name the CLI takes.
EXTRACTION = "extraction-golden"
QUESTIONS = "questions-golden"
PHRASING = "phrasing-golden"
NAMES = (EXTRACTION, QUESTIONS, PHRASING)


def _client(settings: Settings):
    """The Phoenix client.

    Imported here rather than at module scope so `--help` and the tests
    that read the case counts do not need the package installed.
    """
    from phoenix.client import Client

    return Client(
        base_url=settings.base_url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )


def _examples(name: str) -> tuple[list[dict], list[dict]]:
    """The inputs and the expected outputs for one dataset.

    Phoenix keeps the two apart: a task is given the input and never the
    expectation, which is what stops an evaluator that accidentally reads
    the answer from scoring itself.
    """
    if name == EXTRACTION:
        return (
            [
                {"text": one["text"], "language": one["language"]}
                for one in cases.EXTRACTION
            ],
            [{"claims": one["claims"]} for one in cases.EXTRACTION],
        )
    if name == PHRASING:
        return (
            [
                {"question": one["question"], "language": one["language"]}
                for one in cases.PHRASING
            ],
            [
                {
                    "names_its_source": one["names_its_source"],
                    "self_contained": one["self_contained"],
                    "name": one["name"],
                }
                for one in cases.PHRASING
            ],
        )
    return (
        [
            {
                "question": one["question"],
                "passage": one["passage"],
                "target": one["target"],
                "language": one["language"],
            }
            for one in cases.QUESTIONS
        ],
        [
            {"recoverable": one["recoverable"], "name": one["name"]}
            for one in cases.QUESTIONS
        ],
    )


def upload(name: str, settings: Settings) -> str:
    """Puts one golden set in Phoenix, as a new version if it is there.

    Phoenix versions a dataset rather than replacing it, so a case added
    today does not invalidate the experiments run against the set as it
    was yesterday - they stay attached to the version they scored.

    Returns:
        The dataset's id.
    """
    inputs, outputs = _examples(name)
    dataset = _client(settings).datasets.create_dataset(
        name=name,
        inputs=inputs,
        outputs=outputs,
        dataset_description=(
            "The cases this pipeline is supposed to get right, from "
            "evaluation/cases.py. Several are here because they caught "
            "something."
        ),
    )
    log.info("%s: %d example(s) in Phoenix", name, len(inputs))
    return dataset.id


def _passage(text: str, language: str):
    """One passage as chunking would have stored it.

    Built here rather than through tests/factories.py, which does the
    same thing: a tool the Makefile runs must not import a test helper,
    and the helper's defaults are chosen for tests rather than for this.
    The sentence split is the real one - a fact cites a sentence number,
    so a passage numbered any other way would be scored against
    citations that do not mean what they say.
    """
    from extraction.models import PassageToExtract
    from nlp.analysis import sentences

    return PassageToExtract(
        id=0,
        text=text,
        section_path=None,
        block_type=None,
        language=language,
        sentences=sentences(text, language),
        table_cells=[],
    )


def _extraction_task(extractor, checker):
    """Builds the task that reads one passage, for the experiment."""

    def task(input: dict) -> dict:
        """Reads one passage and reports what survived the checks."""
        under = _passage(input["text"], input["language"])
        proposed = extractor.extract(under)
        checked = [checker.check(under, one, "llm") for one in proposed]
        return {
            "proposed": len(proposed),
            "validated": sum(1 for one in checked if one.validated),
            "statements": [one.statement for one in checked if one.validated],
            "refused": [one.rejection_code for one in checked if not one.validated],
        }

    return task


def _recall(output: dict, expected: dict) -> float:
    """How much of what a passage carries the model actually found.

    Capped at one: a model that proposed six facts for a passage carrying
    four has not achieved 150% recall, it has over-decomposed, and that
    shows up in precision instead.
    """
    claims = expected["claims"]
    return min(1.0, output["validated"] / claims) if claims else 0.0


def _precision(output: dict, expected: dict) -> float:
    """What share of what the model proposed survived the checks."""
    del expected
    proposed = output["proposed"]
    return output["validated"] / proposed if proposed else 0.0


def _majority(field: str) -> bool:
    """Whichever answer is commonest for one judgement across the cases.

    What a judge that ignores its input would say to score best. Computed
    off the cases rather than written down, so adding a case moves the
    floor with it instead of leaving a stale number in a docstring.
    """
    said = [bool(one[field]) for one in cases.PHRASING]
    return sum(said) * 2 >= len(said)


def _constant(field: str):
    """An evaluator scoring what a judge that never reads its input gets.

    The number every phrasing score has to be read against, and the reason
    it is here rather than in a README: `self_contained` measured 84.2%
    against a constant-answer floor of 78.9%, which is one case in
    nineteen, and that is the measured version of a judgement that fired
    zero times in 3,131 questions. A score without this beside it looks
    like competence.

    Phoenix averages an evaluator over the set, so returning 1 or 0 per
    case gives the floor as a percentage in the same column as the real
    score.
    """
    answer = _majority(field)

    def evaluator(output: dict, expected: dict) -> float:
        """Whether always answering `answer` would be right here."""
        del output
        return float(bool(expected[field]) == answer)

    evaluator.__name__ = f"{field}_constant_floor"
    return evaluator


def _judged(field: str):
    """An evaluator scoring one phrasing judgement against its label."""

    def evaluator(output: dict, expected: dict) -> float:
        """1 when the pipeline agreed with the label, 0 when it did not."""
        said = output.get(field)
        return float(said is not None and bool(said) == bool(expected[field]))

    evaluator.__name__ = field
    return evaluator


def _answered(output: dict, expected: dict) -> float:
    """What share of the judgements were answered at all.

    `names_its_source` abstains where no rule settles it and no model
    could be reached, and an abstention is not a wrong answer - it is a
    question nobody answered, which must not reject a question. Scored
    separately so a run where the model was unreachable reads as that
    rather than as a run where the judge got everything wrong.
    """
    del expected
    return (
        float(
            sum(
                output.get(one) is not None
                for one in ("names_its_source", "self_contained")
            )
        )
        / 2
    )


def _phrasing_task(judge):
    """Builds the task that reads one question, for the experiment.

    The real path and not a reimplementation of it: `gates.cites_source`
    settles what a pattern settles and only the residue reaches the model,
    `nlp.pointing` finds the pointing words and the model rules on whether
    they land. That is what `checker._phrasing` does, and scoring anything
    else would measure something production does not run.
    """
    from nlp.analysis import pointing
    from question_generation.gates import cites_source

    def task(input: dict) -> dict:
        """Judges one question's phrasing, rules first."""
        question, language = input["question"], input["language"]
        ruled = cites_source(question, language)
        source = ruled if ruled is not None else judge.names_its_source(question)

        pointers = pointing(question, language)
        # No pointing word is not an unanswered question: there is nothing
        # for a pointer to fail to land on, so the question stands on its
        # own by construction and the model is never asked.
        contained = True if not pointers else judge.self_contained(question, pointers)
        return {
            "names_its_source": source,
            "self_contained": contained,
            "settled_by_rule": ruled is not None,
            "pointers": list(pointers),
        }

    return task


def run(name: str, settings: Settings) -> str:
    """Runs one experiment and returns where to read it.

    Only the extraction set is scored here, and that is a scope decision
    rather than a gap. Its numbers are pure measurement - precision and
    recall against a claim count, asserted against nothing - which is
    exactly what is worth comparing between two prompts. The questions
    set already asserts: `test_the_gate_splits_the_golden_cases` requires
    the recoverable cases to pass and the rest to be stopped, so it is a
    gate rather than a trend, and it fails a pull request instead of
    drawing a line.

    It is uploaded regardless, so the cases are browsable in Phoenix and
    an experiment can be run against them from the UI. Wiring a task for
    it here would mean loading EMBEDDING_MODEL's 2.2 GB and a second
    served model to score ten cases a gate already checks.

    Raises:
        NotImplementedError: For any set with no task, naming what does
            measure it instead.
    """
    if name == PHRASING:
        return _phrasing(settings)

    if name != EXTRACTION:
        raise NotImplementedError(
            f"{name} is uploaded but not scored here: its cases are a gate "
            f"rather than a measurement, and `make test-eval` is what fails "
            f"when the model stops splitting them correctly. Run an "
            f"experiment against it from the Phoenix UI if you want one."
        )

    from extraction.extractors.llm import LlmExtractor
    from extraction.validation import FactChecker
    from llm.client import Client as ModelClient
    from llm.config import Settings as ModelSettings

    model = ModelSettings.load()
    log.info("scoring %s against %s", name, model.model)

    client = _client(settings)
    dataset = client.datasets.get_dataset(dataset=name)
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=_extraction_task(LlmExtractor(ModelClient(model)), FactChecker()),
        evaluators={"recall": _recall, "precision": _precision},
        experiment_name=settings.run_name or f"extraction-{model.model}",
        experiment_description=(
            f"{model.model} at temperature {model.temperature}, scored by "
            f"the pipeline's own checker."
        ),
        experiment_metadata={
            "model": model.model,
            "temperature": model.temperature,
            "structured_mode": model.structured_mode,
        },
    )
    url = client.experiments.get_experiment_url(
        dataset_id=dataset.id, experiment_id=_id_of(experiment)
    )
    log.info("%s: recorded at %s", name, url)
    return url


def _phrasing(settings: Settings) -> str:
    """Scores the two phrasing judgements, each against a constant floor.

    A measurement and not a gate, like extraction's and unlike the
    questions set. The numbers it produces are the ones in
    `evaluation/README.md`'s baseline table, which were read off a
    terminal and typed in by hand: the split by language, the 15/19 and
    the 16/19, and the constant-answer floor that showed `self_contained`
    was not reading its input.

    Scored against QUESTIONS_PHRASING_MODEL, which is the model the
    pipeline actually asks, so swapping a 31B for a 12B and re-running
    this is the measurement that decides it.
    """
    from llm.client import Client as ModelClient
    from llm.config import Settings as ModelSettings
    from question_generation.config import Settings as QuestionSettings
    from question_generation.phrasing import PROMPT_VERSION, PhrasingJudge

    model = ModelSettings.load()
    questions = QuestionSettings.load()
    asked = model.overridden(questions.phrasing_model or questions.verifier_model)
    log.info("scoring %s against %s", PHRASING, asked.model)

    client = _client(settings)
    dataset = client.datasets.get_dataset(dataset=PHRASING)
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=_phrasing_task(PhrasingJudge(ModelClient(asked))),
        evaluators={
            "names_its_source": _judged("names_its_source"),
            "self_contained": _judged("self_contained"),
            # Both floors, so neither score is read without the number it
            # has to beat sitting in the next column.
            "names_its_source_floor": _constant("names_its_source"),
            "self_contained_floor": _constant("self_contained"),
            "answered": _answered,
        },
        experiment_name=settings.run_name or f"phrasing-{asked.model}",
        experiment_description=(
            f"{asked.model} at temperature {asked.temperature}, on the "
            f"residue the rules do not settle. Prompt version "
            f"{PROMPT_VERSION}."
        ),
        experiment_metadata={
            "model": asked.model,
            "temperature": asked.temperature,
            "prompt_version": PROMPT_VERSION,
            "names_its_source_floor": _majority("names_its_source"),
            "self_contained_floor": _majority("self_contained"),
        },
    )
    url = client.experiments.get_experiment_url(
        dataset_id=dataset.id, experiment_id=_id_of(experiment)
    )
    log.info("%s: recorded at %s", PHRASING, url)
    return url


def _id_of(experiment: Any) -> str:
    """The experiment's id, however the client hands it back."""
    if isinstance(experiment, dict):
        return experiment.get("id") or experiment["experiment_id"]
    return getattr(experiment, "id", None) or experiment.experiment_id
