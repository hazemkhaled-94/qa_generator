"""What the judge is asked, per metric, and what its answers are called.

**The names are arize-phoenix-evals'. The templates are not.**

That split is the whole of this module. A `hallucination` annotation in
Phoenix's Evaluations view means something to anybody who has used Phoenix
before: two labels, `factual` and `hallucinated`, scored 0.0 and 1.0
because the metric's optimisation direction is *minimise*. Renaming it, or
scoring it the other way round, would produce a column that looks like the
one they know and sorts backwards. So the name, the labels, the scores and
the direction here are read off phoenix-evals' own evaluator configs and
pinned in `_CHOICES` below.

The prompts behind them are this repository's, for three reasons the
shipped ones cannot meet:

- **Half of this corpus is German.** The shipped templates instruct in
  English about an English answer, and `evaluation/README.md` records what
  a judgement that is thinking in the wrong language does: gpt-4.1
  answered a phrasing judgement 7/7 in English and 6/12 in German. Every
  template here says the artefact may be in any language and that the
  judgement is about what it does, not what it is written in.
- **The artefacts are not a RAG answer.** phoenix-evals asks about a
  query, a context and a response. A fact has a statement and the sentences
  it cites; a topic has terms and a name. Substituting those into a
  three-slot template about retrieval asks the model to judge something it
  is not looking at.
- **An answer with no reason is not usable.** What this phase produces is
  a queue for `review/`, so every template asks for the explanation in the
  same call as the label.

The examples are deliberately about nothing - see
`tests/static/test_prompts_name_no_domain.py`, which reads this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from database.qa_generator import ArtifactKind, JudgeMetric

#: Bumped whenever a template below changes what it asks for. Recorded on
#: every assessment, because two templates are two judgements the way two
#: extraction prompts are two datasets.
PROMPT_VERSION = "1"

#: Each metric's labels and the score phoenix-evals gives them, and which
#: label is the good one. Taken from that package's own evaluator configs:
#: `hallucination` scores 1.0 for `hallucinated` because its optimisation
#: direction is minimise, and every other metric here scores 1.0 for the
#: label it wants. `approved` is stored per metric rather than derived from
#: the score for exactly that reason - summing scores across these four
#: would add a hallucination rate to a relevance rate.
_CHOICES: dict[str, tuple[dict[str, float], str]] = {
    JudgeMetric.HALLUCINATION: ({"hallucinated": 1.0, "factual": 0.0}, "factual"),
    JudgeMetric.RELEVANCE: ({"relevant": 1.0, "unrelated": 0.0}, "relevant"),
    JudgeMetric.QA_CORRECTNESS: ({"correct": 1.0, "incorrect": 0.0}, "correct"),
    JudgeMetric.SUMMARIZATION: ({"good": 1.0, "bad": 0.0}, "good"),
}

#: How each metric is optimised, as phoenix-evals declares it. Recorded on
#: the annotation's metadata so a Phoenix chart of the mean is read the
#: right way up.
_DIRECTION: dict[str, str] = {
    JudgeMetric.HALLUCINATION: "minimize",
    JudgeMetric.RELEVANCE: "maximize",
    JudgeMetric.QA_CORRECTNESS: "maximize",
    JudgeMetric.SUMMARIZATION: "maximize",
}

#: The sentence every template carries, because the corpus is not in one
#: language and the judgement is not about which one.
_ANY_LANGUAGE = """The material may be in any language, and the examples below are in English
only because these instructions are. Judge what it does, not what it is
written in."""


@dataclass(frozen=True)
class Template:
    """One judgement, as the model is asked it and as Phoenix files it.

    Attributes:
        metric: The name Phoenix shows, which is phoenix-evals' name.
        kind: Which artefact this judges.
        system: What the model is told it is doing.
        user: The material, with `{{name}}` where a row is substituted.
        labels: The answers allowed, in the order the prompt offers them.
        scores: What each label scores, as phoenix-evals scores it.
        good: The label that counts as approval.
        direction: `maximize` or `minimize`, as phoenix-evals declares it.
    """

    metric: str
    kind: str
    system: str
    user: str
    #: The placeholders `user` carries, which is also what `rendered` must
    #: be given. Declared rather than scraped, so a template gaining a
    #: `{{name}}` that nothing supplies is caught by the check below and by
    #: `tests/unit/assessment/test_templates.py` rather than reaching a
    #: model as the braces themselves.
    fields: tuple[str, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """The answers this metric allows."""
        return tuple(_CHOICES[self.metric][0])

    @property
    def scores(self) -> dict[str, float]:
        """What each label scores."""
        return dict(_CHOICES[self.metric][0])

    @property
    def good(self) -> str:
        """The label that counts as approval."""
        return _CHOICES[self.metric][1]

    @property
    def direction(self) -> str:
        """How this metric is optimised, as phoenix-evals declares it."""
        return _DIRECTION[self.metric]

    def rendered(self, **fields: str) -> str:
        """The user message with this artefact's text substituted in.

        `{{name}}` rather than a format specifier, because the template is
        also what `stages/prompts.py` records for Phoenix, and a prompt
        recorded with `{}` in it is one the playground cannot show.

        Raises:
            KeyError: If a placeholder was left unfilled. A template
                variable nobody substitutes reaches the model as the braces
                themselves, with no error anywhere and an answer that looks
                fine - which is the failure
                `tests/static/test_prompts_name_no_domain.py` catches for
                the prompts that fill theirs one `.replace` at a time. This
                one substitutes generically, so it checks instead.
        """
        missing = [name for name in self.fields if name not in fields]
        if missing:
            raise KeyError(
                f"{self.metric} on a {self.kind} needs {', '.join(self.fields)} "
                f"and was not given {', '.join(missing)}"
            )
        text = self.user
        for name, value in fields.items():
            text = text.replace("{{" + name + "}}", value)
        return text

    def shape(self) -> type[BaseModel]:
        """The Pydantic shape this metric's answer must come back in.

        Built per template so the `label` field is a `Literal` of exactly
        this metric's labels: an answer outside the rails is then a schema
        failure instructor retries, rather than a string nothing maps.
        """
        labels = Literal[self.labels]  # pyright: ignore[reportInvalidTypeForm]

        class Judgement(BaseModel):
            """One judgement, and why."""

            label: labels = Field(  # pyright: ignore[reportInvalidTypeForm]
                description=f"One of: {', '.join(self.labels)}."
            )
            explanation: str = Field(
                description="One or two sentences saying why, naming the part "
                "of the material that decided it."
            )

        # Named for the metric, because the shape's name is what reaches the
        # span as `tag.tags` and the log line as `llm.shape`. Left alone,
        # every one of the seven would arrive as `Judgement`.
        Judgement.__name__ = f"_{self.metric.title().replace('_', '')}"
        return Judgement


# ── Facts ────────────────────────────────────────────────────────────────

_FACT_HALLUCINATION = f"""ROLE
You are a fact checker. You judge ONE property of ONE statement and nothing
else about it.

ACTION
Say whether the cited sentences support every part of the statement.

STEPS
1. Read the cited sentences.
2. Read the statement, and list every thing it asserts - every number, name,
   date, quantity, condition and claim.
3. For each one, find where the cited sentences say it.
4. Answer `factual` only when every part is found. Answer `hallucinated` if
   any part is absent, altered, or asserted more strongly than the sentences
   do.

CONTEXT
The statement was drawn FROM the cited sentences, so the question is not
whether it is true in the world. It is whether these sentences say it.

A statement that says LESS than the sentences is still factual. A statement
that rounds, generalises, or adds a qualifier the sentences do not carry is
not.

{_ANY_LANGUAGE}

EXAMPLES
factual:      sentences say a request must be confirmed within five working
              days; statement says a request must be confirmed within five
              working days.
factual:      sentences list four conditions; statement asserts the second of
              them alone.
hallucinated: sentences say a review happens quarterly; statement says a
              review happens quarterly and is chaired by the site manager.
              (nothing says who chairs it)
hallucinated: sentences say a threshold is "usually" met; statement says it
              "is always" met.

FORMAT
Return `label`, one of `factual` or `hallucinated`, and `explanation`."""

_FACT_RELEVANCE = f"""ROLE
You are a source checker. You judge ONE property of ONE statement and nothing
else about it.

ACTION
Say whether the cited sentences are the right evidence for this statement -
whether they are ABOUT what the statement is about.

STEPS
1. Read the statement and name its subject.
2. Read the cited sentences and name theirs.
3. Decide whether somebody looking for evidence for this statement would be
   satisfied by these sentences.
4. Answer `relevant` when they bear on the statement's subject, `unrelated`
   when they are about something else.

CONTEXT
This is not the same question as whether the sentences SUPPORT the statement,
which is asked separately. Evidence can be on-subject and still not support
what is claimed, and that case is `relevant` here.

What this catches is a citation that points at the wrong place: a statement
about one thing resting on sentences about another, which is what happens
when the sentence numbering and the text have come apart.

{_ANY_LANGUAGE}

EXAMPLES
relevant:  statement is about a notice period; sentences discuss notice
           periods, even if they give a different number.
relevant:  statement names a duty; sentences set out who holds that duty.
unrelated: statement is about a reporting deadline; sentences are a list of
           document revisions.
unrelated: statement asserts a definition; sentences are a page header and a
           copyright notice.

FORMAT
Return `label`, one of `relevant` or `unrelated`, and `explanation`."""

_FACT_USER = """STATEMENT
{{statement}}

CITED SENTENCES
{{evidence}}"""


# ── Topics ───────────────────────────────────────────────────────────────

_TOPIC_SUMMARIZATION = f"""ROLE
You are a librarian. You judge ONE property of ONE label and nothing else
about it.

ACTION
Say whether the label is a good short name for the terms beneath it.

STEPS
1. Read the terms.
2. Decide what, if anything, they are collectively about.
3. Read the label.
4. Answer `good` when somebody shown the label alone would expect these
   terms; `bad` when the label is wrong, far too broad, or names only a
   fraction of them.

CONTEXT
The terms are the strongest words of one cluster, in weight order, so the
first few matter most. A label that names those and ignores a long tail is
still good.

A label is `bad` when it is generic enough to fit any cluster at all - "General
information", "Various topics", "Document" - because such a label tells a
reader nothing they did not already know.

{_ANY_LANGUAGE}

EXAMPLES
good: terms are response, deadline, escalation, urgent, acknowledge; label is
      "Response times and escalation".
good: terms are heat, thermal, insulation, transfer, conductivity; label is
      "Thermal behaviour".
bad:  terms are invoice, payment, net, due, remittance; label is "Documents".
      (true of the terms and true of every other cluster too)
bad:  terms are training, competence, assessor, certificate, refresher; label
      is "Safety equipment". (names something the terms do not)

FORMAT
Return `label`, one of `good` or `bad`, and `explanation`."""

_TOPIC_RELEVANCE = f"""ROLE
You are a librarian. You judge ONE property of ONE cluster of terms and
nothing else about it.

ACTION
Say whether these terms describe a SUBJECT the material is about, or the
furniture that surrounds it.

STEPS
1. Read the terms.
2. Decide whether they name something a reader could want to know about.
3. Answer `relevant` when they do, `unrelated` when they are page furniture,
   boilerplate, or a mixture with no subject in it.

CONTEXT
Every corpus carries text that is real, well-formed and about nothing anybody
wants to ask: page headers and footers, copyright notices, revision tables,
tables of contents, distribution lists, document control blocks.

A cluster of those is not a bad cluster. It is an accurate description of
material that should not be asked about, and saying so is what this judgement
is for.

A mixture is also `unrelated`: terms drawn from several unconnected subjects
at once describe no subject.

{_ANY_LANGUAGE}

EXAMPLES
relevant:  storage, temperature, humidity, ambient, range.
relevant:  appeal, decision, grounds, deadline, submit.
unrelated: page, figure, table, see, appendix.
unrelated: version, revision, approved, date, issued, supersedes.
unrelated: colour, deadline, welding, invoice, staffing. (five subjects, no
           subject)

FORMAT
Return `label`, one of `relevant` or `unrelated`, and `explanation`."""

_TOPIC_USER = """TERMS, strongest first
{{terms}}

LABEL
{{label}}"""

_TOPIC_TERMS_ONLY = """TERMS, strongest first
{{terms}}"""


# ── Questions ────────────────────────────────────────────────────────────

_QUESTION_HALLUCINATION = f"""ROLE
You are an examiner. You judge ONE property of ONE expected answer and
nothing else about it.

ACTION
Say whether the facts beneath support every part of the expected answer.

STEPS
1. Read the facts.
2. Read the question, then the expected answer.
3. List every thing the answer asserts.
4. For each one, find where the facts say it.
5. Answer `factual` only when every part is found. Answer `hallucinated` if
   any part is absent from the facts or contradicts them.

CONTEXT
The question is here so you know what was asked, not to be judged. Whether
the answer is the RIGHT answer is a different judgement, asked separately.
This one is only about whether the answer rests on the facts shown.

An answer that is correct in the world and absent from the facts is
`hallucinated`: the facts are what this dataset's answers have to be
checkable against.

{_ANY_LANGUAGE}

EXAMPLES
factual:      facts state a limit of thirty days; answer is "thirty days".
factual:      facts state three conditions; answer names all three.
hallucinated: facts state a limit of thirty days; answer is "thirty days, or
              sixty by agreement". (nothing about an agreement)
hallucinated: facts describe a procedure; answer names the department that
              runs it, which no fact names.

FORMAT
Return `label`, one of `factual` or `hallucinated`, and `explanation`."""

_QUESTION_QA_CORRECTNESS = f"""ROLE
You are an examiner. You judge ONE property of ONE question-and-answer pair
and nothing else about it.

ACTION
Say whether the expected answer actually answers the question that was asked.

STEPS
1. Read the question and decide precisely what it asks for.
2. Read the expected answer.
3. Decide whether it answers THAT, completely.
4. Answer `correct` when it does; `incorrect` when it answers a different
   question, answers only part of the one asked, or does not answer at all.

CONTEXT
Whether the answer rests on the facts is judged separately. This is about
fit: a true, well-sourced statement that answers a question nobody asked is
`incorrect` here.

A question asking "how many" wants a number. One asking "why" wants a reason.
One asking for a list wants the whole list, and an answer giving two of four
items is `incorrect`.

{_ANY_LANGUAGE}

EXAMPLES
correct:   "How long is allowed for a reply?" / "Five working days."
correct:   "Why must a change be recorded?" / "So that the previous
           configuration can be restored if the change fails."
incorrect: "How long is allowed for a reply?" / "Replies are handled by the
           duty officer." (answers who, not how long)
incorrect: "Which three checks are required?" / "A visual check and a
           pressure test." (two of three)

FORMAT
Return `label`, one of `correct` or `incorrect`, and `explanation`."""

_QUESTION_RELEVANCE = f"""ROLE
You are a retrieval analyst. You judge ONE property of ONE question and
nothing else about it.

ACTION
Say whether the facts beneath contain the information needed to answer the
question.

STEPS
1. Read the question.
2. Read the facts.
3. Decide whether somebody holding only these facts could answer it.
4. Answer `relevant` when they could, `unrelated` when the facts are about
   something else.

CONTEXT
This is the question a retrieval system is scored on: were the right passages
found. Here it is asked of the facts a question was WRITTEN from, so
`unrelated` means the question drifted away from its own source while it was
being written.

Do not judge whether the expected answer is right, and do not use knowledge
of your own. Only whether these facts bear on this question.

{_ANY_LANGUAGE}

EXAMPLES
relevant:  question asks for a notice period; facts state notice periods.
relevant:  question asks why a step exists; facts give the reason.
unrelated: question asks about storage temperature; facts are about invoice
           handling.
unrelated: question asks what a term means; facts use the term and never
           define it.

FORMAT
Return `label`, one of `relevant` or `unrelated`, and `explanation`."""

_QUESTION_USER = """QUESTION
{{question}}

EXPECTED ANSWER
{{answer}}

FACTS IT WAS WRITTEN FROM
{{facts}}"""

_QUESTION_NO_ANSWER_USER = """QUESTION
{{question}}

FACTS IT WAS WRITTEN FROM
{{facts}}"""


#: Every template, by artefact and then by metric, in the order a reader
#: should see them. What a kind is asked is exactly what is listed here:
#: adding a metric is adding an entry, and nothing else reads a metric name.
TEMPLATES: dict[str, tuple[Template, ...]] = {
    ArtifactKind.FACT: (
        Template(
            metric=JudgeMetric.HALLUCINATION,
            kind=ArtifactKind.FACT,
            system=_FACT_HALLUCINATION,
            user=_FACT_USER,
            fields=("statement", "evidence"),
        ),
        Template(
            metric=JudgeMetric.RELEVANCE,
            kind=ArtifactKind.FACT,
            system=_FACT_RELEVANCE,
            user=_FACT_USER,
            fields=("statement", "evidence"),
        ),
    ),
    ArtifactKind.TOPIC: (
        Template(
            metric=JudgeMetric.SUMMARIZATION,
            kind=ArtifactKind.TOPIC,
            system=_TOPIC_SUMMARIZATION,
            user=_TOPIC_USER,
            fields=("terms", "label"),
        ),
        Template(
            metric=JudgeMetric.RELEVANCE,
            kind=ArtifactKind.TOPIC,
            system=_TOPIC_RELEVANCE,
            user=_TOPIC_TERMS_ONLY,
            fields=("terms",),
        ),
    ),
    ArtifactKind.QUESTION: (
        Template(
            metric=JudgeMetric.HALLUCINATION,
            kind=ArtifactKind.QUESTION,
            system=_QUESTION_HALLUCINATION,
            user=_QUESTION_USER,
            fields=("question", "answer", "facts"),
        ),
        Template(
            metric=JudgeMetric.QA_CORRECTNESS,
            kind=ArtifactKind.QUESTION,
            system=_QUESTION_QA_CORRECTNESS,
            user=_QUESTION_USER,
            fields=("question", "answer", "facts"),
        ),
        Template(
            metric=JudgeMetric.RELEVANCE,
            kind=ArtifactKind.QUESTION,
            system=_QUESTION_RELEVANCE,
            user=_QUESTION_NO_ANSWER_USER,
            fields=("question", "facts"),
        ),
    ),
}


def metrics_of(kind: str) -> tuple[str, ...]:
    """Which metrics one artefact kind is judged on."""
    return tuple(one.metric for one in TEMPLATES[kind])


def direction_of(metric: str) -> str:
    """How one metric is optimised, as phoenix-evals declares it.

    Read by whatever reports a mean, because `hallucination`'s is the one
    that runs the other way: a project whose mean score is high is one
    whose judge found hallucinations, and a reader shown that beside three
    maximised metrics without this would read it upside down.
    """
    return _DIRECTION[metric]
