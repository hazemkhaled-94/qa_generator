"""Every setting a deployment may change, and what each one is.

One line each, plus the type, the bounds and the closed set of values where
there is one: what a page needs to draw a control and what a route needs to
refuse a value. `configs/env/backend.env` carries the long form.

One table rather than one per `config.py`, because the spaCy pipelines, the
worker poll and the pool sizes are read where they are used.

`invalidates` names the stages whose stored output this setting produced.
Direct staleness only: naming `parsing` names the passages, facts and
questions that cascade from it. A setting that only shapes the next run
names nothing.

`fixed` is served but never written: the deployment owns it and it needs a
restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: How a value is read. Also the name of its reader in `settings.env` and
#: the control a page draws for it.
Kind = Literal["integer", "decimal", "boolean", "text", "csv", "mapping"]

#: The services a setting may belong to: six stages, and the platform they
#: share. One page configures exactly one.
Service = Literal[
    "ingestion",
    "parsing",
    "chunking",
    "extraction",
    "topics",
    "questions",
    "platform",
]


@dataclass(frozen=True)
class Setting:
    """One setting: what it is, what it may hold, and what it stales.

    Attributes:
        name: The variable, as the environment and the store both spell it.
        service: Which service configures it.
        kind: How it is read.
        help: What it does, in one line.
        invalidates: Stages whose stored output this setting produced.
        low: Smallest accepted value, for a number.
        high: Largest accepted value, for a number.
        choices: The only accepted values, where the set is closed.
        optional: Whether absence is itself meaningful.
        fixed: Whether the deployment owns it, so it is served but not written.
    """

    name: str
    service: Service
    kind: Kind
    help: str
    invalidates: tuple[str, ...] = ()
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    optional: bool = False
    fixed: bool = False


#: Every setting, by service, in the order backend.env declares them, which
#: is the order a page draws them in.
SETTINGS: tuple[Setting, ...] = (
    # ── Ingestion ─────────────────────────────────────────────────────────
    # PIPELINE_VERSION is deliberately absent: it is read from
    # pyproject.toml, because a version describes the build and a
    # deployment given the ability to name its own can only use it to lie.
    Setting(
        name="MAX_FILE_SIZE_MB",
        service="ingestion",
        kind="integer",
        low=1,
        help="Largest upload accepted. Streamlit enforces its own ceiling "
        "first, from server.maxUploadSize, and rejects a larger file before "
        "the API sees it.",
    ),
    Setting(
        name="ALLOWED_MIME_TYPES",
        service="ingestion",
        kind="csv",
        choices=("application/pdf",),
        help="Media types with a parser behind them. Naming one without a "
        "parser reads as support that does not exist.",
    ),
    # ── Parsing ───────────────────────────────────────────────────────────
    Setting(
        name="PARSING_OCR_CHAR_THRESHOLD",
        service="parsing",
        kind="integer",
        low=0,
        invalidates=("parsing",),
        help="Characters per page below which a document counts as scanned. "
        "OCR is not implemented, so such a document is refused rather than "
        "converted into an empty one.",
    ),
    Setting(
        name="PARSING_MIN_CONFIDENCE",
        service="parsing",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("parsing",),
        help="Lowest confidence a conversion may carry and still be kept, "
        "read as the converter's own lower bound.",
    ),
    Setting(
        name="PARSING_TABLE_MODE",
        service="parsing",
        kind="text",
        choices=("fast", "accurate"),
        invalidates=("parsing",),
        help="TableFormer mode.",
    ),
    Setting(
        name="PARSING_HEADING_HIERARCHY",
        service="parsing",
        kind="boolean",
        invalidates=("parsing",),
        help="Whether the converter rebuilds the heading tree. Without it "
        "every section header comes back at level one.",
    ),
    Setting(
        name="PARSING_TIMEOUT_SECONDS",
        service="parsing",
        kind="decimal",
        low=1,
        help="Longest one conversion may run. Sized above an ordinary "
        "document on the slowest hardware it will run on, or it fires on "
        "ordinary files instead of pathological ones.",
    ),
    Setting(
        name="DOCLING_ARTIFACTS_PATH",
        service="parsing",
        kind="text",
        optional=True,
        help="Where the converter's model weights are. Absent lets Docling "
        "download them, which fails in a container with no network.",
    ),
    # ── Chunking ──────────────────────────────────────────────────────────
    Setting(
        name="CHUNKING_MERGE_PEERS",
        service="chunking",
        kind="boolean",
        invalidates=("chunking",),
        help="Combine undersized neighbours that share a heading. Off leaves "
        "one-line passages, which retrieve poorly and carry too little to "
        "draw a fact from.",
    ),
    # ── Extraction ────────────────────────────────────────────────────────
    Setting(
        name="EXTRACTION_KINDS",
        service="extraction",
        kind="csv",
        choices=("atomic", "summary", "outline", "bridge"),
        invalidates=("extraction",),
        help="Which readings of a passage a run produces. `atomic` is not "
        "optional: every passage is read for its claims.",
    ),
    Setting(
        name="EXTRACTION_DIGEST_MAX_SHARE",
        service="extraction",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("extraction",),
        help="The longest a summary or an outline may be, as a share of the "
        "passage it stands in for. `make extract-revalidate` re-judges "
        "stored facts against a new value without calling the model.",
    ),
    Setting(
        name="EXTRACTION_DIGEST_MIN_CHARS",
        service="extraction",
        kind="integer",
        low=0,
        invalidates=("extraction",),
        help="The shortest passage worth digesting, in characters. A digest "
        "is a fixed size whatever it is given, so how much it condenses is "
        "decided by the passage: under 400 characters the median summary "
        "came to 99% of its passage and 92% were refused as not_condensed. "
        "0 digests every passage carrying enough claims.",
    ),
    Setting(
        name="EXTRACTION_MIN_OTHER_SHARE",
        service="extraction",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("extraction",),
        help="Smallest share of a passage's facts that may be something "
        "other than atomic, which caps how many atomic facts one passage "
        "keeps. 0 turns the cap off.",
    ),
    Setting(
        name="EXTRACTION_DIGEST_MODEL",
        service="extraction",
        kind="text",
        optional=True,
        help="The model that condenses a passage into a summary and an "
        "outline. Unset uses whichever model reads it. Condensing is the one "
        "thing extraction asks for that a small model is trained to do.",
    ),
    Setting(
        name="EXTRACTION_DUPLICATE_COSINE",
        service="extraction",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("extraction",),
        help="How alike two statements may be before the second is refused "
        "as a duplicate. Cosine over the whole corpus and over the facts one "
        "passage already kept. 0 turns the gate off and writes no vectors.",
    ),
    Setting(
        name="EXTRACTION_BRIDGES_PER_TOPIC",
        service="extraction",
        kind="integer",
        low=0,
        invalidates=("extraction",),
        help="How many bridge calls one topic is worth.",
    ),
    Setting(
        name="EXTRACTION_BRIDGE_PASSAGES",
        service="extraction",
        kind="integer",
        low=2,
        invalidates=("extraction",),
        help="How many passages one bridge call is shown. Below two there is "
        "nothing to bridge.",
    ),
    Setting(
        name="EXTRACTION_MODEL",
        service="extraction",
        kind="text",
        optional=True,
        help="The model that reads a passage for its facts. Absent calls the "
        "one LLM_MODEL names. Facts already extracted keep the model they "
        "were written by, which is recorded on each of them.",
    ),
    # ── Topic modelling ───────────────────────────────────────────────────
    Setting(
        name="TOPIC_LANGUAGE_NAMES",
        service="topics",
        kind="mapping",
        invalidates=("topics",),
        help="ISO 639-1 code to the language's name, for the naming prompt.",
    ),
    Setting(
        name="TOPIC_NUM_TOPICS",
        service="topics",
        kind="integer",
        low=1,
        invalidates=("topics",),
        help="How many topics to fit per language, or the ceiling on it when "
        "TOPIC_PASSAGES_PER_TOPIC is set.",
    ),
    Setting(
        name="TOPIC_PASSAGES_PER_TOPIC",
        service="topics",
        kind="integer",
        low=0,
        invalidates=("topics",),
        help="How many passages one topic is worth, which scales the count "
        "to each language's share of the corpus. 0 fits TOPIC_NUM_TOPICS "
        "whatever the corpus is.",
    ),
    Setting(
        name="TOPIC_PASSES",
        service="topics",
        kind="integer",
        low=1,
        invalidates=("topics",),
        help="Passes the factorisation makes over the corpus.",
    ),
    Setting(
        name="TOPIC_RANDOM_STATE",
        service="topics",
        kind="integer",
        invalidates=("topics",),
        help="Seed, so one corpus fits the same topics twice.",
    ),
    Setting(
        name="TOPIC_TOP_TERMS",
        service="topics",
        kind="integer",
        low=1,
        invalidates=("topics",),
        help="How many terms are kept to stand for a topic.",
    ),
    Setting(
        name="TOPIC_MIN_WEIGHT",
        service="topics",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("topics",),
        help="Smallest weight a passage's membership of a topic may carry "
        "and still be recorded.",
    ),
    Setting(
        name="TOPIC_NO_BELOW",
        service="topics",
        kind="integer",
        low=0,
        invalidates=("topics",),
        help="A term in fewer passages than this is dropped from the vocabulary.",
    ),
    Setting(
        name="TOPIC_NO_ABOVE",
        service="topics",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("topics",),
        help="A term in more than this share of passages is dropped from the "
        "vocabulary, being too common to tell two topics apart.",
    ),
    Setting(
        name="TOPIC_LABEL_MIN_FACT_SHARE",
        service="topics",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("topics",),
        help="The smallest share of a topic's passages that must carry a "
        "validated fact before the topic is worth naming. Below it the topic "
        "is left unnamed and no model is called for it. 0 names every topic.",
    ),
    Setting(
        name="TOPIC_MODEL",
        service="topics",
        kind="text",
        optional=True,
        help="The model that names a topic from its terms. Absent calls the "
        "one LLM_MODEL names. Topics already fitted keep the label they were "
        "given, and what named them is recorded.",
    ),
    # ── Question generation ───────────────────────────────────────────────
    Setting(
        name="QUESTIONS_PER_TOPIC",
        service="questions",
        kind="integer",
        low=1,
        help="How many questions to aim for per topic. This times the topic "
        "count is what a full run costs. Not below the number of types with "
        "a weight, or a topic never sees some of them.",
    ),
    Setting(
        name="QUESTIONS_FACT_SAMPLE",
        service="questions",
        kind="integer",
        low=1,
        help="How many of a topic's facts are offered per call, divided "
        "between the passages the sample holds.",
    ),
    Setting(
        name="QUESTIONS_SAMPLES_PER_PASSAGE",
        service="questions",
        kind="integer",
        low=1,
        help="How many times one passage may be offered, each time with "
        "facts no earlier sample took.",
    ),
    Setting(
        name="QUESTIONS_FACT_KINDS",
        service="questions",
        kind="csv",
        choices=("atomic", "summary", "outline", "bridge"),
        help="Which kinds of fact a question may be written from. Each shape "
        "seeds a different question.",
    ),
    Setting(
        name="QUESTIONS_TYPE_MIX",
        service="questions",
        kind="mapping",
        help="Which types of question are written and in what proportion, as "
        "`type:weight`. A weight of 0, or a name left out, is never written.",
    ),
    Setting(
        name="QUESTIONS_DIFFICULTY_MIX",
        service="questions",
        kind="mapping",
        help="Which bands the plan aims for, as `band:weight`. A request for "
        "a shape of sample; the band itself stays read off what the question "
        "turned out to cite.",
    ),
    Setting(
        name="QUESTIONS_FOLLOWUP_TYPES",
        service="questions",
        kind="csv",
        help="The types a follow-up may take, cycled in order down a thread.",
    ),
    Setting(
        name="QUESTIONS_UNANSWERABLE_SHARE",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        help="Share of the plan that asks for a question the corpus cannot answer.",
    ),
    Setting(
        name="QUESTIONS_FOLLOWUP_SHARE",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        help="Share of the plan that asks for a follow-up to an earlier "
        "question rather than a new thread.",
    ),
    Setting(
        name="QUESTIONS_MAX_FOLLOWUPS",
        service="questions",
        kind="integer",
        low=0,
        help="How many questions may follow one root, down a single thread.",
    ),
    Setting(
        name="QUESTIONS_RETRIES",
        service="questions",
        kind="integer",
        low=0,
        help="How many more times a question may be written when a gate "
        "refused the draft, or it came out below the band its slot asked "
        "for. 0 keeps the first draft whatever it is.",
    ),
    Setting(
        name="QUESTIONS_ANSWER_CHARS",
        service="questions",
        kind="mapping",
        invalidates=("questions",),
        help="Shortest and longest target answer per form, as "
        "`form:min:max`. `make questions-reverify` re-judges stored "
        "questions against new bounds.",
    ),
    Setting(
        name="QUESTIONS_ANSWER_OVERLAP",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("questions",),
        help="How much of a list or an explanation has to come back for the "
        "verifier to have recovered it. Numbers are always exact.",
    ),
    Setting(
        name="QUESTIONS_OFF_TOPIC_OVERLAP",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("questions",),
        help="How much of what an unanswerable question is about has to "
        "occur in the material it was drawn from. Below it the question is "
        "off topic rather than unanswerable.",
    ),
    Setting(
        name="QUESTIONS_ELSEWHERE_PASSAGES",
        service="questions",
        kind="integer",
        low=0,
        invalidates=("questions",),
        help="How many passages the corpus-wide probe shows the verifier "
        "before an unanswerable question is accepted. 0 judges one against "
        "its own passages alone.",
    ),
    Setting(
        name="QUESTIONS_LONG_ANSWER_CHARS",
        service="questions",
        kind="integer",
        low=1,
        invalidates=("questions",),
        help="Length above which an answer is treated as a long one for the "
        "recall gate.",
    ),
    Setting(
        name="QUESTIONS_DUPLICATE_COSINE",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("questions",),
        help="How close two question embeddings may be before the later one "
        "is refused as already asked.",
    ),
    Setting(
        name="QUESTIONS_RELEASE_SIZE",
        service="questions",
        kind="integer",
        low=0,
        invalidates=("questions",),
        help="How many questions a balanced release holds. 0 draws the "
        "largest one the accepted pool can fill without missing a quota. "
        "`make questions-balance` draws it again.",
    ),
    Setting(
        name="QUESTIONS_RELEASE_UNANSWERABLE",
        service="questions",
        kind="decimal",
        low=0,
        high=1,
        invalidates=("questions",),
        help="The most of a release that may be questions with no answer. A "
        "ceiling, not a target.",
    ),
    Setting(
        name="QUESTIONS_RELEASE_DIFFICULTY",
        service="questions",
        kind="mapping",
        invalidates=("questions",),
        help="How a release spreads over the difficulty bands, counting the "
        "unanswerable questions, which are always easy.",
    ),
    Setting(
        name="QUESTIONS_MODEL",
        service="questions",
        kind="text",
        optional=True,
        help="The model that writes a question. Absent calls the one LLM_MODEL names.",
    ),
    Setting(
        name="QUESTIONS_VERIFIER_MODEL",
        service="questions",
        kind="text",
        optional=True,
        help="The second model, which checks that a question's answer is in "
        "the passages it cites. Naming the writer's own model, or none at "
        "all, turns off the two gates only an independent model may apply.",
    ),
    Setting(
        name="QUESTIONS_PHRASING_MODEL",
        service="questions",
        kind="text",
        optional=True,
        help="The model asked what a question's own wording amounts to, "
        "where a rule has not already settled it. Sees the question and "
        "never a passage, so it can be far smaller than the verifier. "
        "Absent calls QUESTIONS_VERIFIER_MODEL.",
    ),
    # ── The platform the stages share ─────────────────────────────────────
    Setting(
        name="LLM_MODEL",
        service="platform",
        kind="text",
        help="LiteLLM model id; the prefix picks the provider. The default "
        "for every stage that calls a model, and what a stage's own override "
        "replaces.",
    ),
    Setting(
        name="LLM_BASE_URL",
        service="platform",
        kind="text",
        optional=True,
        help="Where that model is served. Absent sends the call wherever the "
        "provider's own default is.",
    ),
    Setting(
        name="LLM_STRUCTURED_MODE",
        service="platform",
        kind="text",
        help="How the model is made to answer in the shape asked for, as an "
        "instructor.Mode name. A property of the provider: JSON_SCHEMA suits "
        "a self-hosted runtime, a hosted one usually wants TOOLS.",
    ),
    Setting(
        name="LLM_TEMPERATURE",
        service="platform",
        kind="decimal",
        low=0,
        high=2,
        help="Sampling temperature. 0 so the same passage gives the same "
        "facts on a re-run.",
    ),
    Setting(
        name="LLM_TIMEOUT_SECONDS",
        service="platform",
        kind="decimal",
        low=1,
        help="How long to wait for one call. Size it from a measurement: it "
        "also derives how long a claimed row may go unfinished.",
    ),
    Setting(
        name="LLM_MAX_ATTEMPTS",
        service="platform",
        kind="integer",
        low=1,
        help="How many times one call is tried before the row fails. Only a "
        "connection, a timeout or a rate limit is retried, and a rate limit "
        "waits as long as the provider asks.",
    ),
    Setting(
        name="LLM_NUM_CTX",
        service="platform",
        kind="integer",
        low=1,
        optional=True,
        help="Context window to ask the runtime for. Absent takes its "
        "default, which on a self-hosted runtime reserves the whole "
        "advertised window as key-value cache before reading anything.",
    ),
    Setting(
        name="LLM_REASONING_EFFORT",
        service="platform",
        kind="text",
        optional=True,
        choices=("none", "low", "medium", "high"),
        help="How much a thinking model may think before answering. Absent "
        "turns thinking off, which is what a structured answer wants. A "
        "hosted reasoning model wants `none` for that, and refuses a "
        "temperature other than 1 without it.",
    ),
    Setting(
        name="EMBEDDING_MODEL",
        service="platform",
        kind="text",
        invalidates=("chunking", "questions"),
        help="The one embedding model, and the only tokenizer in the "
        "project: chunking sizes a passage with it and questions are "
        "embedded by it. Its width must match questions.embedding, which is "
        "vector(1024).",
    ),
    Setting(
        name="EMBEDDING_MAX_TOKENS",
        service="platform",
        kind="integer",
        low=1,
        invalidates=("chunking",),
        help="That model's context window, and so the longest passage the "
        "chunker emits. Larger silently truncates at embed time.",
    ),
    Setting(
        name="NLP_MODELS",
        service="platform",
        kind="mapping",
        invalidates=("chunking", "extraction"),
        help="The spaCy pipeline per language, as `language:model`, and the "
        "languages the detector may answer with. Each must be present in the "
        "image; nothing fetches one at run time.",
    ),
    Setting(
        name="NLP_DEFAULT_LANGUAGE",
        service="platform",
        kind="text",
        invalidates=("chunking", "extraction"),
        help="Used for a passage too short for the detector to judge, and "
        "for any language NLP_MODELS does not name.",
    ),
    Setting(
        name="NLP_CAPITALISED_NOUNS",
        service="platform",
        kind="csv",
        optional=True,
        invalidates=("extraction", "topics"),
        help="Languages that write every noun with a capital, where a "
        "lower-case word tagged as a noun is a word from another language. "
        "Empty means no language is read that way.",
    ),
    Setting(
        name="WORKER_POLL_SECONDS",
        service="platform",
        kind="decimal",
        low=1,
        help="How long a worker waits before looking at an empty queue "
        "again. Read once when a worker starts, so a change reaches it on "
        "its next restart.",
    ),
    Setting(
        name="DATABASE_POOL_SIZE",
        service="platform",
        kind="integer",
        low=1,
        fixed=True,
        help="Connections per process. Every worker, the api and each host "
        "command open their own pool, so the total is this times however "
        "many are running.",
    ),
    Setting(
        name="DATABASE_POOL_OVERFLOW",
        service="platform",
        kind="integer",
        low=0,
        fixed=True,
        help="Connections a process may open beyond the pool size under load.",
    ),
)

#: Every setting by name, which is how a route and a worker both look one up.
BY_NAME: dict[str, Setting] = {one.name: one for one in SETTINGS}


def of(service: str) -> tuple[Setting, ...]:
    """Every setting one service configures, in declaration order."""
    return tuple(one for one in SETTINGS if one.service == service)


def writable(name: str) -> Setting:
    """Looks up a setting that may be written.

    Raises:
        KeyError: If nothing is configurable under that name.
        ValueError: If the deployment owns it, naming where it is set.
    """
    found = BY_NAME.get(name)
    if found is None:
        raise KeyError(f"{name} is not a setting this deployment configures")
    if found.fixed:
        raise ValueError(
            f"{name} belongs to the deployment rather than to the pipeline. "
            f"It is set in .env and read when a process starts."
        )
    return found
