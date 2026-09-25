"""Configuration for the served model."""

from __future__ import annotations

from dataclasses import dataclass, replace

from settings import Source, decimal, integer, optional, required

#: The context window a self-hosted model is asked for when the deployment
#: names none.
#:
#: Measured over 9,334 priced calls: the widest was 4,308 tokens of prompt
#: and answer together, p99 was 3,716, and the median 1,132. Those were
#: calls by models that do not think, and 6,144 covered them.
#:
#: **A reasoning model spends the window on reasoning.** One extraction
#: call against muse-glimmer:30b-mlx measured 1,252 tokens of prompt and
#: 4,702 of answer - 5,954 together, 97% of that 6,144, and the passage was
#: the median one. A p90 passage would have gone over. Thinking is left on
#: by default now (see `Settings.thinking`), so the window has to hold a
#: prompt, however long the model reasons, and the answer after it.
#:
#: Not smaller, and the margin is the reason. A call over the window is
#: TRUNCATED rather than refused - the oldest tokens go, which is the
#: system prompt, and the answer comes back looking fine and written
#: without the rules - so the cost of being too small is silent and the
#: cost of being too large is some cache. `Client.answer` warns within
#: `_NEAR_WINDOW` of it, on prompt AND completion, which is what makes the
#: first half of that sentence survivable.
#:
#: Larger costs nothing measurable here. Timed on the same prompt, same
#: passage, with the model unloaded between runs: 225s at 6,144 against
#: 224s at the model's full 131,072, and the MLX runner allocated LESS
#: memory at the larger window because it grows the cache as it fills it.
#: What a huge window does cost is parallelism - the runtime sizes its
#: concurrent slots from it, and at 131,072 it served one request at a
#: time - which is why this is a bounded number and not the advertised one.
WINDOW = 16384


def _provider(model: str) -> str:
    """The provider a litellm model id names, which is its prefix."""
    return model.split("/", 1)[0] if "/" in model else ""


def _self_hosted(model: str) -> bool:
    """Whether a model id names a runtime this deployment runs itself.

    `ollama` and `ollama_chat` are two prefixes and one runtime, which is
    why this reads the start of the provider rather than comparing it.
    """
    return _provider(model).startswith("ollama")


def _same_address(one: str | None, other: str | None) -> bool:
    """Whether two addresses are the same host, trailing slash aside."""
    if not one or not other:
        return False
    return one.rstrip("/") == other.rstrip("/")


@dataclass(frozen=True)
class Settings:
    """Which model to call, where, and how patiently.

    One set of values for every stage that calls a model: extraction reads
    passages with it and topic modelling names topics with it, and two
    settings for one served model is how the two come to disagree.

    A stage may name a different model with `overridden`, and only the
    model: the address, the mode and the patience are the deployment's.
    """

    model: str
    base_url: str | None
    structured_mode: str
    temperature: float
    timeout_seconds: float
    max_attempts: int
    #: The context window to ask the runtime for, or None to take its default.
    #: A self-hosted runtime reserves the whole window as key-value cache
    #: before it reads anything, so a model advertising 131,072 tokens holds
    #: gigabytes for a prompt of two thousand.
    num_ctx: int | None
    #: How much a thinking model may think before answering, or None to leave
    #: it to the model. `low`, `medium` and `high` let it think; anything else
    #: turns thinking off, which is what a structured answer wants.
    reasoning_effort: str | None
    #: Where a self-hosted runtime is, for a stage that overrides the model
    #: with one served there. None leaves every override on `base_url`.
    ollama_base_url: str | None = None

    def overridden(self, model: str | None) -> Settings:
        """These settings, calling a named model instead of the shared one.

        Returns self when nothing is named, so a caller needs no branch.

        **Only a CROSS-PROVIDER override moves anything else.** Naming
        another model of the provider already configured is the one-thing
        change this has always been; naming a model of a different one is
        not, because two of these values belong to the provider rather than
        to the deployment.

        **The address follows the provider, in both directions.** litellm
        reads a model id's prefix to pick the provider, so a stage naming
        `ollama_chat/...` while `LLM_BASE_URL` points at Azure sends an
        Ollama request to Azure and is answered with a 404 - which is what
        happened the first time a phrasing model was pointed at a local
        runtime. The one thing a stage may override is the model, and an
        address that contradicts it is not a second override; it is the
        first one not working.

        The other direction was missing, and `.env.example` is the shape it
        bites in: `LLM_MODEL` is an `ollama_chat/...` and `LLM_BASE_URL` is
        `http://localhost:11434`. A deployment built from that file which
        points one stage at a hosted model kept the local address and sent
        the hosted request to Ollama's port.

        What is dropped is only an address this can PROVE belongs to the
        runtime being left, which is when it is the one `OLLAMA_BASE_URL`
        names. A shared address that is something else is left alone: it may
        be a gateway in front of several providers, where the model id is
        what routes and the address is the deployment's after all. Absent,
        litellm uses the provider's own, which is what unset has always
        meant.

        Args:
            model: The model to call, or None to keep the shared one.
        """
        if not model or _provider(model) == _provider(self.model):
            return self if not model else replace(self, model=model)
        moved: dict[str, object] = {"model": model}
        if _self_hosted(model):
            if self.ollama_base_url:
                moved["base_url"] = self.ollama_base_url
        elif _self_hosted(self.model) and _same_address(
            self.base_url, self.ollama_base_url
        ):
            moved["base_url"] = None
        # Thinking is not decided here; see `thinking`. It follows the
        # model being called and not how that model was arrived at.
        return replace(self, **moved)

    @property
    def window(self) -> int | None:
        """The context window to ask for, or None to let the provider size it.

        A self-hosted runtime reserves the whole window as key-value cache
        before it reads anything, so a model advertising 131,072 tokens
        holds several gigabytes for a prompt of two thousand. It also sizes
        its parallelism from that window: at the advertised one this
        machine's Ollama served a single request at a time, so a worker
        holding a call blocked every other process for the length of it.

        Sized from the longest prompt a stage sends. Extraction sends one
        passage, capped at EMBEDDING_MAX_TOKENS; question generation sends
        up to QUESTIONS_FACT_SAMPLE facts and two passages plus the prompt,
        measured at 1,441 tokens. `WINDOW` covers both several times over.

        Derived per model for the reason `thinking` is: a hosted provider
        has no such parameter and refuses a request carrying it, so there
        is no one value and no way to default it in the file. LLM_NUM_CTX
        overrides it wherever a deployment knows better.
        """
        if self.num_ctx:
            return self.num_ctx
        return WINDOW if _self_hosted(self.model) else None

    @property
    def thinking(self) -> str | None:
        """What to send as `reasoning_effort`, or None to omit it.

        **Nothing, unless the deployment says otherwise.** The model is left
        its own default, which for a reasoning model means it thinks and for
        every other model means there is nothing to turn off.

        This used to send `off` to a self-hosted model, and the measurement
        behind that is still true: on this corpus's writer prompt gemma4:12b
        took 414.7 seconds and 8,555 output tokens - 32,436 characters of
        reasoning against 283 of answer - where thinking off took 12.6
        seconds and wrote a better question. A single boolean measured the
        same way was a median 19.5s against 3.2s.

        What that measurement could not see is the other half. `off` is a
        real instruction, not the absence of one, and a model built to think
        answers an instruction not to with **nothing at all**: muse-glimmer
        :30b-mlx returned 0 facts on every golden case and 6 output tokens
        per call, and proposed 2 of 2 correctly the moment it was allowed to
        think. Both pathologies end in an empty answer, so a default can
        only choose which family of open models is broken out of the box.

        It now chooses neither. Running on local models is the point of this
        pipeline, so an open model that thinks has to work on a clone, and
        one that should not think is one setting away. The cost of the
        reversal is paid where it can be seen rather than silently: a
        passage that yields nothing is a warning now, which is what says
        `off` is the setting this deployment wants.

        `off` is Ollama's spelling, a hosted reasoning model wants `none`,
        and each refuses the other's - which is why there is still no value
        that could be defaulted here for both. Ollama accepts `off` on a
        model that does not think at all, so it costs nothing to set.

        LLM_REASONING_EFFORT decides it, and nothing overrides that.
        """
        return self.reasoning_effort or None

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.
        """
        return cls(
            model=required("LLM_MODEL", source),
            base_url=optional("LLM_BASE_URL", source),
            structured_mode=required("LLM_STRUCTURED_MODE", source),
            temperature=decimal("LLM_TEMPERATURE", source),
            timeout_seconds=decimal("LLM_TIMEOUT_SECONDS", source),
            max_attempts=integer("LLM_MAX_ATTEMPTS", source),
            num_ctx=int(window)
            if (window := optional("LLM_NUM_CTX", source))
            else None,
            reasoning_effort=optional("LLM_REASONING_EFFORT", source),
            ollama_base_url=optional("OLLAMA_BASE_URL", source),
        )
