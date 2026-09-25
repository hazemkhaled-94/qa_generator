# NLP

Sentence splitting, claim analysis, vocabulary and language detection — the
linguistic surface every other stage reads off text.

One of the three packages that load something expensive; the others are
[`database/`](../database/README.md) and
[`blob_store/`](../blob_store/README.md). A caller asks for a sentence split;
it never loads a pipeline.

**No model is served from here.** Everything is local and deterministic, which
is what lets the fact checks run without a served model and why
`make extract-revalidate` costs seconds rather than hours.

Four files load weights, and nothing outside a worker may import them:

| File | Loads | Asked |
|---|---|---|
| [`embedding.py`](embedding.py) | `EMBEDDING_MODEL` | one text's vector |
| [`entailment.py`](entailment.py) | `NLI_MODEL` | whether one premise entails one hypothesis |
| [`qa.py`](qa.py) | `QA_MODEL` | where in a passage the answer is, or that there is none |

Each is held to its own window by [`windows.py`](windows.py):
`ENCODER_MAX_TOKENS` is the ceiling a deployment asks for, and a checkpoint
that cannot read that far is given what it can.
`tests/static/test_api_stays_light.py` pins that the API loads none of them.

## What it does

### Sentence numbering

Chunking splits every passage into numbered sentences and stores them. A fact
**cites one of those numbers** rather than quoting text, so its source span is
exact by construction — a citation is either a sentence the passage has or one
it does not.

It is computed once, by chunking, and stored. Nothing recomputes it.

### Counting claims

`not_atomic` — the check that a fact carries exactly one claim — is a
**finite-verb count**. A statement with more than one carries several claims;
one with none carries nothing.

The same reading is what lets the passage gate skip a heading, a caption or a
navigation line **before** the model is called.

### Vocabulary

Lemmas with the stop words and punctuation dropped, written onto every passage
by chunking and read by the topic model as the document it is fitted over.

`NLP_CAPITALISED_NOUNS` names the languages where capitalisation says nothing
about whether a noun is proper — German, where every noun is capitalised.

### Language detection

Detected on the **passage**, not inherited from its document. The detector may
only answer with a language `NLP_MODELS` names, because an answer with no
pipeline behind it is a passage nothing can read. `NLP_DEFAULT_LANGUAGE` is
what a passage too short to judge falls back to.

### One call, one judgement

[`entailment.py`](entailment.py) answers one question about one premise and
one hypothesis; a caller wanting two makes two calls.

Two kinds of head are accepted and each score is read off the **label the
model put beside it**, never off a position: a three-way NLI model naming
entailment/neutral/contradiction, and a two-way zero-shot head naming
entailment/not_entailment. mDeBERTa-xnli and bart-mnli are ordered opposite
ways round, so assuming an order inverts every verdict on half the models
anybody would configure.

`transformers`' own text-classification pipeline hands back the label names,
so what is left here is the one thing it does not do: refuse a checkpoint
whose labels are not an NLI model's. A sentiment head scores every pair
fluently and means nothing by it.

The two-way heads are wanted rather than tolerated. `bge-m3-zeroshot-v2.0`
reads 8,192 tokens where `mDeBERTa-v3-base-xnli` reads 512, and a premise here
is a passage already sized to 512.

What the encoder returns is a **probability**, so a deployment can decide how
sure it wants a gate to be. `NLI_ENTAILMENT_THRESHOLD` is that decision, and
`QA_ANSWER_CONFIDENCE` is the same for the extractive reader.

[`embedding.py`](embedding.py) is `sentence-transformers`, which reads the
checkpoint's own `1_Pooling` config rather than assuming the mean.

## Adding a language

German and English are configured. Most readings are Universal Dependencies
features every tagset marks; a handful are lemma lists. This is all of it.

### 1. Configure it

| Where | What |
|---|---|
| `NLP_MODELS` | `xx:xx_core_news_md`. Names both the pipeline a language is read with and the languages the detector may answer with |
| `NLP_CAPITALISED_NOUNS` | Add the code **only** if the language writes every noun with a capital |
| `EMBEDDING_MODEL` | Check the new language is one the model covers |

**Medium, not small.** `de_core_news_sm` does not tag a German modal as a
finite verb, which silently changes what `not_atomic` means.

### 2. Add the language to five lemma lists

These are the readings no feature marks. A language missing from one means
that gate **silently never fires** for it — no error, no log.

| List | Where | What goes in it | Example |
|---|---|---|---|
| `_DIVISIONS` | [`question_generation/gates.py`](../question_generation/gates.py) | Words naming a division of a document | `Abschnitt`, `section` |
| `_DOCUMENTS` | same | Words naming a document **type**, never a subject | `Richtlinie`, `guideline` |
| `_ATTRIBUTIONS` | same | Words that attribute what follows to a source | `laut`, `according` |
| `_AGENTS` | same | The interrogatives that ask after a **party** rather than a thing | `wer`, `who` |
| `_ANAPHORIC` | [`analysis.py`](analysis.py) | Adjectives pointing back at something already said | `besagt`, `aforementioned` |

Read by **lemma and never as a substring**: `Risikobericht` is a thing a
corpus is about and `Bericht` is a thing a corpus IS.

`_AGENTS` could not be a feature — no tagset marks animacy on an
interrogative.

### 3. Check three things the language has to support

Not edits. A language failing one of these needs code:

| | Needed by | Without it |
|---|---|---|
| `noun_chunks` | `phrases()`, and so `subject`, `anchored`, `enumerates`, `compares` | **Raises.** Russian, Chinese and Ukrainian have no syntax iterator; Japanese needs SudachiPy |
| An NER component with `PER` or `PERSON` | `names_parties`, the credits-page reading | Density is always 0, so the reading never fires and nothing says so |
| `like_num` on spelled-out numerals | `claim().units` | Asymmetric: `de_core_news_md` tags `zwei` `like_num` FALSE where `en_core_web_md` gives `two` both |

### 4. One pattern that is Gregorian

`_DATED` in [`question_generation/gates.py`](../question_generation/gates.py)
reads a period off a four-digit year in 1800–2099 or a `DD.MM`-style pair. It
misses an era year, a Hijri year, Arabic-Indic digits and `FY24`. Only
`temporal` questions read it, and that type ships weighted `0`.

### What needs no edit

| Reading | Feature |
|---|---|
| Question words, for `compound` | `PronType=Int`, or a tag prefix as a fallback |
| Content words and lemmas | `NOUN`, `PROPN`, `ADJ` |
| A number, for the circularity exemption | `NUM` |
| Pointing outward | `PronType=Dem`, and `Definite=Def` with `NumType=Card` |
| Everything measured on an embedding | `EMBEDDING_MODEL`, which is multilingual |
| Every length, share and citation reading | no language at all |

## Configuration

Only the first three are read here. The rest are the caller's, and all are
catalogued under `platform`.

| Setting | Default | What it does |
|---|---|---|
| `NLP_MODELS` | `de:de_core_news_md,en:en_core_web_md` | The spaCy pipeline per language, **and** the languages the detector may answer with |
| `NLP_DEFAULT_LANGUAGE` | `en` | What a passage too short to judge is read as |
| `NLP_CAPITALISED_NOUNS` | `de` | Languages where capitalisation is not evidence of a proper noun |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | The one embedding model, and the tokenizer chunking sizes a passage by |
| `NLI_MODEL` | `MoritzLaurer/bge-m3-zeroshot-v2.0` | The entailment encoder. Absent asks the verifier instead |
| `NLI_ENTAILMENT_THRESHOLD` | `0.7` | How sure it must be before it rescues an answer |
| `QA_MODEL` | unset | The extractive reader asked for a span before the verifier is |
| `QA_ANSWER_CONFIDENCE` | `0.9` | How sure that reader must be before its span is taken |
| `ENCODER_MAX_TOKENS` | `8192` | The longest pair an encoder reads. A **ceiling** |

Changing `NLP_MODELS` changes what a fact is, in the same way changing the
prompt does. Both are recorded on every fact — `spacy_model`,
`spacy_version`, `extraction_model`, `prompt_version`.

spaCy and lingua are baked into the backend image because the runtime has no
network. The encoder weights are fetched on first use into the `models`
volume. A model named here must be in the image.

## Tests

```sh
poetry run pytest tests/unit/nlp -m nlp
```

| File | Covers |
|---|---|
| [`test_sentences.py`](../../tests/unit/nlp/test_sentences.py) | Sentence numbering, which every citation resolves against |
| [`test_claims.py`](../../tests/unit/nlp/test_claims.py) | How many claims a sentence reads as |
| [`test_language.py`](../../tests/unit/nlp/test_language.py) | Reading the language a passage is written in |
| [`test_pointing.py`](../../tests/unit/nlp/test_pointing.py) | What a claim points at, and what it does not |
| [`test_entailment.py`](../../tests/unit/nlp/test_entailment.py) | A verdict read off the label, and the refusal of a head that is not an NLI model's |
| [`test_qa.py`](../../tests/unit/nlp/test_qa.py) | The span decode, the no-answer head, and the window that slides |
| [`test_embedding.py`](../../tests/unit/nlp/test_embedding.py) | The vector, when `EMBEDDING_MODEL` is already cached |

`test_sentences`, `test_claims`, `test_pointing` and `test_entailment` carry
the `nlp` marker, and `make test-fast` excludes them. No encoder weights are
loaded by any of these.

## Limits

- **The small spaCy models change what a fact is.** Use the medium pipelines.
- **The detector cannot answer with a language `NLP_MODELS` does not name.**
  A French passage in a German corpus is read as German or English.
- **Sentence numbering is stored, not recomputed.** Changing `NLP_MODELS`
  after a corpus is chunked leaves every stored citation resolving against
  the old split. `make chunk-revocabulary` re-reads language and lemmas; the
  sentence offsets need `chunk-rerun`.
- **Nothing here calls a served model.**
