# NLP

Sentence splitting, claim analysis, vocabulary and language detection — the
linguistic surface every other stage reads off text.

One of the three packages that load something expensive; the others are
[`database/`](../database/README.md) and
[`blob_store/`](../blob_store/README.md). A caller asks for a sentence split;
it never loads a pipeline.

**No model is SERVED from here.** Everything in this package is local and
deterministic — spaCy, a language detector, and two encoders that run in the
worker's own process. That is what lets the fact checks run without a served
model, and it is why `make extract-revalidate` costs seconds rather than
hours.

Three of the files load weights, and nothing outside a worker may import
them: [`embedding.py`](embedding.py), [`entailment.py`](entailment.py) and
the spaCy pipelines. `tests/static/test_api_stays_light.py` pins that the API
process loads none of them.

| File | Loads | Asked |
|---|---|---|
| [`embedding.py`](embedding.py) | `EMBEDDING_MODEL` | one text's vector, for every column that holds one |
| [`entailment.py`](entailment.py) | `QUESTIONS_ENTAILMENT_MODEL` | whether one premise entails one hypothesis |

## One call, one judgement

[`entailment.py`](entailment.py) answers **one** question about **one**
premise and **one** hypothesis, and a caller wanting two makes two calls. The
thing it replaces is a prompt that asked a served model for four judgements
at once and confused them — see
[question generation](../question_generation/README.md#what-that-one-call-was-carrying-and-where-those-judgements-went).

Two kinds of head are accepted and the label order is read off the
checkpoint's own `id2label`, never assumed: a three-way NLI model naming
entailment/neutral/contradiction, and a two-way zero-shot head naming
entailment/not_entailment. Assuming an order silently inverts every verdict
on half the models anybody would configure.

The two-way heads are wanted rather than tolerated. `bge-m3-zeroshot-v2.0`
reads **8,194 tokens** where `mDeBERTa-v3-base-xnli` reads 512, and a premise
here is a passage this pipeline already sized to 512 of its own — so the
three-way model would truncate exactly the text the judgement rests on. What
a two-way head cannot do is tell "says something different" from "does not
address it", and both are refusals here, so it costs this caller nothing.

What the encoder returns is a **probability**, which is the other reason to
prefer it to a served model answering true or false: a boolean carries no
confidence, so a deployment cannot decide how sure it wants a gate to be.
`QUESTIONS_ENTAILMENT_THRESHOLD` is that decision.

## What it does

### Sentence numbering

Chunking splits every passage into numbered sentences and stores them. A fact
**cites one of those numbers** rather than quoting text, so its source span is
exact by construction: there is no quote to search for, nothing to match
character for character, and no near miss. A citation is either a sentence the
passage has or one it does not.

Everything downstream rests on this numbering being stable. It is computed
once, by chunking, and stored — not recomputed by each reader.

### Counting claims

`not_atomic` — the check that a fact carries exactly one claim — is a
**finite-verb count**. A statement with more than one carries several claims;
one with none carries nothing.

That replaced a similarity threshold, and the replacement is the point.
Counting finite verbs is what "one claim" actually means; measuring how many
words a statement shares with its source rejected genuine narrowings — a
statement that drops a qualifier and keeps the subject's wording scored as a
reword — and let a paraphrase through for swapping a noun.

The same reading is what lets the passage gate skip a heading, a caption or a
navigation line **before** the model is called rather than sending it and
rejecting the answer afterwards. At a median 473 s per passage, that is the
cheapest saving in the pipeline.

### Vocabulary

Lemmas, with the stop words and the punctuation dropped. Written onto every
passage by chunking, and read by the topic model as the document it is fitted
over. A topic fit reads a column instead of re-tokenising the corpus.

`NLP_CAPITALISED_NOUNS` names the languages where capitalisation says nothing
about whether a noun is proper — German, where every noun is capitalised — so
the vocabulary is not cluttered with false proper nouns.

### Language detection

Detected on the **passage**, not inherited from its document. 45 of this
corpus's passages are German inside an English-labelled file, and a
document-wide label read every one of them with the wrong pipeline.

The detector may only answer with a language `NLP_MODELS` names, because an
answer with no pipeline behind it is a passage nothing can read.
`NLP_DEFAULT_LANGUAGE` is what a passage too short to judge falls back to.

## Adding a language

German and English are configured. The pipeline is not bound to them, but it
is not free of them either: most readings are Universal Dependencies features
that every tagset marks, and a handful are lemma lists that only two
languages are in. This is all of it.

### 1. Configure it

| Where | What |
|---|---|
| `NLP_MODELS` | `xx:xx_core_news_md`. One list, naming both the pipeline a language is read with and the languages the detector may answer with. `make install` and the Dockerfile download whatever it names |
| `NLP_CAPITALISED_NOUNS` | Add the code **only** if the language writes every noun with a capital, as German does. In one of those a lower-case word tagged a noun is a word from another language |
| `EMBEDDING_MODEL` | Check the new language is one the model covers. The default, `multilingual-e5-large`, covers about a hundred |

**Medium, not small.** `de_core_news_sm` does not tag a German modal as a
finite verb, so the atomicity check refused 11% of German facts for a parser
limitation: measured over eight sentences, `sm` got 3 and `md` got 8.

### 2. Add the language to five lemma lists

These are the readings no feature marks, so they are words. Each is small,
each is per-language, and a language missing from one means that gate
**silently never fires** for it — no error, no log, just a judgement nothing
makes.

| List | Where | What goes in it | Example |
|---|---|---|---|
| `_DIVISIONS` | [`question_generation/gates.py`](../question_generation/gates.py) | Words naming a division of a document | `Abschnitt`, `section` |
| `_DOCUMENTS` | same | Words naming a document **type**, never a subject | `Richtlinie`, `guideline` |
| `_ATTRIBUTIONS` | same | Words that attribute what follows to a source | `laut`, `according` |
| `_AGENTS` | same | The interrogatives that ask after a **party** rather than a thing | `wer`, `who` |
| `_ANAPHORIC` | [`analysis.py`](analysis.py) | Adjectives pointing back at something already said | `besagt`, `aforementioned` |

Read by **lemma and never as a substring**, and German compounding is why:
`Risikobericht` is a thing a corpus is about and `Bericht` is a thing a corpus
IS, and a substring test cannot tell them apart.

`_AGENTS` is the one that could not be a feature. No tagset marks animacy on
an interrogative — `wer` and `was` are both `PRON` with `PronType=Int`, as
`who` and `what` are — so a list is the only reading available.

### 3. Check three things the language has to support

Not edits. A language failing one of these needs code, and the first is a
crash rather than a gap:

| | Needed by | What happens without it |
|---|---|---|
| **`noun_chunks`** | `phrases()`, and so `subject`, `anchored`, `enumerates`, `compares` | **Raises.** spaCy implements the syntax iterator per language and Russian, Chinese and Ukrainian have none; Japanese needs SudachiPy |
| **An NER component with `PER` or `PERSON`** | `names_parties`, the credits-page reading | Density is always 0, so the reading never fires and nothing says so |
| **`like_num` on spelled-out numerals** | `claim().units` | Asymmetric rather than broken: `de_core_news_md` tags `zwei` as `NUM` with `like_num` FALSE where `en_core_web_md` gives `two` both, so `units` collects English number words and not German ones |

That third one is a trap rather than a bug, and it already bit: the
circularity gate first read numbers off `claim().units` and therefore fired
on German numerals and not English ones, in a corpus that is German. It reads
the `NUM` part of speech now.

### 4. One pattern that is Gregorian

`_DATED` in [`question_generation/gates.py`](../question_generation/gates.py)
reads a period off a four-digit year in 1800–2099 or a `DD.MM`-style pair. It
matches `2024`, `2024-03-15`, `15.03.2024` and `Q1 2024`, and misses an era
year, a Hijri year, Arabic-Indic digits and `FY24`. Only `temporal` questions
read it, and that type ships weighted `0`.

### What needs no edit at all

Most of it, because most readings are features rather than words:

| Reading | Feature |
|---|---|
| Question words, for `compound` | `PronType=Int`, or a Penn/STTS tag prefix as a fallback |
| Content words and lemmas | `NOUN`, `PROPN`, `ADJ` |
| A number, for the circularity exemption | `NUM` |
| Pointing outward | `PronType=Dem`, and `Definite=Def` with `NumType=Card` for the definite-and-counted case |
| Everything measured on an embedding | `EMBEDDING_MODEL`, which is multilingual |
| Every length, share and citation reading | no language at all |

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **spaCy** | [`pipelines.py`](pipelines.py), [`analysis.py`](analysis.py) | Sentence boundaries, the POS tags the finite-verb count reads, and the lemmas. One pipeline per language, loaded once per process |
| **lingua** | [`language.py`](language.py) | Answers from a closed set of languages and is confident on short text, which a passage often is |

Both are baked into the backend image, because the runtime has no network.

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `NLP_MODELS` | `de:de_core_news_md,en:en_core_web_md` | The spaCy pipeline per language, **and** the languages the detector may answer with. One list, so the two cannot disagree |
| `NLP_DEFAULT_LANGUAGE` | `en` | What a passage too short to judge is read as |
| `NLP_CAPITALISED_NOUNS` | `de` | Languages where capitalisation is not evidence of a proper noun |

**Medium, not small.** The small German model does not tag a modal as a finite
verb, which silently changes what `not_atomic` means.

**Changing `NLP_MODELS` changes what a fact is**, in the same way changing the
prompt does. Both are recorded on every fact — `spacy_model`, `spacy_version`,
`extraction_model`, `prompt_version` — so two generations of the dataset can
be told apart.

A model named here must be in the image. `make install` downloads the
pipelines `NLP_MODELS` names for the host as well.

## Tests

```sh
poetry run pytest tests/unit/nlp -m nlp
```

| File | Covers |
|---|---|
| [`test_sentences.py`](../../tests/unit/nlp/test_sentences.py) | Sentence numbering, which every citation resolves against |
| [`test_claims.py`](../../tests/unit/nlp/test_claims.py) | How many claims a sentence reads as |
| [`test_language.py`](../../tests/unit/nlp/test_language.py) | Reading the language a passage is written in |

These carry the `nlp` marker, because they load a pipeline. `make test-fast`
excludes them.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **The small spaCy models change what a fact is.** `de_core_news_sm` does not
  tag a modal as a finite verb, so `not_atomic` starts accepting statements it
  should refuse. Use the medium pipelines.
- **The detector cannot answer with a language `NLP_MODELS` does not name.**
  A French passage in a German corpus is read as German or English, whichever
  it scores higher as. Adding French means adding its pipeline.
- **Sentence numbering is stored, not recomputed.** Changing `NLP_MODELS`
  after a corpus is chunked leaves every stored citation resolving against the
  old split. `make chunk-revocabulary` re-reads language and lemmas; the
  sentence offsets need `chunk-rerun`.
- **Nothing here calls a served model.** If a check seems to need one, it is
  in [`extraction/`](../extraction/README.md) or
  [`question_generation/`](../question_generation/README.md), not here.
