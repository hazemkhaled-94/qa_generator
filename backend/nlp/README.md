# NLP

Sentence splitting, claim analysis, vocabulary and language detection — the
linguistic surface every other stage reads off text.

One of the three packages that load something expensive; the others are
[`database/`](../database/README.md) and
[`blob_store/`](../blob_store/README.md). A caller asks for a sentence split;
it never loads a pipeline.

**No model is called here.** Everything in this package is spaCy and a
language detector, both local and both deterministic. That is what lets the
fact checks run without a served model, and it is why
`make extract-revalidate` costs seconds rather than hours.

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
