# Frontend

A Streamlit application: one page per stage, and a page runs that stage and no
other.

It holds **one address it calls** — `BACKEND_URL` — and no knowledge of the
database, the object store, or the services behind the API.

`PHOENIX_BASE_URL` is a second address and not a second exception: nothing
here calls it. It is written into a link a **browser** follows, which is why
compose leaves it as the host sees it rather than setting a container name
over it.

## The pages

| Page | Lists | Runs | Over | Configures |
|---|---|---|---|---|
| Upload | — | ingestion | the files you choose | ingestion |
| Documents | documents | parsing | a document | parsing |
| Passages | passages | chunking | a document | chunking |
| Facts | facts | extraction | a passage | extraction |
| Topics | topics | topic modelling | the whole corpus | topics |
| Questions | questions | question generation | a topic | questions |
| Assessment | what an LLM judge said about each artefact | the evaluation phase | one kind of artefact | assessment |
| System health | components, and every container with a link to it | nothing | — | the platform all seven share |

**Nothing on a page can reach another page's stage.**

Assessment is the one page that lists an opinion rather than an artefact,
and the one whose stage can be switched off entirely — see
[`backend/assessment/`](../backend/assessment/README.md).

System health is the one that lists something the pipeline did not produce.
It asks `GET /services` rather than probing anything itself — the topology
puts the api between the browser and everything else.

The queue's unit is not always the row: chunking replaces all of a document's
passages at once, and extraction reads a passage and writes all of its facts
together, so a row picked on those pages is run through the thing its stage
actually queues over. The page says so where it offers the control.

### A picked question links out to what produced it

| | From | Opens |
|---|---|---|
| The calls | `questions.span_id` and `questions.trace_id` | Phoenix. **Two links**: the span is the gate decision, the trace is the whole topic |
| The prompt | `questions.prompt_version` and `question_type`, through `GET /prompts` | The prompt as it was sent, inline — both halves, system then user |

Both degrade rather than break. A question written before the columns existed
says so; a deployment with no `PHOENIX_BASE_URL` on the `streamlit` service
shows the ids as text instead of links.

## The shape of every page

The same sequence of panels, on all seven:

```
the few figures worth seeing on arrival
an `Analysis` fold nobody has to open
the stage's controls, with a `Configuration` fold beside them
the search box
the filters
the table
— only once a row is picked —
everything held about that row, as one table of every field it has
```

The configuration is where it is because the remedy for changing it is
directly above it: a setting that stales what the stage produced is rebuilt
by the Redo button in the same panel.

Its controls are drawn from what the API says each setting is, so the page
holds no list of settings — adding one to
[`settings/catalog.py`](../backend/settings/catalog.py) adds it to the page.

Two pages carry a delete box, because two things can be deleted: a document,
with or without what was derived from it, and the topics, all at once. A
passage, a fact and a question have no delete.

While a stage has anything queued or in hand, its panel carries a spinner and
its counts and redraws every few seconds until the queue is empty. **No verb
does the work**: they move rows between statuses, and a worker picks up what
became claimable. The page stays usable throughout, and Stop stays live.

## Layout

| Path | Holds |
|---|---|
| [`app.py`](app.py) | The entry point: `st.navigation` over the views |
| [`views/`](views/) | One module per page, run top to bottom on every interaction |
| [`lib/page.py`](lib/page.py) | The shared layout every view calls |
| [`lib/stage.py`](lib/stage.py) | The one service a page runs, and the controls that move its queue |
| [`lib/catalog.py`](lib/catalog.py) | The search panel, the filter panel and the pager |
| [`lib/configure.py`](lib/configure.py) | The configuration panel |
| [`lib/backend/`](lib/backend/) | The API clients, one module per group of calls |
| [`styles.css`](styles.css) | The custom styling, on top of the theme |

A view is a **script**, not a component tree. Streamlit runs it top to bottom
on every interaction, which is why the tests run each one the same way.

## Themes and colour

The 2025 brand: `#A100FF` core purple, `#460073` and `#7500C0`
deeps, `#C2A3FF` and `#E6DCFF` lights, `#FF50A0` / `#224BFF` / `#05F2DB`
secondaries, neutral greys. No logo: the wordmark and the `>` chevron are
both the brand marks.

The template's **content** slides are the reference, not its covers. They are
white, with black type and purple carrying the structure: a filled header on
every table, figures set in the deep purple, purple section headings. The
black slides are dividers, which an application does not have — so the app is
light, and the purple is what makes it vivid.

[`.streamlit/config.toml`](.streamlit/config.toml) carries most of it. It
gives Streamlit a palette under `[theme.light]` and another under
`[theme.dark]`, starting from the browser's `prefers-color-scheme` and
switched per page from the toolbar menu. Put a colour in `[theme]` itself, or
set `theme.base`, and it applies to **both**.

The dark theme is a near-black carrying the brand's violet cast rather than
the flat black of a cover slide, and swaps the core purple for the light one,
which a dark ground needs to read an accent at all.

Tables get the template's light-header variant. The grid draws its header
text on a canvas, so no stylesheet can correct that text: the fill has to be
one it already reads on, which rules out the filled deep-purple header the
printed tables use.

Type is Graphik, the brand face. It is licensed and not shipped, so the stack
falls through to Arial — the substitute the brand itself names — and picks up
a locally installed Graphik ahead of it.

[`styles.css`](styles.css) holds what the theme config cannot express: the
masthead, the section headings, the queue state line and the delete box. It
reads its own palette off `light-dark()`, which resolves against the
`color-scheme` Streamlit sets from the theme it settled on — so the custom
styling follows the chrome whichever way the chrome was decided. Two hues are
written twice, the purple and the red that only deletion uses; the neutrals
are mixed from `currentColor`.

**Colour carries one meaning.** Purple fills the one control that commits
something in a group — Start, Fit, Save, Accept — and red is spent only on
deletion. Everything else is an outlined button, and a control that would do
nothing right now is greyed rather than hidden.

Red is not in the palette. It stays because a destructive control should not
be the first place a reader learns what the brand colours mean.

Every colour pair here is at or above 4.5:1 against the ground it sits on, in
both themes. `--qa-muted` is mixed at 65% rather than 58% for that reason.

The topic map stays on white in either theme: it is a pyLDAvis document
inside an iframe, so nothing outside it can restyle it.

Selectors here are tied to Streamlit's DOM, which it renames between
versions. `footer` no longer matches anything.

## Configuration

| Setting | Where | What it does |
|---|---|---|
| `BACKEND_URL` | `.env` | The one address. `http://api:8000` from a container |
| `PAGE_SIZE` | `.env` | Rows per page in the listings |
| `PHOENIX_BASE_URL` | `.env` | Where a **browser** reaches Phoenix. Optional: unset shows the ids as text |
| `LOG_LEVEL` | `.env` | The frontend logs through the same telemetry configuration as every other process |

The first two are **required**, and the error names the trap: being in `.env`
alone is not enough — the variable also has to be listed in the `streamlit`
service's `environment` in `compose.yaml`.

`server.maxUploadSize` in [`.streamlit/config.toml`](.streamlit/config.toml)
must be kept in step with `MAX_FILE_SIZE_MB`.

## Tests

```sh
poetry run pytest tests/frontend
```

Each view is a script Streamlit runs top to bottom, so an `AppTest` runs it
the same way. **The backend is stubbed**: what these cover is what the page
does with an answer, including the answers that are refusals.

| File | Covers |
|---|---|
| [`test_views.py`](../tests/frontend/test_views.py) | The rules every page keeps, checked on every page |
| [`test_facts_page.py`](../tests/frontend/test_facts_page.py), [`test_topics_page.py`](../tests/frontend/test_topics_page.py) | Two pages in depth, against a scripted backend |
| [`test_configuration_panel.py`](../tests/frontend/test_configuration_panel.py) | The configuration panel |
| [`pages.py`](../tests/frontend/pages.py) | One page object over every view, and the answers each view needs |
| [`test_frontend_vocabularies.py`](../tests/static/test_frontend_vocabularies.py) | That the gate list on the Questions page is exactly what the checker can reject under |

These need no container and no spaCy. `make test-fast` runs them.

## Limits

- **A variable in `.env` that is not in `compose.yaml` does not reach the
  frontend.** The error says so by name.
- **`MAX_FILE_SIZE_MB` and `server.maxUploadSize` are two numbers for one
  limit.** Streamlit enforces its own first.
- **A page that shows a spinner is not doing the work.** Closing the tab does
  not stop the worker.
- **Every listing filters server-side** apart from `/topics`, which returns
  every topic at once.
- **A rejected question stays visible on purpose.** The share that was thrown
  away is the evidence behind the coverage report.
