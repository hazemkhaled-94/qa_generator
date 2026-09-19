# Frontend

A Streamlit application: one page per stage, and a page runs that stage and no
other.

It holds **one address** — `BACKEND_URL` — and no knowledge of the database,
the object store, or the services behind the API. Everything it shows, it
asked the API for.

## The pages

| Page | Lists | Runs | Over | Configures |
|---|---|---|---|---|
| Upload | — | ingestion | the files you choose | ingestion |
| Documents | documents | parsing | a document | parsing |
| Passages | passages | chunking | a document | chunking |
| Facts | facts | extraction | a passage | extraction |
| Topics | topics | topic modelling | the whole corpus | topics |
| Questions | questions | question generation | a topic | questions |
| System health | components | nothing | — | the platform all six share |

A page lists what its stage produces and runs the stage that produced it.
**Nothing on a page can reach another page's stage.**

The queue's unit is not always the row: chunking replaces all of a document's
passages at once, and extraction reads a passage and writes all of its facts
together, so a row picked on those pages is run through the thing its stage
actually queues over. The page says so where it offers the control.

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

The configuration is where it is because **the remedy for changing it is
directly above it**: a setting that stales what the stage already produced is
rebuilt by the Redo button in the same panel.

Its controls are drawn from what the API says each setting is, so the page
holds no list of settings — adding one to
[`settings/catalog.py`](../backend/settings/catalog.py) adds it to the page.

Two pages carry a delete box, because two things can be deleted: a document,
with or without what was derived from it, and the topics, all at once. A
passage, a fact and a question have no delete — a passage belongs to its
document, and a rejected question is kept because the share that was thrown
away is the evidence behind the coverage report.

### While a stage is running

Each stage can be run over everything it owns or over the one row picked. **No
verb does the work**: they move rows between statuses, and a worker picks up
what became claimable.

While a stage has anything queued or in hand, its panel carries a spinner and
its counts, and redraws every few seconds until the queue is empty. The page
stays usable throughout, and Stop stays live.

## Layout

| Path | Holds |
|---|---|
| [`app.py`](app.py) | The entry point: `st.navigation` over the views |
| [`views/`](views/) | One module per page, run top to bottom on every interaction |
| [`lib/page.py`](lib/page.py) | The shared layout every view calls |
| [`lib/stage.py`](lib/stage.py) | The one service a page runs, and the controls that move its queue |
| [`lib/catalog.py`](lib/catalog.py) | The search panel, the filter panel and the pager they share |
| [`lib/configure.py`](lib/configure.py) | The configuration panel, beside the controls that run the service |
| [`lib/backend/`](lib/backend/) | The API clients, one module per group of calls |
| [`styles.css`](styles.css) | The custom styling, on top of the theme |

A view is a **script**, not a component tree. Streamlit runs it top to bottom
on every interaction, which is why the tests run each one the same way.

## Themes and colour

The app comes in light and dark.
[`.streamlit/config.toml`](.streamlit/config.toml) gives Streamlit a palette
under `[theme.light]` and another under `[theme.dark]`; it starts from the
browser's `prefers-color-scheme`, and the toolbar menu switches between them
per page.

Put a colour in `[theme]` itself, or set `theme.base`, and it applies to
**both** themes — which is what pinned this app to light before. Only what
genuinely does not vary belongs there.

[`styles.css`](styles.css) reads its own palette off `light-dark()`, which
resolves against the `color-scheme` Streamlit sets on the app container from
the theme it actually settled on. So the custom styling follows the chrome
whichever way the chrome was decided — a `prefers-color-scheme` media query
would have got the menu wrong, staying light while everything around it went
dark.

Two hues are written twice, the brand purple and the red that only deletion
uses; the neutrals are mixed from `currentColor`, so they need no second
value.

**Colour carries one meaning.** Purple fills the one control that commits
something in a group — Start, Fit, Save, Accept — and red is spent only on
deletion. Everything else is an outlined button, and a control that would do
nothing right now is greyed rather than hidden, so a row keeps its shape.

The topic map stays on white in either theme. It is a pyLDAvis document inside
an iframe, so nothing outside it can restyle it; it is framed and given a
background of its own so it reads as a figure printed on white.

## Configuration

| Setting | Where | What it does |
|---|---|---|
| `BACKEND_URL` | `.env` | The one address. `http://api:8000` from a container |
| `PAGE_SIZE` | `.env` | Rows per page in the listings |
| `LOG_LEVEL` | `.env` | The frontend logs through the same telemetry configuration as every other process |

Both of the first two are **required**, and the error names the trap: being in
`.env` alone is not enough — the variable also has to be listed in the
`streamlit` service's `environment` in `compose.yaml`.

`server.maxUploadSize` in [`.streamlit/config.toml`](.streamlit/config.toml)
must be kept in step with `MAX_FILE_SIZE_MB`, or Streamlit rejects the file
before the API sees it.

## Tests

```sh
poetry run pytest tests/frontend
```

Each view is a script Streamlit runs top to bottom, so an `AppTest` runs it
the same way. **The backend is stubbed**: what these cover is what the page
does with an answer, including the answers that are refusals, which is the
half no integration test reaches.

| File | Covers |
|---|---|
| [`test_views.py`](../tests/frontend/test_views.py) | The rules every page keeps, checked on every page |
| [`test_facts_page.py`](../tests/frontend/test_facts_page.py), [`test_topics_page.py`](../tests/frontend/test_topics_page.py) | Two pages in depth, against a scripted backend |
| [`test_configuration_panel.py`](../tests/frontend/test_configuration_panel.py) | The configuration panel, on the page that runs the service it configures |
| [`pages.py`](../tests/frontend/pages.py) | One page object over every view, and the answers each view needs |
| [`tests/static/test_frontend_vocabularies.py`](../tests/static/test_frontend_vocabularies.py) | That the gate list on the Questions page is exactly what the checker can reject under |

These need no container and no spaCy. `make test-fast` runs them.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A variable in `.env` that is not in `compose.yaml` does not reach the
  frontend.** The error says so by name, because it has caught people.
- **`MAX_FILE_SIZE_MB` and `server.maxUploadSize` are two numbers for one
  limit.** Streamlit enforces its own first, and rejects the file before the
  API is asked.
- **A page that shows a spinner is not doing the work.** It moved rows and is
  polling `/status`. Closing the tab does not stop the worker.
- **Every listing filters server-side** — apart from `/topics`, which returns
  every topic at once, because one fit produces a list a person can read.
- **A rejected question stays visible on purpose.** The share that was thrown
  away is the evidence behind the coverage report.
