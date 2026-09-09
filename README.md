# causal-kg-agent

An LLM-provider-agnostic agent that answers questions about biomedical
literature using **two retrieval mechanisms side by side**, and picks
between them per question:

```
PubMed abstracts ──► LLM extraction ──► Neo4j causal graph
  (real, ~36 docs)     (Pydantic-        (Entity)-[INCREASES|DECREASES|
                        validated,         CAUSES|TREATS|TARGETS|
                        cited)             ASSOCIATED_WITH]->(Entity)
        │
        └──► sentence-transformer embeddings ──► local vector store (RAG)

Question ──► Agent ──┬──► search_abstracts()   [plain RAG: vector similarity]
                      └──► traverse_graph()      [KG: multi-hop Cypher]
                             │
                             ▼
                      Answer, every claim cited by PMID —
                      citation checked against real tool
                      output IN CODE, not just prompted
```

Demo topic: **GLP-1 receptor agonists** (semaglutide, tirzepatide, etc.) —
mechanism-of-action and target-validation questions over real PubMed
abstracts, not synthetic data.

## What it does

Ask it a question about GLP-1 receptor agonist literature, and the agent
decides for itself which retrieval mechanism fits:

- **"What does the literature say about digital engagement and semaglutide
  persistence for weight loss?"** — a single-document lookup question, so
  the agent calls `search_abstracts()`, a plain vector-similarity search
  over cached abstract embeddings.
- **"What downstream conditions could treating obesity with GLP-1 receptor
  agonists be associated with?"** — a relational, multi-hop question no
  single abstract states outright, so the agent calls `traverse_graph()`,
  running a Cypher traversal over the causal knowledge graph and returning
  typed, directional, multi-hop paths.

Every answer cites its sources as `(PMID: 12345678)`, and that citation is
verified in code (see below) — not just requested in the prompt.

## Why two retrieval mechanisms, not just RAG

Plain vector search answers "what does the literature say about X" well,
but can't chain relationships across documents ("what downstream effect
does inhibiting protein X have on disease Y") — no single chunk of text
states that. The knowledge graph captures explicit structured relationships
and can traverse multiple hops. The agent has both tools and decides which
fits the question.

## LLM-provider-agnostic by design

Every call site talks to `causal_kg.llm.base.LLMClient`, never to a
provider SDK directly (see `src/causal_kg/llm/`). `OpenAIClient` is the
concrete implementation used here; swapping providers means writing one
new class and changing one construction site (`webapp/main.py`,
`scripts/build_pipeline.py`) — nothing else changes.

## Guardrail: citation-health is code, not a prompt instruction

The agent's system prompt asks it to cite every claim with `(PMID: ...)`.
That's a *soft* instruction — models can still hallucinate a citation.
`agent/loop.py` therefore extracts every cited PMID from the model's final
answer with a regex and checks it in code against the PMIDs the tool calls
*in that conversation* actually returned. A fabricated or borrowed citation
is rejected and the model is told exactly why and asked to retry (up to 2
times), then the system refuses outright rather than presenting an
unverifiable answer.

## How the data is built

```
PubMed E-utilities (free, no key)  ──►  data/abstracts.json  (cached)
        │
        ├──►  sentence-transformers embeddings  ──►  data/embeddings.npz
        │
        └──►  LLM extraction (schema-constrained,   ──►  data/relations.json
               PMID stamped from source abstract         (cached)
               in code, not trusted from the model)
                        │
                        ▼
                Neo4j (idempotent MERGE load)
```

`scripts/build_pipeline.py` orchestrates all of this and is idempotent —
every step checks its cache file first, so re-running costs nothing once
the caches exist.

## Requirements

- Docker (for Neo4j)
- Python 3.11+
- An **OpenAI API key** — required. PubMed ingestion is free, but relation
  extraction and the agent itself both call the OpenAI API, so nothing
  past ingestion runs without one.

## Quickstart

```bash
cp .env.example .env               # fill in OPENAI_API_KEY (required — see Requirements)
docker compose up -d               # Neo4j on 7475 (browser) / 7688 (bolt)
                                    # — deliberately NOT the default ports,
                                    # to not clash with other local demo repos
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/build_pipeline.py   # ingest -> extract -> load graph -> build index
uvicorn causal_kg.webapp.main:app --reload --port 8001 --app-dir src
pytest tests/ -v
```

**If you're on a machine behind a corporate TLS-inspecting proxy**, set
`SSL_CERT_FILE` to a combined CA bundle before running the pipeline or
webapp — see "Known limitations" below for why.

### Step by step

1. **Clone and configure**
   ```bash
   git clone git@github.com:Alejandro-86/causal-kg-agent.git
   cd causal-kg-agent
   cp .env.example .env
   ```
   Fill in `OPENAI_API_KEY` in `.env` — **required**, the pipeline's
   extraction step and the agent itself both call the OpenAI API (never
   commit `.env`, it's gitignored).

2. **Start Neo4j**
   ```bash
   docker compose up -d
   ```
   Requires Docker (Docker Desktop or equivalent) with Compose bundled —
   most installs already include it. Runs on `localhost:7475` (browser
   UI) / `localhost:7688` (bolt) — non-default ports, chosen to avoid
   clashing with other local Neo4j containers.

3. **Create a virtualenv and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

4. **Build the pipeline** (ingest → extract → load graph → build index)
   ```bash
   python scripts/build_pipeline.py
   ```
   Ingestion pulls real abstracts from PubMed (free, no key needed) the
   first time, then caches to `data/abstracts.json`. Extraction calls the
   LLM once per abstract to build `data/relations.json`, then loads it
   into Neo4j. Both steps are skipped on subsequent runs if their cache
   file already exists — only the Neo4j load re-runs (it's an idempotent
   `MERGE`, so no duplicates).

5. **Run the webapp**
   ```bash
   uvicorn causal_kg.webapp.main:app --reload --port 8001 --app-dir src
   ```
   Open `http://localhost:8001`. Ask a question in the chat panel; watch
   the graph panel highlight the nodes/edges a graph-traversal answer
   actually used.

6. **Run the tests** (fully mocked, no network/API calls needed)
   ```bash
   pytest tests/ -v
   ```

7. **(Optional) Inspect the graph directly** in Neo4j Browser at
   `http://localhost:7475`:
   ```cypher
   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
   ```

8. **(Optional) Run the demo questions directly against the agent**,
   bypassing the webapp, with full tool-call tracing printed to the
   terminal:
   ```bash
   python scripts/demo_questions.py
   ```

## Known limitations

- **Citation regex only captures the first PMID in a grouped citation**
  like `(PMID: 123, 456)` — it silently misses `456`. A real, known gap,
  left as-is deliberately rather than risk destabilizing a working,
  tested system. Fix: capture and split on comma-separated digits, not
  just the first.
- **No fallback-to-parametric-knowledge guardrail yet.** The code-level
  citation check catches *fabricated* PMIDs, but doesn't yet catch a
  model silently answering from its own general knowledge with zero
  citations when tool results came back empty or irrelevant. A natural
  next guardrail would reject any final answer with zero citations when
  a citation-worthy claim is made.
- **36 abstracts, not literature-scale.** This is a demonstration of the
  pipeline shape (extraction → graph → grounded retrieval), not a claim
  of production scale. No external vector DB is used deliberately, since
  the corpus is small by design.
- **Tool choice isn't perfectly deterministic** — the model occasionally
  picks `search_abstracts` for a question the graph could also answer
  directly. Expected LLM non-determinism in tool selection, not a code
  defect.
- **Entity types are LLM-assigned, open-vocabulary**, not a curated
  ontology — a deliberate simplicity choice for a small demo corpus, not
  a claim of ontology-engineering depth.
- **A real TLS fix was needed to call the OpenAI API at all on some
  machines:** the `openai` SDK's vendored HTTP client can ship a CA
  bundle missing a root needed to validate `api.openai.com`'s current
  cert chain. Fixed in `llm/openai_client.py` by explicitly using the
  top-level `certifi` package's bundle, merging in `SSL_CERT_FILE` if
  set for genuine corporate-proxy environments.

## Testing

`pytest tests/ -v` — both test files are fully mocked, zero network calls,
no API key needed. `test_citation_guardrail.py` covers the fabricated-vs-
real PMID logic with mock tool results.
