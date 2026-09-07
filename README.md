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

Built as a live demo topic: **GLP-1 receptor agonists** (semaglutide,
tirzepatide, etc.) — mechanism-of-action / target-validation questions,
mirroring the kind of biomedical knowledge-graph work this project is
built to speak to in an interview context.

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
`scripts/build_pipeline.py`) — nothing else changes. Same pattern as the
`financial-qa-agent` portfolio project.

## Guardrail: citation-health is code, not a prompt instruction

The agent's system prompt asks it to cite every claim with `(PMID: ...)`.
That's a *soft* instruction — models can still hallucinate a citation.
`agent/loop.py` therefore extracts every cited PMID from the model's final
answer with a regex and checks it in code against the PMIDs the tool calls
*in that conversation* actually returned. A fabricated or borrowed citation
is rejected and the model is told exactly why and asked to retry (up to 2
times), then the system refuses outright rather than presenting an
unverifiable answer. Same "prompt = please, code = guarantee" philosophy as
the read-only, citation-grounded MCP servers already in production use.

## Quickstart

```bash
cp .env.example .env               # fill in OPENAI_API_KEY
make up                            # Neo4j on 7475 (browser) / 7688 (bolt)
                                    # — deliberately NOT the default ports,
                                    # to not clash with other local demo repos
make install
make pipeline                      # ingest -> extract -> load graph -> build index
make run                           # http://localhost:8001
make test
```

### Cold-start runbook for demo day

```bash
cd ~/IdeaProjects/causal-kg-agent
docker compose up -d
source .venv/bin/activate
export PYTHONPATH=src
python scripts/build_pipeline.py   # only re-runs steps whose cache is missing
uvicorn causal_kg.webapp.main:app --reload --port 8001 --app-dir src
# open http://localhost:8001
# Neo4j Browser (optional, for a live visual graph): http://localhost:7475
#   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

## Status as of 2026-09-07 — verified, real, not simulated

- **Ingestion:** real, ran end-to-end. Fetched **36 real PubMed abstracts**
  from NCBI E-utilities across 5 GLP-1-related search terms, deduped by
  PMID, cached to `data/abstracts.json`.
- **Vector index:** real, ran end-to-end. `all-MiniLM-L6-v2` embeddings
  built over all 36 abstracts via `sentence-transformers`, saved to
  `data/embeddings.npz`. Self-tested with 3 sample queries — retrieval is
  topically on-target but not laser-precise at this corpus size (see
  caveats below).
- **Neo4j:** confirmed reachable and queryable on `bolt://localhost:7688`
  (currently empty — graph loading depends on extraction).
- **Tests:** `pytest tests/ -v` → **7/7 passed**, no network/API key
  required — Pydantic model validation + the citation-guardrail logic
  (fabricated-PMID rejection and valid-PMID pass-through), fully mocked.
- **Blocked on `OPENAI_API_KEY`:** extraction (`data/relations.json`),
  graph loading, and the agent's live `/ask` behavior. Everything for
  these is code-complete (`extraction/extractor.py`, `agent/loop.py`,
  `webapp/`) but has not executed against a real OpenAI call yet.

### Next steps — once `OPENAI_API_KEY` is in `.env`

```bash
source .venv/bin/activate && export PYTHONPATH=src
python scripts/build_pipeline.py   # will now also run extraction + graph load
make run
```
Then rehearse questions like:
- "What does semaglutide do to gastric emptying?" (RAG path)
- "What downstream effects does GLP-1 receptor activation have?" (KG path)

## Honest caveats

- **36 abstracts vs Biorelate's 50M+ documents** — this is a miniature,
  built to demonstrate the pipeline shape (extraction → graph → grounded
  retrieval), not a claim of production scale. Say this upfront if asked.
- **Retrieval precision at this corpus size:** the sample self-test above
  returned one topically-adjacent-but-not-precise result for a gastric
  emptying query, because the underlying abstract set (skewed toward
  clinical-management papers, not pure pharmacology) doesn't cover every
  mechanism angle deeply — an honest, expected consequence of a 36-document
  corpus rather than a bug. Worth having this framing ready.
- **Neo4j depth:** entity extraction is LLM-assigned, open-vocabulary types
  (not a curated ontology) — a deliberate simplicity choice for a demo,
  not a claim of ontology-engineering depth.
