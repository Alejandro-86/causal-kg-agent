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

**If you're on a machine behind a corporate TLS-inspecting proxy** (this one
is): `export SSL_CERT_FILE=~/combined-certs.pem` before running the
pipeline or webapp — see "Rough edges" below for why.

### Cold-start runbook for demo day — verified end-to-end 2026-09-07

This exact sequence was actually run from a cold container restart
(`docker compose down` including the network, then back up) and reproduced
the identical graph (235 nodes / 177 relationships) with zero new API
calls, since extraction is cached to disk:

```bash
cd ~/IdeaProjects/causal-kg-agent
docker compose up -d
source .venv/bin/activate
export PYTHONPATH=src
export SSL_CERT_FILE=~/combined-certs.pem   # corporate proxy machines only
python scripts/build_pipeline.py   # only re-runs steps whose cache is missing
uvicorn causal_kg.webapp.main:app --reload --port 8001 --app-dir src
# open http://localhost:8001
# Neo4j Browser (optional, for a live visual graph): http://localhost:7475
#   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

## Status as of 2026-09-07 — fully verified end-to-end, real API calls

- **Ingestion:** 36 real PubMed abstracts, cached to `data/abstracts.json`.
- **Vector index:** real `all-MiniLM-L6-v2` embeddings over all 36
  abstracts, cached to `data/embeddings.npz` (gitignored, regenerate via
  `make pipeline`).
- **Extraction (real OpenAI calls, gpt-4o-mini):** 36/36 abstracts
  processed, **178 relations extracted**, zero retries needed, zero
  fabricated citations found on manual spot-check of 5 random relations
  (4 exact substring matches against source text, 1 accurate light
  paraphrase — verified by hand against the source abstract).
- **Neo4j:** loaded and confirmed via direct Cypher count: **235 nodes,
  177 relationships** (one pair of relations MERGEd onto the same edge —
  expected, idempotent-loader behavior). Relationship type breakdown:
  CAUSES 43, INCREASES 33, ASSOCIATED_WITH 32, TREATS 26, TARGETS 22,
  DECREASES 21.
- **Tests:** `pytest tests/ -v` → **7/7 passed**, both before and after
  the real extraction/graph-load run (no regressions).
- **Live agent, 3 real questions run end-to-end** (see `scripts/demo_questions.py`):
  1. *"What does the literature say about digital engagement and
     semaglutide persistence for weight loss?"* → correctly called
     `search_abstracts` (RAG), answered with 1 real citation.
  2. *"What downstream conditions could treating obesity with GLP-1
     receptor agonists be associated with?"* → correctly called
     `traverse_graph` (KG, multi-hop), answered citing 2 real PMIDs,
     `graph_touched` populated with 16 real nodes / 25 real edges for
     the webapp's visual highlight.
  3. *"What effect do GLP-1 receptor agonists have on insulin
     secretion?"* → see "Rough edges" below, this one surfaced a real
     gap worth knowing before the interview.
- **Webapp:** `GET /` → 200, `GET /graph` → real 235/177 graph dump,
  `POST /ask` → real grounded answer with a real citation. Full HTTP
  round-trip confirmed via curl.
- **Cost:** ~45 real OpenAI calls total across extraction + agent runs +
  ad-hoc checks, all on `gpt-4o-mini`. Consistent with the original
  estimate — comfortably under $0.05 for everything run so far, nowhere
  near the $25 credit.

## Rough edges to know before the interview

- **A real fix was needed to run OpenAI calls on this machine at all:**
  the `openai` SDK's vendored HTTP client ships its own internal CA
  bundle, which was missing a root needed to validate `api.openai.com`'s
  current cert chain — independent of whether an actual corporate proxy
  is in the path that day. Fixed in `llm/openai_client.py` by explicitly
  using the top-level `certifi` package's bundle (and merging in
  `SSL_CERT_FILE` if present, for genuine proxy environments). Good
  incidental talking point: shows you debug from first principles
  (isolated it down to "which HTTP client, which CA bundle" via curl vs
  openssl vs raw httpx before touching code) rather than guessing.
- **A real, honest grounding gap, found by actually running the agent:**
  question 3 above got retrieval results that were all topically
  adjacent but none specifically addressed insulin-secretion mechanism.
  The model correctly said "no specific information was found in the
  search" — genuinely true, verified by hand-checking the 5 retrieved
  abstracts — but then answered anyway from its own general knowledge,
  uncited. The **hard, code-level citation guardrail correctly did not
  fire** (no fabricated PMID was cited, so there was nothing to reject)
  — but the *softer* system-prompt instruction ("say so explicitly
  rather than guessing") was only half-followed. This is a real
  "prompt = please, code = guarantee" gap: catching *fabricated*
  citations is enforced in code; catching *silent fallback to
  parametric knowledge with zero citations* is not yet. A natural
  next guardrail (not yet built) would reject any final answer with
  zero citations when tool results were returned but a citation-worthy
  claim is made. Good to have this ready if the interviewer probes deeper.
- **36 abstracts vs Biorelate's 50M+ documents** — this is a miniature,
  built to demonstrate the pipeline shape (extraction → graph → grounded
  retrieval), not a claim of production scale. Say this upfront if asked.
- **Tool-choice isn't perfectly consistent:** one `/ask` question about
  cardiovascular events used `search_abstracts` even though the exact
  relationship existed in the graph (seen in question 2's traversal).
  Expected LLM non-determinism in tool selection, not a code defect —
  worth knowing so it doesn't look like a bug live.
- **Neo4j depth:** entity extraction is LLM-assigned, open-vocabulary
  types (not a curated ontology) — a deliberate simplicity choice for a
  demo, not a claim of ontology-engineering depth.
