"""Orchestrates: ingest -> extract -> load graph -> build vector index.

Idempotent — safe to re-run. Ingestion and vector-index build need no API
key; extraction and graph loading need OPENAI_API_KEY in .env.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_kg.extraction.extractor import extract_relations
from causal_kg.graph.neo4j_repo import Neo4jCausalGraph
from causal_kg.ingestion.pubmed import fetch_glp1_abstracts
from causal_kg.llm.openai_client import OpenAIClient
from causal_kg.models import Abstract, CausalRelation
from causal_kg.retrieval.vector_store import VectorStore

DATA_DIR = Path(__file__).parent.parent / "data"


def step_ingest() -> list[Abstract]:
    path = DATA_DIR / "abstracts.json"
    if path.exists():
        print(f"[ingest] using cached {path}")
        raw = json.loads(path.read_text())
        return [Abstract(**item) for item in raw]

    print("[ingest] fetching from PubMed...")
    abstracts = fetch_glp1_abstracts()
    path.write_text(json.dumps([a.model_dump() for a in abstracts], indent=2))
    print(f"[ingest] fetched {len(abstracts)} abstracts, cached to {path}")
    return abstracts


def step_build_index(abstracts: list[Abstract]) -> None:
    path = DATA_DIR / "embeddings.npz"
    print("[index] building sentence-transformer embeddings...")
    store = VectorStore()
    store.build(abstracts)
    store.save(str(path))
    print(f"[index] saved to {path}")


def step_extract(abstracts: list[Abstract]) -> list[CausalRelation]:
    path = DATA_DIR / "relations.json"
    if path.exists():
        print(f"[extract] using cached {path}")
        raw = json.loads(path.read_text())
        return [CausalRelation(**item) for item in raw]

    print("[extract] calling LLM for each abstract...")
    llm = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    all_relations: list[CausalRelation] = []
    for i, abstract in enumerate(abstracts):
        relations = extract_relations(llm, abstract)
        all_relations.extend(relations)
        print(f"[extract] {i+1}/{len(abstracts)} PMID {abstract.pmid}: {len(relations)} relations")

    path.write_text(json.dumps([r.model_dump() for r in all_relations], indent=2, default=str))
    print(f"[extract] {len(all_relations)} total relations, cached to {path}")
    return all_relations


def step_load_graph(relations: list[CausalRelation]) -> None:
    print("[graph] loading into Neo4j...")
    graph = Neo4jCausalGraph(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    )
    graph.load_relations(relations)
    graph.close()
    print(f"[graph] loaded {len(relations)} relations")


if __name__ == "__main__":
    load_dotenv()
    DATA_DIR.mkdir(exist_ok=True)

    abstracts = step_ingest()
    step_build_index(abstracts)

    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        relations = step_extract(abstracts)
        step_load_graph(relations)
    else:
        print("\n[pipeline] OPENAI_API_KEY not set — stopping before extraction/graph-load.")
        print("[pipeline] set it in .env, then re-run this script.")
