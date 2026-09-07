"""Ad-hoc script: run real questions through the live agent, print full
tool-call traces. Used to verify the agent for real before the interview,
not part of the pipeline."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from causal_kg.agent.loop import CausalGraphAgent
from causal_kg.graph.neo4j_repo import Neo4jCausalGraph
from causal_kg.llm.openai_client import OpenAIClient
from causal_kg.retrieval.vector_store import VectorStore

load_dotenv()

QUESTIONS = [
    "What does the literature say about digital engagement and semaglutide persistence for weight loss?",
    "What downstream conditions could treating obesity with GLP-1 receptor agonists be associated with?",
    "What effect do GLP-1 receptor agonists have on insulin secretion?",
]

if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.load("data/embeddings.npz")

    graph = Neo4jCausalGraph(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    )
    llm = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    agent = CausalGraphAgent(llm=llm, vector_store=vector_store, graph=graph)

    for q in QUESTIONS:
        print("=" * 90)
        print(f"Q: {q}")
        result = agent.ask(q)
        print(f"\nANSWER: {result['answer']}")
        print(f"\nCITATIONS: {result['citations']}")
        print(f"GRAPH_TOUCHED: {result['graph_touched']}")
        print()

    graph.close()
