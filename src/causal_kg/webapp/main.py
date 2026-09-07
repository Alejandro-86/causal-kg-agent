"""FastAPI app: /ask wired to the real agent, / serves the demo page."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from causal_kg.agent.loop import CausalGraphAgent
from causal_kg.graph.neo4j_repo import Neo4jCausalGraph
from causal_kg.llm.openai_client import OpenAIClient
from causal_kg.retrieval.vector_store import VectorStore

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="causal-kg-agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_vector_store = VectorStore()
_vector_store.load("data/embeddings.npz")

_graph = Neo4jCausalGraph(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
)

_llm = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
_agent = CausalGraphAgent(llm=_llm, vector_store=_vector_store, graph=_graph)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/graph")
def graph_dump() -> dict:
    return _graph.all_nodes_and_edges()


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return _agent.ask(request.question)
