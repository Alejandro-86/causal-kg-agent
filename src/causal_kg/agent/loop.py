"""Tool-calling agent with two retrieval mechanisms and a hard,
code-level citation guardrail.

Two tools, two different retrieval shapes:
  - search_abstracts  -> plain RAG (vector similarity over abstract text).
                         Good for "what does the literature say about X".
  - traverse_graph    -> knowledge-graph traversal (Cypher multi-hop).
                         Good for relational/multi-hop questions no single
                         chunk of text states explicitly.

The model decides which tool(s) fit the question. After it produces a
final answer, every PMID it cites is checked in code against the PMIDs
actually returned by tool calls made *in this conversation* — a fabricated
or borrowed citation is not a "please don't do that" prompt instruction,
it is a rejected answer, regenerated up to twice, then a refusal.
"""

import re

from causal_kg.graph.neo4j_repo import Neo4jCausalGraph
from causal_kg.llm.base import LLMClient
from causal_kg.retrieval.vector_store import VectorStore

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_abstracts",
            "description": (
                "Semantic search over biomedical abstract text. Use for "
                "open-ended questions about what the literature says."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traverse_graph",
            "description": (
                "Traverse the causal knowledge graph from a named entity "
                "(drug, protein, gene, disease, ...) up to max_hops away. "
                "Use for relational/multi-hop questions, e.g. 'what "
                "downstream effects does inhibiting protein X have'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "max_hops": {"type": "integer", "default": 2},
                },
                "required": ["entity_name"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a biomedical research assistant. Answer only using
information returned by your tools. Cite every claim with its PMID in
the form (PMID: 12345678). If your tools return nothing relevant, say so
explicitly rather than guessing."""

PMID_PATTERN = re.compile(r"PMID:\s*(\d+)")


class CausalGraphAgent:
    def __init__(self, llm: LLMClient, vector_store: VectorStore, graph: Neo4jCausalGraph) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._graph = graph

    def _execute_tool(self, name: str, arguments: dict) -> tuple[str, dict]:
        """Run a tool call, return (content string for the LLM, structured
        data for the caller to inspect — e.g. which graph nodes/edges were
        touched, for the webapp's visualization)."""
        if name == "search_abstracts":
            results = self._vector_store.search(arguments["query"])
            content = "\n".join(
                f"- ({r['pmid']}) {r['title']}: {r['text'][:300]}" for r in results
            )
            return content, {"kind": "abstracts", "results": results}

        if name == "traverse_graph":
            paths = self._graph.traverse(
                arguments["entity_name"], arguments.get("max_hops", 2)
            )
            content = "\n".join(
                f"- {' -> '.join(p['node_chain'])} via "
                f"{[hop['type'] for hop in p['rel_chain']]} "
                f"(PMIDs: {[hop['pmid'] for hop in p['rel_chain']]})"
                for p in paths
            ) or "No paths found."
            return content, {"kind": "graph", "paths": paths}

        raise ValueError(f"unknown tool: {name}")

    def ask(self, question: str, max_retries: int = 2) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        seen_pmids: set[str] = set()
        graph_touched: dict = {"nodes": set(), "edges": []}

        for _ in range(6):  # hard cap on tool-call rounds
            result = self._llm.complete_with_tools(messages, TOOLS)

            if not result.is_tool_call:
                answer = result.text or ""
                cited = set(PMID_PATTERN.findall(answer))
                fabricated = cited - seen_pmids

                if fabricated and max_retries > 0:
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You cited PMID(s) {sorted(fabricated)} that were "
                                "not returned by any tool call. Cite only PMIDs "
                                "from the tool results above, or say the "
                                "information isn't available."
                            ),
                        }
                    )
                    max_retries -= 1
                    continue

                if fabricated:
                    answer = (
                        "I can't verify all citations in my draft answer against "
                        "retrieved sources, so I won't present it. "
                        "Try rephrasing the question."
                    )

                return {
                    "answer": answer,
                    "citations": sorted(seen_pmids & cited) if not fabricated else [],
                    "graph_touched": {
                        "nodes": sorted(graph_touched["nodes"]),
                        "edges": graph_touched["edges"],
                    },
                }

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": str(tc.arguments)},
                        }
                        for tc in result.tool_calls
                    ],
                }
            )
            for tc in result.tool_calls:
                content, data = self._execute_tool(tc.name, tc.arguments)
                if data["kind"] == "abstracts":
                    seen_pmids.update(r["pmid"] for r in data["results"])
                elif data["kind"] == "graph":
                    for path in data["paths"]:
                        graph_touched["nodes"].update(path["node_chain"])
                        for i, hop in enumerate(path["rel_chain"]):
                            seen_pmids.add(hop["pmid"])
                            graph_touched["edges"].append(
                                {
                                    "source": path["node_chain"][i],
                                    "target": path["node_chain"][i + 1],
                                    "type": hop["type"],
                                }
                            )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": content}
                )

        return {
            "answer": "Couldn't reach a grounded answer within the tool-call budget.",
            "citations": [],
            "graph_touched": {"nodes": [], "edges": []},
        }
