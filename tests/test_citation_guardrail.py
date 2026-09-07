"""Pure-function tests of the code-level citation guardrail in
agent/loop.py — no network calls, no API key. The LLM is faked with a
scripted sequence of responses; the vector store and graph are faked
with fixed return values."""

from causal_kg.agent.loop import CausalGraphAgent
from causal_kg.llm.base import CompletionResult, LLMClient, ToolCall


class FakeLLMClient(LLMClient):
    """Returns canned CompletionResults in sequence, ignoring input."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)

    def complete_structured(self, prompt, schema):
        raise NotImplementedError("not used in this test")

    def complete_with_tools(self, messages, tools) -> CompletionResult:
        return self._responses.pop(0)


class FakeVectorStore:
    def search(self, query: str, k: int = 5) -> list[dict]:
        return [
            {"pmid": "11111111", "title": "Real paper", "text": "Some real evidence.", "score": 0.9}
        ]


class FakeGraph:
    def traverse(self, entity_name: str, max_hops: int = 2) -> list[dict]:
        return []


def _tool_call_response() -> CompletionResult:
    return CompletionResult(
        tool_calls=[ToolCall(id="call_1", name="search_abstracts", arguments={"query": "test"})]
    )


def test_fabricated_citation_is_rejected():
    fake_pmid_answer = "Semaglutide slows gastric emptying (PMID: 99999999)."
    llm = FakeLLMClient(
        [
            _tool_call_response(),
            CompletionResult(text=fake_pmid_answer),
            CompletionResult(text=fake_pmid_answer),
            CompletionResult(text=fake_pmid_answer),
        ]
    )
    agent = CausalGraphAgent(llm=llm, vector_store=FakeVectorStore(), graph=FakeGraph())

    result = agent.ask("Does semaglutide affect gastric emptying?")

    assert result["citations"] == []
    assert "can't verify" in result["answer"].lower()


def test_valid_citation_passes():
    valid_answer = "This paper reports real evidence (PMID: 11111111)."
    llm = FakeLLMClient(
        [
            _tool_call_response(),
            CompletionResult(text=valid_answer),
        ]
    )
    agent = CausalGraphAgent(llm=llm, vector_store=FakeVectorStore(), graph=FakeGraph())

    result = agent.ask("What does the literature say?")

    assert result["citations"] == ["11111111"]
    assert result["answer"] == valid_answer
