"""Provider-agnostic LLM client interface.

Every call site in this project talks to this interface, never to a
provider SDK directly. Swapping providers means writing one new class
here and changing a single construction site — nothing downstream
changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class CompletionResult:
    """Either a final text answer, or a list of tool calls to execute."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(ABC):
    @abstractmethod
    def complete_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object conforming to the given JSON schema."""

    @abstractmethod
    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CompletionResult:
        """Send a conversation + tool definitions, get back either a final
        answer or tool calls the caller must execute and feed back in as
        role='tool' messages on the next call."""
