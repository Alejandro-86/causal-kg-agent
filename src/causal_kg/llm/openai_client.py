"""OpenAI implementation of LLMClient.

Default model is gpt-4o-mini (cheap enough that the whole build + demo
costs cents). Pass model="gpt-4o" at construction time for the live
demo if the strongest possible answers matter more than cost.
"""

import json
from typing import Any

from openai import OpenAI

from causal_kg.llm.base import CompletionResult, LLMClient, ToolCall


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "response"),
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return json.loads(response.choices[0].message.content)

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CompletionResult:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        choice = response.choices[0].message

        if choice.tool_calls:
            return CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in choice.tool_calls
                ]
            )
        return CompletionResult(text=choice.content)
