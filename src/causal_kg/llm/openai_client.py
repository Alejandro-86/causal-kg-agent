"""OpenAI implementation of LLMClient.

Default model is gpt-4o-mini (cheap enough that the whole build + demo
costs cents). Pass model="gpt-4o" at construction time for the live
demo if the strongest possible answers matter more than cost.
"""

import json
import os
import tempfile
from typing import Any

import certifi
import httpx2
from openai import OpenAI

from causal_kg.llm.base import CompletionResult, LLMClient, ToolCall


def _build_http_client() -> httpx2.Client:
    """openai's vendored httpx client ships its own internal CA bundle,
    which was missing a root needed to validate api.openai.com's current
    cert chain (independently confirmed: the top-level `certifi` package
    validates fine, httpx2's own default does not). Force it to use the
    real certifi bundle; if a corporate proxy CA is present (e.g. this
    machine's SSL_CERT_FILE), append it too so this also works on
    networks that actually do TLS-inspect."""
    bundle = certifi.where()
    extra = os.environ.get("SSL_CERT_FILE")
    if extra and os.path.exists(extra):
        cache_dir = tempfile.gettempdir()
        combined = os.path.join(cache_dir, "causal_kg_combined_cacert.pem")
        with open(bundle, "rb") as f1, open(extra, "rb") as f2:
            data = f1.read() + b"\n" + f2.read()
        with open(combined, "wb") as out:
            out.write(data)
        bundle = combined
    return httpx2.Client(verify=bundle)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key, http_client=_build_http_client())
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
