# client.py
"""LLM client — HTTP communication with llama.cpp's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator

import httpx


class ServerStatus(Enum):
    ONLINE = "online"
    LOADING = "loading"
    OFFLINE = "offline"


@dataclass
class ModelInfo:
    id: str
    server: str  # "host:port"


class ChatStream:
    """Async iterator over an SSE stream. Yields content token strings."""

    def __init__(self, response: httpx.Response):
        self._response = response
        self.content: str = ""
        self.token_count: int = 0
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.was_interrupted: bool = False
        self.finish_reason: str | None = None
        self._usage_tokens: int | None = None
        self._prompt_tokens: int | None = None
        self._tool_calls: list[dict] = []

    @property
    def is_empty(self) -> bool:
        return self.token_count == 0

    @property
    def hit_max_tokens(self) -> bool:
        return self.finish_reason == "length"

    @property
    def duration(self) -> float:
        if self.first_token_time is not None and self.last_token_time is not None:
            return self.last_token_time - self.first_token_time
        return 0.0

    @property
    def final_token_count(self) -> int:
        if self._usage_tokens is not None:
            return self._usage_tokens
        return self.token_count

    @property
    def total_context_tokens(self) -> int | None:
        """Total tokens used (prompt + completion). None if unavailable."""
        if self._prompt_tokens is not None:
            completion = self._usage_tokens if self._usage_tokens is not None else self.token_count
            return self._prompt_tokens + completion
        return None

    @property
    def has_tool_calls(self) -> bool:
        return len(self._tool_calls) > 0

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._tool_calls)

    def set_prompt_tokens(self, count: int) -> None:
        """Set the prompt token count (called before streaming starts)."""
        self._prompt_tokens = count

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        try:
            async for line in self._response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                # Accumulate tool calls by index
                tool_call_chunks = delta.get("tool_calls")
                if tool_call_chunks:
                    for tc_chunk in tool_call_chunks:
                        index = tc_chunk["index"]
                        while len(self._tool_calls) <= index:
                            self._tool_calls.append(
                                {"id": "", "function": {"name": "", "arguments": ""}}
                            )
                        tc = self._tool_calls[index]
                        if "id" in tc_chunk:
                            tc["id"] = tc_chunk["id"]
                        func = tc_chunk.get("function", {})
                        if "name" in func:
                            tc["function"]["name"] = func["name"]
                        if "arguments" in func:
                            tc["function"]["arguments"] += func["arguments"]
                    continue
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    self.finish_reason = finish_reason
                usage = chunk.get("usage")
                if usage:
                    if "completion_tokens" in usage:
                        self._usage_tokens = usage["completion_tokens"]
                    if "prompt_tokens" in usage:
                        self._prompt_tokens = usage["prompt_tokens"]
                # Handle both reasoning and content tokens
                token_content = reasoning or content
                is_reasoning = reasoning is not None
                if token_content:
                    now = time.monotonic()
                    if self.first_token_time is None:
                        self.first_token_time = now
                    self.last_token_time = now
                    self.token_count += 1
                    if not is_reasoning:
                        self.content += token_content
                    yield (token_content, is_reasoning)
        except (httpx.StreamError, httpx.RemoteProtocolError):
            self.was_interrupted = True
        finally:
            await self._response.aclose()


class LlamaClient:
    """Async client for llama.cpp's OpenAI-compatible API."""

    _RETRY_DELAYS = [1, 2, 4]
    _MAX_RETRIES = 3

    def __init__(self, host: str, port: int, *, http_client: httpx.AsyncClient | None = None):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._http = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        )
        self._owns_client = http_client is None

    @property
    def server(self) -> str:
        return f"{self.host}:{self.port}"

    async def check_health(self) -> ServerStatus:
        """Check server health. Returns ONLINE, LOADING, or OFFLINE."""
        try:
            response = await self._http.get("/health")
            if response.status_code == 200:
                return ServerStatus.ONLINE
            elif response.status_code == 503:
                return ServerStatus.LOADING
            else:
                return ServerStatus.OFFLINE
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return ServerStatus.OFFLINE

    async def get_models(self) -> list[ModelInfo]:
        """Fetch the list of models currently loaded on this server."""
        response = await self._http.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return [ModelInfo(id=m["id"], server=self.server) for m in data.get("data", [])]

    async def get_context_length(self) -> int | None:
        """Auto-detect context length via /props. Returns None if unavailable."""
        try:
            response = await self._http.get("/props")
            response.raise_for_status()
            data = response.json()
            n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
            return int(n_ctx) if n_ctx is not None else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using the server's tokenizer."""
        response = await self._http.post("/tokenize", json={"content": text})
        response.raise_for_status()
        data = response.json()
        return len(data.get("tokens", []))

    async def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count tokens in a list of chat messages.

        Concatenates message contents with role prefixes for a reasonable estimate.
        Note: This won't exactly match the chat template, but is close.
        """
        parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content") or ""
            parts.append(f"{role}: {content}")
            if "tool_calls" in message:
                parts.append(json.dumps(message["tool_calls"]))
        text = "\n".join(parts)
        return await self.count_tokens(text)

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
    ) -> ChatStream:
        """Start a streaming chat completion. Retries connection errors up to _MAX_RETRIES times."""
        body: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            body["top_k"] = 64
        last_error: Exception | None = None

        for attempt in range(1 + self._MAX_RETRIES):
            try:
                request = self._http.build_request("POST", "/v1/chat/completions", json=body)
                response = await self._http.send(request, stream=True)
                response.raise_for_status()
                return ChatStream(response)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_DELAYS[attempt])
            except httpx.HTTPStatusError:
                raise

        raise last_error  # type: ignore[misc]

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._http.aclose()
