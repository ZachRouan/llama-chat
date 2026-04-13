import json
import httpx
import pytest
from client import LlamaClient, ServerStatus, ModelInfo, ChatStream


async def test_health_online():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    status = await client.check_health()
    assert status == ServerStatus.ONLINE
    await http.aclose()


async def test_health_loading():
    def handler(request):
        return httpx.Response(503, json={"error": {"message": "Loading model"}})
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    status = await client.check_health()
    assert status == ServerStatus.LOADING
    await http.aclose()


async def test_health_offline():
    def handler(request):
        raise httpx.ConnectError("Connection refused")
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    status = await client.check_health()
    assert status == ServerStatus.OFFLINE
    await http.aclose()


async def test_get_models():
    def handler(request):
        return httpx.Response(200, json={
            "data": [{"id": "gemma-4-e4b-it-q8_0", "object": "model"}]
        })
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    models = await client.get_models()
    assert len(models) == 1
    assert models[0].id == "gemma-4-e4b-it-q8_0"
    assert models[0].server == "localhost:8082"
    await http.aclose()


async def test_get_context_length():
    def handler(request):
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 4096}})
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    ctx = await client.get_context_length()
    assert ctx == 4096
    await http.aclose()


async def test_get_context_length_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    ctx = await client.get_context_length()
    assert ctx is None
    await http.aclose()


# --- ChatStream SSE parsing tests ---

def _sse_response(*events: str) -> httpx.Response:
    """Build a mock SSE response from data values."""
    body = "".join(f"data: {e}\n\n" for e in events)
    return httpx.Response(200, text=body)


async def test_stream_normal():
    response = _sse_response(
        '{"choices":[{"delta":{"role":"assistant"}}]}',
        '{"choices":[{"delta":{"content":"Hello"}}]}',
        '{"choices":[{"delta":{"content":" world"}}]}',
        '{"choices":[{"delta":{}}],"usage":{"completion_tokens":2}}',
        "[DONE]",
    )
    stream = ChatStream(response)
    tokens = [t async for t in stream]
    assert tokens == ["Hello", " world"]
    assert stream.content == "Hello world"
    assert stream.final_token_count == 2
    assert stream.token_count == 2
    assert not stream.is_empty
    assert not stream.was_interrupted
    assert stream.duration >= 0


async def test_stream_empty():
    response = _sse_response(
        '{"choices":[{"delta":{"role":"assistant"}}]}',
        '{"choices":[{"delta":{}}]}',
        "[DONE]",
    )
    stream = ChatStream(response)
    tokens = [t async for t in stream]
    assert tokens == []
    assert stream.content == ""
    assert stream.is_empty
    assert stream.final_token_count == 0
    assert not stream.was_interrupted


async def test_stream_usage_overrides_count():
    response = _sse_response(
        '{"choices":[{"delta":{"content":"Hi"}}]}',
        '{"choices":[{"delta":{}}],"usage":{"completion_tokens":5}}',
        "[DONE]",
    )
    stream = ChatStream(response)
    _ = [t async for t in stream]
    assert stream.token_count == 1
    assert stream.final_token_count == 5


async def test_stream_skips_malformed_lines():
    response = _sse_response(
        "not json",
        '{"choices":[{"delta":{"content":"ok"}}]}',
        "[DONE]",
    )
    stream = ChatStream(response)
    tokens = [t async for t in stream]
    assert tokens == ["ok"]


# --- stream_chat with retry tests ---

async def test_stream_chat_success():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["temperature"] == 0.7
        return _sse_response(
            '{"choices":[{"delta":{"content":"Hi"}}]}',
            "[DONE]",
        )
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    messages = [{"role": "user", "content": "hello"}]
    stream = await client.stream_chat(messages, temperature=0.7, max_tokens=2048)
    tokens = [t async for t in stream]
    assert tokens == ["Hi"]
    await http.aclose()


async def test_stream_chat_retries_on_connect_error():
    attempts = 0
    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise httpx.ConnectError("Connection refused")
        return _sse_response(
            '{"choices":[{"delta":{"content":"ok"}}]}',
            "[DONE]",
        )
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    client._RETRY_DELAYS = [0, 0, 0]  # skip delays in tests
    messages = [{"role": "user", "content": "hi"}]
    stream = await client.stream_chat(messages, temperature=0.7, max_tokens=100)
    tokens = [t async for t in stream]
    assert tokens == ["ok"]
    assert attempts == 3
    await http.aclose()


async def test_stream_chat_no_retry_on_4xx():
    def handler(request):
        return httpx.Response(400, json={"error": "bad request"})
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.stream_chat([{"role": "user", "content": "hi"}], 0.7, 100)
    assert exc_info.value.response.status_code == 400
    await http.aclose()


async def test_stream_chat_exhausts_retries():
    def handler(request):
        raise httpx.ConnectError("Connection refused")
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://localhost:8082")
    client = LlamaClient("localhost", 8082, http_client=http)
    client._RETRY_DELAYS = [0, 0, 0]
    with pytest.raises(httpx.ConnectError):
        await client.stream_chat([{"role": "user", "content": "hi"}], 0.7, 100)
    await http.aclose()
