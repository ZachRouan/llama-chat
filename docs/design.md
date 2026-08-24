> Design document written before implementation (April 2026). The README and ARCHITECTURE.md describe current behavior; this records the original design and rationale.

# Local LLM Terminal Chat — Design Spec

## Overview

A lightweight terminal chat application that connects to local LLM instances running via llama.cpp's HTTP server. Provides a clean, rich-formatted chat interface with streaming responses, multi-turn conversation memory, and support for multiple models across GPUs.

## Architecture

Clean separation into UI, logic, and config layers. Only `ui.py` imports `rich` — no rich calls leak into other modules. `client.py` and `chat.py` are pure logic, making a future GUI swap straightforward.

### File Structure

```
├── main.py          # Entry point, main loop, command dispatch
├── client.py        # LLM client — API calls, streaming, retries
├── chat.py          # Chat session — message history, context management
├── ui.py            # Terminal UI — all rich rendering, input, spinners
├── config.py        # Config loading, first-run setup, .env management
├── requirements.txt
├── .env.example
└── README.md
```

### Module Responsibilities

**`config.py`** — Loads config from `~/.config/llama-chat/.env` via `python-dotenv`. On first run (no `.env` found), runs interactive setup using plain `input()`/`print()` (no `rich` dependency) that walks the user through each setting with sensible defaults, then writes `.env` to the fixed path and prints: "Config saved to ~/.config/llama-chat/.env". Creates the parent directory with `os.makedirs(parent, exist_ok=True)` before writing. Exposes a config object/dataclass used by other modules.

**`client.py`** — `LlamaClient` class. Handles HTTP communication with llama.cpp's OpenAI-compatible API. Async streaming via `httpx`. SSE parsing is manual: iterate lines from httpx's async byte stream, match `data: ` prefixes, parse JSON chunks, handle the `[DONE]` sentinel. Methods: connect/health check, list models (`/v1/models`), query server properties (`/props`), stream chat completions (`/v1/chat/completions` with `stream: true`). Handles edge cases: empty responses (stream completes with zero content tokens → returns an empty-response signal), mid-stream connection failures (returns partial content received so far with an error flag). Retry policy: 3 retries with exponential backoff (1s, 2s, 4s) for connection errors only — no retries on 4xx or during active streaming. No UI imports.

**`chat.py`** — `ChatSession` class. Manages conversation message history (list of `{role, content}` dicts). The system prompt is stored as a separate field, not in the messages list — it is prepended to the messages array when constructing each API request. Handles history clearing, system prompt changes. Context window management: truncates oldest messages (keeping the system prompt) when estimated token count approaches the active model's context length (see Context Window Management section), using a `chars / 4` approximation. No UI imports.

**`ui.py`** — All terminal presentation. Imports `rich`. Renders: banner, model selection menu, user prompt (`You > `), streaming markdown responses (via `rich.live.Live` with batched updates), spinners (with queue status and elapsed time), token/s stats, error messages, help text (including config file path). Uses `readline` (stdlib) for input pre-fill after Ctrl+C cancellation. This is the only file that knows about the terminal.

**`main.py`** — Wires everything together. Main async loop (`async def main()`), command dispatch. User input is blocking, so it runs in an executor via `asyncio.to_thread(ui.get_input)`. Streaming uses `httpx.AsyncClient` natively. Ctrl+C handling uses `loop.add_signal_handler(signal.SIGINT, handler)` for clean integration with the async event loop. Imports from all other modules.

## Startup Flow

1. **Config** — Look for `~/.config/llama-chat/.env`. If missing, run first-run interactive setup:
   - Server list: prompt for `host:port` pairs one at a time with "Add another? (y/n)" loop. Defaults to `localhost:8082` and `localhost:8081`. Per-server context length override is not collected here — set manually in `.env` using `host:port:context_length` if needed.
   - System prompt (default: `You are a helpful assistant.`)
   - Temperature (default: `0.7`)
   - Max tokens per response (default: `2048`)
   - Fallback context length (default: `4096`)
   - Write `.env` file. Print: "Config saved to ~/.config/llama-chat/.env"
2. **Resume check** — If `~/.local/share/llama-chat/session.json` exists, offer to resume (e.g., "Resume previous session with gemma-4-e4b-it on localhost:8082? (y/n)"). If accepted, verify the saved server by pinging `/health`:
   - **Server online, same model** → resume normally, skip to step 4.
   - **Server online, different model** → warn: "Server is now running [new_model] (session was with [old_model]). Continue anyway? (y/n)". If declined, fall through to step 3 (keep conversation history, switch model).
   - **Server offline or unreachable** → warn: "Server [host:port] is offline. Selecting a new model..." → fall through to step 3.
   If the user declines the initial resume offer, archive to `history/` and start fresh.
3. **Model selection** — Query each configured server's `/health` endpoint first, then `/v1/models` for online servers (use first model returned per server). Present a numbered menu:
   ```
   Select a model:
   [1] gemma-4-e4b-it-q8_0 (localhost:8082)
   [2] gemma-4-26b-a4b-it-q4_k_m (localhost:8081) (loading...)
   [3] localhost:8083 (offline)
   ```
   - **Online** servers (health returns 200) are selectable immediately.
   - **Loading** servers (health returns HTTP 503 with "Loading model") are selectable — selecting one shows a spinner ("Waiting for server to finish loading...") that polls `/health` every 2 seconds until it flips to `ok`. Ctrl+C during the wait returns to the model selection menu.
   - **Offline** servers (connection refused / timeout) are shown for visibility but cannot be selected.
   If all servers are offline, print an error with the list of unreachable servers and exit.
4. **Banner** — Display project name, selected model, server health status, model info.
5. **Enter chat loop.**

## Chat Loop

1. Display `You > ` prompt, wait for user input.
2. If input is empty or whitespace-only, re-prompt silently.
3. If input starts with `/`, dispatch to command handler. Unrecognized commands print "Unknown command. Type /help for available commands."
4. Otherwise, add user message to `ChatSession` history.
5. Before sending, `ChatSession` truncates oldest message pairs (keeping system prompt) if `estimated_tokens + max_tokens` exceeds the active model's context length.
6. Show spinner "Thinking..." while waiting for first SSE event. If no SSE event arrives within 3 seconds, update spinner to "Queued — waiting for an available slot on [host:port] (Ns)..." with a live elapsed-time counter. When the first SSE event arrives, the spinner stops and streaming begins.
7. Stream response tokens via SSE from `/v1/chat/completions`.
8. Render streaming markdown via `rich.live.Live` with `refresh_per_second=10`. Accumulate the full response text in a buffer; on each token, append to the buffer and update the `Live` display with `Markdown(buffer)`. The refresh rate cap prevents flicker from per-token re-renders.
9. On stream completion:
   - **Normal completion:** Add assistant message to history.
   - **Empty response** (zero content tokens — role chunk then `[DONE]`): Display `(empty response)` in muted style. Do not add to history.
   - **Mid-stream failure** (connection drops after tokens were received): Display the partial response followed by `(stream interrupted)` in muted style. Do not add the partial response or the user message to history. Pre-fill the next input prompt with the user's original message via `readline`.
10. Display token/s stat: `42 tokens in 3.2s (13.1 tok/s)`. Count SSE chunks containing content (not the initial role chunk or the `[DONE]` sentinel). Track wall-clock time from first content chunk to last content chunk. If the final SSE chunk includes `usage.completion_tokens`, use that for the token count instead of chunk count. Skip the stat line for empty responses.
11. Loop.

## Commands

| Command | Action |
|---|---|
| `/help` | Show available commands and config file path (`~/.config/llama-chat/.env`) |
| `/clear` | Archive the current session to `~/.local/share/llama-chat/history/` and reset conversation history. No-op if history is empty ("Nothing to clear.") |
| `/system <prompt>` | Change the system prompt field for this session. Takes effect on the next message sent — does not retroactively alter history. |
| `/model` | Show model selection menu, switch model (keeps conversation history). If the estimated conversation size exceeds 50% of the new model's context window, warn: "Current conversation (~N tokens) may not fit in [model]'s context (M tokens). Oldest messages will be truncated. Continue? (y/n)" |
| `/quit` | Save conversation and exit |

## Configuration

Stored in `~/.config/llama-chat/.env`, loaded via `python-dotenv`. Supports any number of servers.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVERS` | `localhost:8082,localhost:8081` | Comma-separated list of `host:port` or `host:port:context_length` pairs. Whitespace around entries is stripped. |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt |
| `LLAMA_MAX_TOKENS` | `2048` | Max tokens per response |
| `LLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLAMA_CONTEXT_LENGTH` | `4096` | Fallback context window size (used only if not specified per-server and auto-detection fails) |

## Context Window Management

Context length is resolved per-model with this priority:

1. **Per-server config override** — `host:port:context_length` in `LLAMA_SERVERS`
2. **Auto-detection** — query `/props` at connection time → `default_generation_settings.n_ctx` (the runtime context size, i.e., what the server was launched with via `-c`). Do not use `/v1/models` → `meta.n_ctx_train`, which is the model's training context and may be much larger than the runtime context.
3. **Global fallback** — `LLAMA_CONTEXT_LENGTH` from config (default 4096)

`ChatSession` estimates token usage with `len(text) / 4` (rough chars-to-tokens approximation). Before each API call, it checks that `estimated_conversation_tokens + max_tokens <= context_length`, reserving space for the model's response. If the budget is exceeded, it drops the oldest user/assistant message pairs (preserving the system prompt) until the conversation fits. This is a simple strategy — good enough for v1, and can be replaced with a proper tokenizer later.

## Error Handling

- **Connection refused / timeout:** Clear message to the user, offer to retry or pick another model.
- **Malformed responses:** Log the issue, skip gracefully, don't crash.
- **Empty response:** Stream completes with zero content tokens (role chunk → `[DONE]`). Display `(empty response)` in muted style. Do not add to history.
- **Mid-stream failure:** Connection drops after tokens were received. Display the partial response followed by `(stream interrupted)`. Do not add the partial response or user message to history. Pre-fill the next input prompt with the user's original message.
- **Ctrl+C during generation:** Stop streaming, discard the partial response and the user message that triggered it. Pre-fill the user's input prompt with their original message (via `readline.set_startup_hook` / `readline.insert_text`) so they can edit and resubmit. Session stays alive.
- **Ctrl+C at prompt:** First press shows "Press Ctrl+C again to quit." Second press within 3 seconds exits cleanly — no tracebacks. After 3 seconds the warning resets.
- **Ctrl+C during setup / model selection / resume prompt:** Exits immediately with a clean message ("Exiting."). No double-press required outside the chat loop.
- **Model offline at selection:** Indicate status in menu with `(offline)`, prevent selection.
- **Model loading at selection:** Indicate status with `(loading...)`, allow selection — poll `/health` every 2 seconds until ready.

## Retry Policy

Connection errors only (refused, timeout, network unreachable). 3 retries with exponential backoff: 1s, 2s, 4s. No retries on HTTP 4xx responses — those indicate a client-side problem. Never retry during active streaming — if the stream breaks after tokens have been delivered, show what was received and let the user decide whether to resend. On retry exhaustion for connection errors, display the error and return to the `You > ` prompt. The user can try again or use `/model` to switch.

## Request Timeouts

- **Connection timeout:** 10 seconds. The server is local; if it hasn't responded in 10s, it's not going to.
- **Read/stream timeout:** None. Once connected, never timeout — generation on large models can take minutes, and queued requests may wait indefinitely for a slot. The user can always Ctrl+C to cancel.

## Conversation Persistence

On `/quit` (or clean double-Ctrl+C exit), save the conversation to `~/.local/share/llama-chat/session.json`. Schema:

```json
{
  "server": "localhost:8082",
  "model": "gemma-4-e4b-it-q8_0",
  "system_prompt": "You are a helpful assistant.",
  "context_length": 4096,
  "started_at": "2026-04-12T22:15:00.000Z",
  "updated_at": "2026-04-12T22:45:30.123Z",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

- Timestamps are ISO 8601 with milliseconds and UTC `Z` suffix.
- `started_at`: session creation time. `updated_at`: time of last message.
- `model`: the model ID as returned by `/v1/models`.
- `context_length`: the resolved context length at session creation.
- System prompt is stored as metadata, not in the messages array.

On next launch, if `session.json` exists, offer to resume before model selection (e.g., "Resume previous session with gemma-4-e4b-it on localhost:8082? (y/n)"). If the user declines, archive the file to `~/.local/share/llama-chat/history/` with a timestamped filename based on `updated_at` (e.g., `2026-04-12T22-15-00-123.json`) and start fresh.

`/clear` also archives the current session to the history folder before resetting, unless the history is empty.

No conversation branching or multi-session management beyond this archival.

## Dependencies

Pinned to major versions in `requirements.txt`:

- `httpx>=0.27,<1` — async HTTP client for streaming SSE
- `rich>=13,<14` — terminal markdown rendering, spinners, styled output
- `python-dotenv>=1,<2` — `.env` file loading

Input pre-fill uses `readline` from the Python standard library (works on Linux/macOS; `prompt_toolkit` would be needed for cross-platform support in the future).

## README.md

Include a README with:
- What the project is (one paragraph)
- Prerequisites: Python 3.10+, llama.cpp with a GGUF model
- Quick start: how to install llama.cpp, download a model, launch the server, install Python deps, run the chat
- Configuration reference (the `.env` table above), including the config file location (`~/.config/llama-chat/.env`)
- Available commands

## Future Considerations

- GUI layer can be added by creating a new UI module that imports `client.py` and `chat.py` directly — no refactoring needed since `rich` is isolated to `ui.py`.
- Replace `chars / 4` token estimation with a proper tokenizer for more accurate context management.
- Multi-session conversation history (named sessions, browsing past chats).
- Cross-platform input pre-fill via `prompt_toolkit` (currently uses `readline`, which is Linux/macOS only).
- Unit tests for `client.py` and `chat.py` — the UI-isolated architecture makes these cleanly testable.
