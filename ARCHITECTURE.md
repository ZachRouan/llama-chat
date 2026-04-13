# Architecture

A lightweight terminal chat application for local LLMs running via llama.cpp's HTTP server.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Terminal (TTY)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           ui.py                                 │
│  • Rich markdown rendering    • Spinners & streaming display    │
│  • Input handling (readline)  • Banner, menus, stats            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│  • Async main loop            • Command dispatch (/help, etc.)  │
│  • Signal handling (Ctrl+C)   • Session resume flow             │
│  • Model selection            • Message send orchestration      │
└─────────────────────────────────────────────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│         chat.py           │   │          client.py              │
│  • Message history        │   │  • HTTP client (httpx)          │
│  • Context truncation     │   │  • SSE stream parsing           │
│  • Session persistence    │   │  • Token counting (/tokenize)   │
│  • Token estimation       │   │  • Health checks, retries       │
└───────────────────────────┘   └─────────────────────────────────┘
                                                │
                                                ▼
                                ┌─────────────────────────────────┐
                                │     llama.cpp HTTP Server       │
                                │  • /v1/chat/completions (SSE)   │
                                │  • /v1/models, /health, /props  │
                                │  • /tokenize                    │
                                └─────────────────────────────────┘
```

## Modules

### main.py
Entry point and orchestration. Handles:
- Async event loop and signal handling
- Model selection menu with health checks
- Session resume flow
- Command dispatch (`/help`, `/clear`, `/model`, `/system`, `/agent`, `/quit`)
- Message send with streaming display coordination
- Agent mode: tool loop with tool call parsing and execution
- Graceful exit (double Ctrl+C)

### client.py
HTTP communication with llama.cpp's OpenAI-compatible API:
- `LlamaClient` — async client with connection pooling
- `ChatStream` — SSE parser yielding `(token, is_reasoning)` tuples, tool call accumulation
- `ServerStatus` enum — ONLINE, LOADING, OFFLINE
- Token counting via `/tokenize` endpoint
- Context length auto-detection via `/props`
- Retry logic (3 attempts with exponential backoff for connection errors)

### chat.py
Conversation state management:
- `ChatSession` — message history, system prompt
- Turn-aware context window truncation (drops complete turns including tool sequences)
- Session persistence to JSON
- Archive to timestamped history files

### ui.py
Terminal rendering (only module that imports `rich`):
- `SpinnerDisplay` — "Thinking..." / "Queued" with elapsed time
- `StreamingDisplay` — live markdown rendering, reasoning in dim italic
- Banner, model menu, stats, error messages
- Input handling with readline prefill
- Tool call and result display (dim styling, streaming command output)

### tools.py
Coding agent tool definitions and executors:
- `TOOL_DEFINITIONS` — OpenAI-format tool schemas for 5 tools
- `clean_arguments` — strips Gemma 4 `<|"|>` delimiter tokens
- `execute_tool` — sync dispatch for read_file, write_file, list_directory, search_files
- `execute_command` — async generator streaming command output with 60s timeout

### config.py
Configuration loading:
- Parse `.env` file and environment variables
- `ServerConfig` — host, port, optional context length override
- `AppConfig` — servers, system prompt, max tokens, temperature
- First-run interactive setup with server probing

## Data Flow

### Startup
```
1. Load config from ~/.config/llama-chat/.env (or run first-time setup)
2. Check for existing session at ~/.local/share/llama-chat/session.json
3. If session exists, offer to resume (check if server/model still available)
4. Otherwise, show model selection menu (probe all servers in parallel)
5. Display banner with model info and context length
```

### Message Send
```
1. Add user message to session
2. Count tokens via /tokenize API
3. Truncate old messages if needed (context window management)
4. Start spinner, create queue timer task
5. POST to /v1/chat/completions with stream: true
6. Parse SSE events:
   - reasoning_content → display dim italic, don't save
   - content → display markdown, accumulate for history
7. On first token: cancel queue timer, switch to streaming display
8. On completion: show stats (tokens, speed, context usage)
9. Add assistant message to session
```

### Session Persistence
```
Save triggers:
- /quit command
- /clear command (before clearing)
- Double Ctrl+C exit

Archive triggers:
- /clear command (moves session.json to history/)
- Declining session resume (archives old session)
```

## File Structure

```
llama-chat/
├── main.py              # Entry point, main loop, command dispatch
├── client.py            # LlamaClient, ChatStream, API communication
├── chat.py              # ChatSession, message history, persistence
├── ui.py                # Terminal UI (rich), spinners, streaming
├── tools.py             # Agent tool definitions and executors
├── config.py            # Config loading, first-run setup
├── llama-chat.sh        # Launch script (activates venv)
├── requirements.txt     # httpx, rich, python-dotenv
├── tests/               # pytest tests
│   ├── test_chat.py
│   ├── test_client.py
│   ├── test_config.py
│   └── test_tools.py
└── docs/
    └── *.md             # Design documents
```

## Data Paths (XDG Base Directory)

| Path | Purpose |
|------|---------|
| `~/.config/llama-chat/.env` | Configuration file |
| `~/.local/share/llama-chat/session.json` | Current session |
| `~/.local/share/llama-chat/history/` | Archived conversations |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Streaming chat (SSE) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Server status (200=online, 503=loading) |
| `/props` | GET | Server properties including `n_ctx` |
| `/tokenize` | POST | Accurate token counting |

## Configuration

Environment variables (via `.env` or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVERS` | `localhost:8082,localhost:8081` | Comma-separated `host:port` or `host:port:context_length` |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt |
| `LLAMA_MAX_TOKENS` | `8192` | Max tokens per response |
| `LLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLAMA_CONTEXT_LENGTH` | `4096` | Fallback context window |

Context length resolution order:
1. Per-server override in `LLAMA_SERVERS` (e.g., `localhost:8081:100000`)
2. Auto-detect via `/props` endpoint
3. Fallback to `LLAMA_CONTEXT_LENGTH`

## Design Decisions

### UI Isolation
Only `ui.py` imports `rich`. All other modules are UI-agnostic, enabling future GUI implementations without modifying core logic.

### Async Throughout
Uses `asyncio` and `httpx.AsyncClient` for non-blocking I/O. Input is handled via `asyncio.to_thread()` to avoid blocking the event loop.

### SSE Parsing
Manual parsing of Server-Sent Events rather than using a library. Handles both `content` and `reasoning_content` delta types for reasoning models.

### Token Counting
Uses llama.cpp's `/tokenize` endpoint for accurate counts rather than estimation. Estimation (chars/4) is only used for calculating token reduction during truncation.

### Signal Handling
Custom SIGINT handler via `loop.add_signal_handler()`:
- During generation: cancels the stream
- At prompt: first press warns, second press exits
- Uses `os._exit(0)` for clean termination (avoids async cleanup issues)

### No CLI Arguments
All configuration via environment variables / `.env` file. Simplifies usage and makes configuration persistent.

## Dependencies

- **httpx** — async HTTP client with streaming support
- **rich** — terminal formatting, markdown rendering, live displays
- **python-dotenv** — `.env` file loading

No heavy frameworks. Total dependencies kept minimal for fast startup.
