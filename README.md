# llama-chat

A lightweight terminal chat application for local LLMs running via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s HTTP server. Features streaming responses, markdown rendering, multi-turn conversation memory, and support for multiple models across GPUs.

## Features

- **Streaming responses** with real-time markdown rendering
- **Reasoning display** for models with thinking/reasoning (shown in dim italic)
- **Context usage tracking** with accurate token counts via `/tokenize` API
- **Multi-model support** with per-server context length configuration
- **Session persistence** with automatic save/resume
- **Auto-detection** of model context length via `/props`

## Prerequisites

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with a GGUF model

## Quick Start

### 1. Start llama.cpp server(s)

```bash
# Example: serve a model on GPU
llama-server -m model.gguf -ngl 99 --host 0.0.0.0 --port 8082
```

### 2. Install and run

```bash
git clone <repo-url>
cd llama-chat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./llama-chat.sh  # or: python main.py
```

On first run, an interactive setup will detect running servers and walk you through configuration.

## Configuration

Config is stored at `~/.config/llama-chat/.env`. Edit directly or delete to re-run setup.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVERS` | `localhost:8082,localhost:8081` | Comma-separated `host:port` or `host:port:context_length` list |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt |
| `LLAMA_MAX_TOKENS` | `8192` | Max tokens per response |
| `LLAMA_TEMPERATURE` | `1.0` | Sampling temperature |
| `LLAMA_CONTEXT_LENGTH` | `4096` | Fallback context window size |

## Commands

| Command | Action |
|---|---|
| `/help` | Show commands and config path |
| `/clear` | Clear and archive conversation |
| `/system <prompt>` | Change system prompt |
| `/model` | Switch model |
| `/agent` | Toggle coding agent mode |
| `/permissions` | Show or manage tool permission rules |
| `/quit` | Save and exit |

**Keyboard shortcuts:** Ctrl+C during generation cancels it. Double Ctrl+C at the prompt exits.

## Response Display

After each response, you'll see stats like:
```
527 tokens in 22.2s (23.7 tok/s) · context: 1,632/100,096 (2%)
```

For reasoning models (like Gemma 4), internal thinking is displayed in dim italic before the actual response.

If a response hits the max token limit, a warning is shown.

## Agent Mode

Type `/agent` to toggle agent mode. In this mode, the model can use tools to read files, write files, run shell commands, list directories, and search files. The model loops — calling tools and processing results — until it responds with plain text.

Command output streams to your terminal in real time. The agent is capped at 15 tool iterations per message.

## Permissions

When agent mode is active, `write_file` and `run_command` tools require your approval before executing. A permissions file (`.local-chat-llm-permissions`) is auto-created in your working directory.

When prompted, you can respond:
- **(y)es** — allow this once
- **(n)o** — deny (the model sees "User denied this tool call" and adjusts)
- **(a)lways** — add a permanent rule for this pattern

Manage rules with `/permissions`, `/permissions clear`, or `/permissions remove <n>`.

## Data

- Config: `~/.config/llama-chat/.env`
- Session: `~/.local/share/llama-chat/session.json`
- History: `~/.local/share/llama-chat/history/`
