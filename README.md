# llama-chat

A lightweight terminal chat application for local LLMs running via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s HTTP server. Features streaming responses, markdown rendering, multi-turn conversation memory, and support for multiple models across GPUs.

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
python main.py
```

On first run, an interactive setup will walk you through configuration.

## Configuration

Config is stored at `~/.config/llama-chat/.env`. Edit directly or delete to re-run setup.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVERS` | `localhost:8082,localhost:8081` | Comma-separated `host:port` or `host:port:context_length` list |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt |
| `LLAMA_MAX_TOKENS` | `2048` | Max tokens per response |
| `LLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLAMA_CONTEXT_LENGTH` | `4096` | Fallback context window size |

## Commands

| Command | Action |
|---|---|
| `/help` | Show commands and config path |
| `/clear` | Clear and archive conversation |
| `/system <prompt>` | Change system prompt |
| `/model` | Switch model |
| `/quit` | Save and exit |

**Keyboard shortcuts:** Ctrl+C during generation cancels it. Double Ctrl+C at the prompt exits.

## Data

- Session: `~/.local/share/llama-chat/session.json`
- History: `~/.local/share/llama-chat/history/`
