> Design document written before implementation (April 2026). The README and ARCHITECTURE.md describe current behavior; this records the original design and rationale.

# Agent Tool Loop — Design Spec

**Goal:** Add a coding agent mode to llama-chat where the model can read files, write files, run commands, list directories, and search files via OpenAI-style function calling. The agent loops until the model responds with plain text instead of tool calls.

**Approach:** Inline in `main.py` (Approach A). One new module (`tools.py`), targeted changes to `client.py`, `chat.py`, `ui.py`, `main.py`, and `config.py`.

---

## 1. `tools.py` — Tool Definitions and Executors

New module with three responsibilities.

### Tool Definitions

`TOOL_DEFINITIONS` — a list of OpenAI-format tool dicts. Short descriptions to minimize token overhead (~200 tokens total for all 5):

| Tool | Params | Description |
|------|--------|-------------|
| `read_file` | `path` (string, required) | Read and return file contents |
| `write_file` | `path` (string, required), `content` (string, required) | Write content to a file, creating parent dirs as needed |
| `list_directory` | `path` (string, required), `recursive` (boolean, optional, default false) | List files and directories; tree view when recursive |
| `run_command` | `command` (string, required) | Execute a shell command, return stdout and stderr |
| `search_files` | `pattern` (string, required), `path` (string, optional, default `.`) | Search for a regex pattern in files under path |

### Argument Sanitizer

`clean_arguments(raw_args: str) -> dict` — strips `<|"|>` delimiter tokens from raw argument JSON before parsing. Handles a known Gemma 4 llama.cpp bug (#21316/#21384) where `<|"|>` tokens leak into tool call arguments when values contain curly braces.

```python
def clean_arguments(raw_args: str) -> dict:
    cleaned = re.sub(r'<\|"\|>', '', raw_args)
    return json.loads(cleaned)
```

### Executor

Two public functions:

**`execute_tool(name: str, arguments: dict) -> str`** — dispatches to the right tool for all tools except `run_command`. Never raises; catches all exceptions and returns error strings like `"Error: FileNotFoundError: /path/to/file"`. Outputs truncated at 50,000 chars with a `[truncated]` suffix.

- `read_file`: read and return file contents.
- `write_file`: `os.makedirs(parent, exist_ok=True)` then write. Return success message.
- `list_directory`: `os.listdir` for flat, `os.walk` tree for recursive.
- `search_files`: `grep -rn` via subprocess.

**`execute_command(command: str) -> AsyncGenerator[str, None]`** — separate async function for `run_command`. Yields output lines for real-time streaming to the UI.

- `asyncio.create_subprocess_shell` with `stdin=subprocess.DEVNULL` (prevents interactive commands from hanging).
- 60-second timeout. Yields error string on timeout.
- Yields each line of combined stdout/stderr as it arrives.
- The caller (`main.py`) collects lines, passes each to `ui.py` for display, then joins them into the final result string for the model.

`main.py` checks the tool name: if `run_command`, call `execute_command`; otherwise call `execute_tool`.

## 2. `client.py` — Tool Call Support in `ChatStream`

### `stream_chat` Changes

Add optional `tools: list[dict] | None` parameter. When provided:

```python
body["tools"] = tools
body["tool_choice"] = "auto"
body["top_k"] = 64  # Gemma 4 recommended, top-level field accepted by llama.cpp
```

No validation on the request dict — httpx sends raw JSON, so non-standard fields like `top_k` pass through to llama.cpp.

### `ChatStream` Changes

New fields:
- `_tool_calls: list[dict]` — accumulated tool calls, keyed by `index`
- `has_tool_calls: bool` property — True if any tool calls received
- `tool_calls: list[dict]` property — returns `[{"id": str, "function": {"name": str, "arguments": str}}]`

In `__aiter__`:
- Check `delta.get("tool_calls")` — if present, accumulate by `index`. First chunk has `id`, `name`, partial `arguments`; subsequent chunks append to `arguments`.
- Don't yield anything for tool call chunks. Content/reasoning tokens yield as before.
- `finish_reason`: treat both `"tool_calls"` and `"tool"` as tool call indicators (llama.cpp inconsistency between builds).

Existing `content`, `token_count`, timing, and usage tracking untouched.

## 3. `chat.py` — Tool Messages in History

### `add_message` Signature Change

From `(role: str, content: str)` to:

```python
def add_message(self, role: str, content: str | None = None, **kwargs) -> None:
    msg = {"role": role}
    if content is not None:
        msg["content"] = content
    msg.update(kwargs)
    self._messages.append(msg)
```

Supports:
- Regular: `add_message("user", "hello")`
- Tool call: `add_message("assistant", content=None, tool_calls=[...])`
- Tool result: `add_message("tool", "result text", tool_call_id="abc123")`

### Type Hint Changes

- `_messages`: `list[dict[str, str]]` → `list[dict]`
- `get_messages_for_api` return type: `list[dict[str, str]]` → `list[dict]`

### `estimate_tokens` Update

Handle `content` being `None` and account for `tool_calls` overhead:

```python
content = msg.get("content") or ""
total_chars += len(str(content))
if "tool_calls" in msg:
    total_chars += len(json.dumps(msg["tool_calls"]))
```

### `truncate_if_needed` — Turn-Aware Truncation

Current behavior drops pairs (user + assistant). With tool messages, a single turn can be:

```
user: "fix the bug"
assistant: tool_calls=[read_file]
tool: (file contents)
assistant: tool_calls=[write_file]
tool: (success)
assistant: "Done, I fixed the bug"
```

Dropping in pairs would orphan tool results — the model sees a tool response without the corresponding call, breaking the conversation format.

New behavior: drop from the front until hitting the next `"user"` message. This removes complete turns regardless of how many tool call/result pairs they contain.

### `count_messages_tokens` in `client.py`

Type hint widened. Serialization updated to handle messages where `content` is `None` or where extra fields like `tool_calls` are present — serialize the full message dict, not just `role: content`.

## 4. `main.py` — Agent Mode and Tool Loop

### `ChatState` Change

New field: `agent_mode: bool = False`.

### New Command: `/agent`

- `/agent` — toggle agent mode on/off
- `/agent on` — enable
- `/agent off` — disable
- Displays status change via `ui.print_muted`

### Agent System Prompt

When `agent_mode` is on, the agent system prompt (from `config.py`) is prepended to the user's system prompt when building messages for the API. This happens in `send_message`, not in `ChatSession` — the stored prompt stays clean. The user's custom system prompt is preserved and combined, not replaced.

### `send_message` — Tool Loop

When `agent_mode` is True, the send logic wraps in a loop with a fresh spinner per iteration:

```
for iteration in range(15):
    spinner = SpinnerDisplay()
    spinner.start()
    queue_task = create_task(queue_updater(...))

    build messages (with agent system prompt prepended if agent mode)
    count tokens, truncate if needed
    stream = stream_chat(..., tools=TOOL_DEFINITIONS if agent_mode else None)

    stream tokens (cancel queue_task on first token, start fresh StreamingDisplay)
    stop streaming display

    if not stream.has_tool_calls:
        add to history, show stats, break

    # tool calls received
    add assistant tool_call message to history
    for each tool call:
        clean_arguments to parse args
        ui.print_tool_call (dim summary)
        execute tool (run_command streams lines to ui in real time)
        ui.print_tool_result (brief result summary)
        add tool result message to history
    # loop → new spinner starts at top
else:
    # hit 15-iteration cap
    show warning via ui
    display model's last partial response if there was one
```

**Loop rhythm:** spinner → model streams response → (if tool calls) display + execute tools → loop back → spinner restarts. The user sees a natural thinking → acting → thinking → acting cycle.

**Ctrl+C** — existing `CancelledError` handling covers cancellation mid-loop. Clean up and return the prefill string.

**Prompt string** — `chat_loop` passes `"Agent > "` or `"You > "` to `ui.get_input` based on `state.agent_mode`.

When agent mode is off, everything works exactly as before — no tools sent, no tool parsing.

## 5. `ui.py` — Tool Display Functions

### `print_tool_call(name: str, args: dict)`

Dim one-liner showing what the model is doing:

| Tool | Format |
|------|--------|
| `read_file` | `→ read_file: /path/to/file` |
| `write_file` | `→ write_file: /path/to/file (12 lines)` |
| `run_command` | `→ run_command: pytest tests/` |
| `list_directory` | `→ list_directory: src/` |
| `search_files` | `→ search_files: "def test" in tests/` |
| Unknown | First 100 chars of args |

### `print_tool_result(name: str, result: str, max_display: int = 200)`

- Errors: `[red]  ✗ Error: FileNotFoundError: ...[/red]`
- Large output (>5 lines): `[dim]  ✓ (42 lines)[/dim]`
- Small output: `[dim]  ✓ result preview...[/dim]`

### `print_command_output(line: str)`

Streams `run_command` output line by line in dim style. Called from `main.py` as each line arrives from the async subprocess.

### `/help` Table

Add row: `/agent` → `"Toggle coding agent mode (tool use)"`

## 6. `config.py` — Constants and Default Changes

### Agent System Prompt Constant

```python
AGENT_SYSTEM_PROMPT = """You are a coding agent. You have tools to read files, write files, run commands, list directories, and search files.

When given a task:
1. Read relevant files to understand the current code
2. Plan your changes
3. Make changes file by file
4. Verify your changes work (run tests, check syntax)

Work step by step. Only call one or two tools at a time. After writing a file, verify it looks correct by reading it back or running a relevant command."""
```

### Default Temperature Change

Update default from `0.7` to `1.0` in `load_config` and `run_first_time_setup`. This is Google's official recommendation for Gemma 4. Also update `.env.example` and `README.md` to document `1.0` as the default.

## 7. Tests

### `tests/test_tools.py` (new)

- `clean_arguments` strips `<|"|>` tokens correctly
- `clean_arguments` handles normal JSON without tokens
- `execute_tool("read_file", ...)` reads a temp file
- `execute_tool("write_file", ...)` creates file and parent dirs
- `execute_tool("run_command", {"command": "echo hello"})` returns `"hello\n"`
- `execute_tool("run_command", ...)` with command exceeding 60s returns timeout error
- `execute_tool("list_directory", ...)` flat and recursive
- `execute_tool("search_files", {"pattern": "def test", "path": ...})` finds matches
- Unknown tool name returns error string
- Output truncation at 50,000 chars
- `run_command` with `stdin`-seeking command (`cat` no args) returns immediately due to `DEVNULL`

### `tests/test_client.py` (update)

- `stream_chat` includes `tools`, `tool_choice`, and `top_k` in request body when tools provided
- `stream_chat` omits those fields when tools is None
- `ChatStream` accumulates tool call chunks across multiple SSE events
- `has_tool_calls` True when tool calls received, False for normal text
- Handles both `finish_reason: "tool_calls"` and `"tool"`

### `tests/test_chat.py` (update)

- `add_message` with `tool_calls` kwarg stores correctly
- `add_message` with `role="tool"` and `tool_call_id` stores correctly
- `get_messages_for_api` includes tool messages
- `estimate_tokens` handles `None` content
- `truncate_if_needed` drops complete turns (user → tool calls → tool results → assistant) rather than orphaning tool messages

## Constraints

- **UI isolation:** Tool display in `ui.py`. Tool execution in `tools.py`. No `rich` imports outside `ui.py`.
- **Async:** `stream_chat` stays async. `run_command` uses `asyncio.create_subprocess_shell`. Other tool executors are synchronous but brief.
- **No new dependencies.** Everything uses stdlib (`subprocess`, `os`, `re`, `json`, `pathlib`).
- **Backwards compatible.** Agent mode off = everything works exactly as before.
- **No server changes.** llama-server already runs with `--jinja`. Tool calling requires only `tools` in the request body.
- **Error display:** Tool errors in red. Model sees error string as tool result and can adjust.
- **Safety:** `run_command` uses `shell=True` — personal tool on local machine. `stdin=DEVNULL` prevents interactive hangs. 60-second timeout prevents stuck processes.
- **Truncation:** Large outputs truncated to 50,000 chars with `[truncated]` suffix.

## File Map

| File | Change |
|------|--------|
| `tools.py` | **New** — tool definitions, argument sanitizer, executors |
| `client.py` | Add `tools` param to `stream_chat`, tool call accumulation in `ChatStream` |
| `chat.py` | Widen `add_message`, turn-aware truncation, handle `None` content |
| `main.py` | `agent_mode` on `ChatState`, `/agent` command, tool loop in `send_message` |
| `ui.py` | `print_tool_call`, `print_tool_result`, `print_command_output`, `/help` update |
| `config.py` | `AGENT_SYSTEM_PROMPT` constant, default temperature → 1.0 |
| `.env.example` | Temperature default → 1.0 |
| `README.md` | Temperature default → 1.0 |
| `tests/test_tools.py` | **New** — tool executor tests |
| `tests/test_client.py` | Tool call streaming tests |
| `tests/test_chat.py` | Tool message and truncation tests |
