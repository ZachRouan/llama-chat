> Design document written before implementation (April 2026). The README and ARCHITECTURE.md describe current behavior; this records the original design and rationale.

# Agent Permissions — Design Spec

**Goal:** Add a permission system for agent tool calls so `run_command` and `write_file` require user confirmation before executing, with the ability to permanently allow specific patterns.

**Approach:** New `permissions.py` module. Permission check in `main.py` between tool call parsing and execution. Permissions stored in `.local-chat-llm-permissions` JSON file in the working directory.

---

## 1. Permissions File

File: `.local-chat-llm-permissions` in the working directory (wherever `llama-chat` was launched from).

Auto-created with these defaults when missing:

```json
{
  "read_file": "allow",
  "list_directory": "allow",
  "search_files": "allow",
  "write_file": "ask",
  "run_command": "ask",
  "allow_rules": []
}
```

- `"allow"` — tool executes without prompting
- `"ask"` — user is prompted before each call (unless an `allow_rules` entry matches)

### Allow Rules

`allow_rules` is a list of objects with `tool` and `pattern` fields:

- **`write_file`** rules match against the `path` argument using **directory prefix matching** — if the path starts with the rule's directory, it matches. This handles nested subdirectories naturally (e.g., pattern `/tmp/calculator/` matches `/tmp/calculator/tests/test_calc.py`).
- **`run_command`** rules match against the `command` argument using `fnmatch` glob matching.

Example after the user says "always" a few times:

```json
{
  "read_file": "allow",
  "list_directory": "allow",
  "search_files": "allow",
  "write_file": "ask",
  "run_command": "ask",
  "allow_rules": [
    {"tool": "write_file", "pattern": "/tmp/calculator/"},
    {"tool": "run_command", "pattern": "pytest*"}
  ]
}
```

## 2. Permission Check Flow

In `main.py`, after `clean_arguments` parses the tool call but before execution:

1. Load permissions from `.local-chat-llm-permissions` (create default if missing)
2. Check tool's default: `"allow"` → proceed, `"ask"` → continue to step 3
3. Check `allow_rules`: if any rule matches tool name + relevant argument → proceed
4. Prompt user via `ui.prompt_tool_permission`:
   - **yes** → execute this once
   - **no** → skip execution, return `"User denied this tool call"` as tool result to model
   - **always** → add rule to permissions file, then execute

The model sees denial as a normal tool result and can adjust its approach.

### Prompt Format

```
  ⚠ write_file: /tmp/calculator/calc.py (12 lines)
  Allow? (y)es / (n)o / (a)lways: 
```

Uses the same tool summary format as `print_tool_call` (dim style) but with a ⚠ prefix and the confirmation prompt.

## 3. Pattern Derivation for "Always"

When the user says "always", a pattern is auto-derived from the arguments:

- **`write_file`**: parent directory path with trailing `/`. Example: `/tmp/calculator/calc.py` → `/tmp/calculator/`. Matches any file in that directory or its subdirectories (prefix match).
- **`run_command`**: first word + `*`. Example: `pytest tests/ -v` → `pytest*`.

Edge cases:
- **Bare filename** (e.g., `calc.py` with no parent): the pattern becomes the working directory. The prompt shows a warning: `⚠ This will allow all writes in the current directory.` so the user knows the scope before confirming.
- **Complex commands** with pipes/chaining: first word is still the primary command. `grep -r foo | wc -l` → `grep*`.

Users can hand-edit `.local-chat-llm-permissions` to tighten or loosen rules.

## 4. `/permissions` Command

New command in `main.py`:

- `/permissions` — show current tool defaults and allow rules in a table
- `/permissions clear` — remove all allow rules (resets to defaults)
- `/permissions remove <n>` — remove a specific rule by number (as shown in the table)

Display format:
```
  Tool defaults:
    read_file       allow
    list_directory  allow
    search_files    allow
    write_file      ask
    run_command     ask

  Allow rules:
    1. write_file: /tmp/calculator/
    2. run_command: pytest*
```

## 5. Module Structure

### `permissions.py` (new)

Three public functions:

- `load_permissions(directory: Path) -> dict` — reads `.local-chat-llm-permissions`, creates default if missing, returns parsed JSON
- `check_permission(permissions: dict, tool_name: str, arguments: dict) -> str` — returns `"allow"` or `"ask"`. Checks tool default first, then `allow_rules` via `fnmatch`. Unknown tools or invalid values default to `"ask"`
- `add_allow_rule(directory: Path, tool_name: str, arguments: dict) -> None` — derives pattern from arguments, appends to `allow_rules`, writes file back
- `remove_allow_rule(directory: Path, index: int) -> None` — removes rule by index, writes file back
- `clear_allow_rules(directory: Path) -> None` — removes all rules, writes file back

### `ui.py` (update)

New function:

- `prompt_tool_permission(name: str, summary: str) -> str` — displays the ⚠ prompt with tool summary, returns `"yes"`, `"no"`, or `"always"`

### `main.py` (update)

In the tool execution loop (inside `send_message`), after `clean_arguments` and `ui.print_tool_call`, before `execute_tool`/`execute_command`:

1. Call `check_permission`
2. If `"ask"`, call `ui.prompt_tool_permission` via `asyncio.to_thread` (since it uses `input()`)
3. Handle response: execute, skip with denial message, or add rule then execute

### No changes to

- `tools.py` — executors are unchanged
- `client.py` — streaming is unchanged
- `chat.py` — session management is unchanged
- `config.py` — permissions are per-directory, not global config

## 6. Tests

`tests/test_permissions.py` (new):

- `load_permissions` creates default file when missing
- `load_permissions` reads existing file correctly
- `check_permission` returns `"allow"` for auto-approved tools
- `check_permission` returns `"ask"` for write_file and run_command with no matching rules
- `check_permission` returns `"allow"` when allow_rule matches write_file path
- `check_permission` returns `"allow"` when allow_rule matches run_command pattern
- `check_permission` returns `"ask"` when allow_rule doesn't match
- `add_allow_rule` for write_file derives correct parent directory pattern
- `add_allow_rule` for run_command derives correct command pattern
- `add_allow_rule` persists to file and is visible on subsequent load
- `add_allow_rule` for bare filename derives working directory pattern
- `remove_allow_rule` removes correct rule by index
- `clear_allow_rules` removes all rules
- `check_permission` with write_file prefix matching handles nested subdirectories

No tests for `ui.prompt_tool_permission` or `main.py` integration — verified manually.

## File Map

| File | Change |
|------|--------|
| `permissions.py` | **New** — load, check, add_allow_rule |
| `tests/test_permissions.py` | **New** — permission logic tests |
| `ui.py` | Add `prompt_tool_permission` |
| `main.py` | Add permission check in tool execution loop, `/permissions` command |
