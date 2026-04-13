# tools.py
"""Coding agent tools — definitions and executors."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


MAX_OUTPUT_CHARS = 50_000

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively as tree",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command, return stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a regex pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search"},
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


def _truncate(text: str) -> str:
    """Truncate output to MAX_OUTPUT_CHARS with a suffix."""
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n[truncated]"
    return text


def _read_file(arguments: dict) -> str:
    path = arguments["path"]
    content = Path(path).read_text()
    return _truncate(content)


def _write_file(arguments: dict) -> str:
    path = Path(arguments["path"])
    content = arguments["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    lines = content.count("\n") + 1
    return f"Wrote {lines} lines to {path}"


def _list_directory(arguments: dict) -> str:
    path = Path(arguments["path"])
    recursive = arguments.get("recursive", False)
    if not recursive:
        entries = sorted(os.listdir(path))
        return "\n".join(entries)
    # Recursive tree
    lines = []
    for root, dirs, files in os.walk(path):
        level = len(Path(root).relative_to(path).parts)
        indent = "  " * level
        lines.append(f"{indent}{Path(root).name}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files):
            lines.append(f"{sub_indent}{f}")
        dirs.sort()
    return _truncate("\n".join(lines))


def _search_files(arguments: dict) -> str:
    pattern = arguments["pattern"]
    path = arguments.get("path", ".")
    result = subprocess.run(
        ["grep", "-rn", pattern, path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout
    if result.returncode == 1:
        return ""  # no matches
    if result.returncode != 0 and result.stderr:
        return f"Error: grep failed: {result.stderr.strip()}"
    return _truncate(output)


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name. Never raises — returns error strings."""
    try:
        if name == "read_file":
            return _read_file(arguments)
        elif name == "write_file":
            return _write_file(arguments)
        elif name == "list_directory":
            return _list_directory(arguments)
        elif name == "search_files":
            return _search_files(arguments)
        else:
            return f"Error: Unknown tool '{name}'"
    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"


def clean_arguments(raw_args: str) -> dict:
    """Strip Gemma 4 <|"|> delimiter tokens from tool call arguments before parsing.

    Gemma 4 wraps string values in <|"|> tokens.  These may appear with or
    without an escaped quote (e.g. <|\\"|> in serialised form), so both
    variants are stripped.
    """
    # Match <|"|> with an optional preceding backslash on the inner quote.
    cleaned = re.sub(r'<\|\\?"\|>', "", raw_args)
    return json.loads(cleaned)
