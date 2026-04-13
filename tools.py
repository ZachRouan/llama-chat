# tools.py
"""Coding agent tools — definitions and executors."""

from __future__ import annotations

import json
import re


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


def clean_arguments(raw_args: str) -> dict:
    """Strip Gemma 4 <|"|> delimiter tokens from tool call arguments before parsing.

    Gemma 4 wraps string values in <|"|> tokens.  These may appear with or
    without an escaped quote (e.g. <|\\"|> in serialised form), so both
    variants are stripped.
    """
    # Match <|"|> with an optional preceding backslash on the inner quote.
    cleaned = re.sub(r'<\|\\?"\|>', "", raw_args)
    return json.loads(cleaned)
