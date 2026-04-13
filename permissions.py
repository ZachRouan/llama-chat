# permissions.py
"""Agent tool permission management — per-directory permission files."""

from __future__ import annotations

import json
from pathlib import Path

PERMISSIONS_FILENAME = ".local-chat-llm-permissions"

DEFAULT_PERMISSIONS: dict = {
    "read_file": "allow",
    "list_directory": "allow",
    "search_files": "allow",
    "write_file": "ask",
    "run_command": "ask",
    "allow_rules": [],
}


def load_permissions(directory: Path) -> dict:
    """Load permissions from the directory's permissions file.

    Creates the file with defaults if missing. Returns defaults if
    the file is corrupted or unreadable.
    """
    permissions_path = directory / PERMISSIONS_FILENAME
    if not permissions_path.exists():
        permissions_path.write_text(json.dumps(DEFAULT_PERMISSIONS, indent=2) + "\n")
        return dict(DEFAULT_PERMISSIONS)
    try:
        data = json.loads(permissions_path.read_text())
        return data
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_PERMISSIONS)
