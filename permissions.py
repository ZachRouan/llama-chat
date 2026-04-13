# permissions.py
"""Agent tool permission management — per-directory permission files."""

from __future__ import annotations

import fnmatch
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


def _get_match_value(tool_name: str, arguments: dict) -> str:
    """Get the value to match against allow_rules for a given tool."""
    if tool_name == "write_file":
        return arguments.get("path", "")
    elif tool_name == "run_command":
        return arguments.get("command", "")
    return ""


def _matches_rule(rule: dict, tool_name: str, match_value: str) -> bool:
    """Check if an allow_rule matches the tool call."""
    if rule.get("tool") != tool_name:
        return False
    pattern = rule.get("pattern", "")
    if tool_name == "write_file":
        # Prefix match — path starts with the rule's directory
        return match_value.startswith(pattern)
    else:
        # Glob match for run_command
        return fnmatch.fnmatch(match_value, pattern)


def _derive_pattern(tool_name: str, arguments: dict) -> str:
    """Derive an allow rule pattern from tool arguments."""
    if tool_name == "write_file":
        path = arguments.get("path", "")
        parent = str(Path(path).parent)
        if parent == ".":
            return ""  # bare filename — matches everything in cwd
        return parent + "/"
    elif tool_name == "run_command":
        command = arguments.get("command", "")
        first_word = command.split()[0] if command.split() else ""
        return first_word + "*"
    return ""


def _save_permissions(directory: Path, permissions: dict) -> None:
    """Write permissions dict to the permissions file."""
    permissions_path = directory / PERMISSIONS_FILENAME
    permissions_path.write_text(json.dumps(permissions, indent=2) + "\n")


def add_allow_rule(directory: Path, tool_name: str, arguments: dict) -> None:
    """Add an allow rule derived from the tool arguments. Skips duplicates."""
    permissions = load_permissions(directory)
    pattern = _derive_pattern(tool_name, arguments)
    rule = {"tool": tool_name, "pattern": pattern}
    if rule not in permissions["allow_rules"]:
        permissions["allow_rules"].append(rule)
        _save_permissions(directory, permissions)


def remove_allow_rule(directory: Path, index: int) -> None:
    """Remove an allow rule by index. Silently ignores invalid indices."""
    permissions = load_permissions(directory)
    rules = permissions.get("allow_rules", [])
    if 0 <= index < len(rules):
        rules.pop(index)
        _save_permissions(directory, permissions)


def clear_allow_rules(directory: Path) -> None:
    """Remove all allow rules."""
    permissions = load_permissions(directory)
    permissions["allow_rules"] = []
    _save_permissions(directory, permissions)


def check_permission(permissions: dict, tool_name: str, arguments: dict) -> str:
    """Check if a tool call is allowed or needs approval.

    Returns "allow" or "ask". Unknown tools or invalid values default to "ask".
    Rule matching: write_file uses prefix match; run_command uses glob match.
    """
    default = permissions.get(tool_name, "ask")
    if default == "allow":
        return "allow"
    if default != "ask":
        return "ask"  # invalid value, safe default

    # Check allow_rules for explicit overrides
    match_value = _get_match_value(tool_name, arguments)
    for rule in permissions.get("allow_rules", []):
        if _matches_rule(rule, tool_name, match_value):
            return "allow"

    return "ask"
