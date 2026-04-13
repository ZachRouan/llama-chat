import json
from pathlib import Path
from permissions import load_permissions, PERMISSIONS_FILENAME, DEFAULT_PERMISSIONS


def test_load_permissions_creates_default_when_missing(tmp_path):
    permissions = load_permissions(tmp_path)
    assert permissions == DEFAULT_PERMISSIONS
    permissions_file = tmp_path / PERMISSIONS_FILENAME
    assert permissions_file.exists()
    data = json.loads(permissions_file.read_text())
    assert data == DEFAULT_PERMISSIONS


def test_load_permissions_reads_existing_file(tmp_path):
    permissions_file = tmp_path / PERMISSIONS_FILENAME
    custom = {
        "read_file": "allow",
        "list_directory": "allow",
        "search_files": "allow",
        "write_file": "allow",
        "run_command": "ask",
        "allow_rules": [{"tool": "run_command", "pattern": "pytest*"}],
    }
    permissions_file.write_text(json.dumps(custom, indent=2))
    permissions = load_permissions(tmp_path)
    assert permissions == custom


def test_load_permissions_handles_corrupted_file(tmp_path):
    permissions_file = tmp_path / PERMISSIONS_FILENAME
    permissions_file.write_text("not json{{{")
    permissions = load_permissions(tmp_path)
    assert permissions == DEFAULT_PERMISSIONS
