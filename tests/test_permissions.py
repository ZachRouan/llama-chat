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


from permissions import check_permission


def test_check_permission_allow_for_read_file():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "read_file", {"path": "/tmp/test.py"})
    assert result == "allow"


def test_check_permission_allow_for_list_directory():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "list_directory", {"path": "."})
    assert result == "allow"


def test_check_permission_allow_for_search_files():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "search_files", {"pattern": "def test", "path": "."})
    assert result == "allow"


def test_check_permission_ask_for_write_file():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "write_file", {"path": "/tmp/test.py"})
    assert result == "ask"


def test_check_permission_ask_for_run_command():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "run_command", {"command": "rm -rf /"})
    assert result == "ask"


def test_check_permission_ask_for_unknown_tool():
    permissions = dict(DEFAULT_PERMISSIONS)
    result = check_permission(permissions, "unknown_tool", {})
    assert result == "ask"


def test_check_permission_ask_for_invalid_value():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["write_file"] = "typo"
    result = check_permission(permissions, "write_file", {"path": "/tmp/test.py"})
    assert result == "ask"


def test_check_permission_write_file_matches_rule():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["allow_rules"] = [{"tool": "write_file", "pattern": "/tmp/calculator/"}]
    result = check_permission(permissions, "write_file", {"path": "/tmp/calculator/calc.py"})
    assert result == "allow"


def test_check_permission_write_file_matches_nested_subdir():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["allow_rules"] = [{"tool": "write_file", "pattern": "/tmp/calculator/"}]
    result = check_permission(permissions, "write_file", {"path": "/tmp/calculator/tests/test_calc.py"})
    assert result == "allow"


def test_check_permission_write_file_no_match():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["allow_rules"] = [{"tool": "write_file", "pattern": "/tmp/calculator/"}]
    result = check_permission(permissions, "write_file", {"path": "/tmp/other/file.py"})
    assert result == "ask"


def test_check_permission_run_command_matches_rule():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["allow_rules"] = [{"tool": "run_command", "pattern": "pytest*"}]
    result = check_permission(permissions, "run_command", {"command": "pytest tests/ -v"})
    assert result == "allow"


def test_check_permission_run_command_no_match():
    permissions = dict(DEFAULT_PERMISSIONS)
    permissions["allow_rules"] = [{"tool": "run_command", "pattern": "pytest*"}]
    result = check_permission(permissions, "run_command", {"command": "rm -rf /"})
    assert result == "ask"
