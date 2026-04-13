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


from permissions import add_allow_rule, remove_allow_rule, clear_allow_rules


def test_add_allow_rule_write_file_derives_parent_dir(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "write_file", {"path": "/tmp/calculator/calc.py"})
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 1
    assert permissions["allow_rules"][0] == {"tool": "write_file", "pattern": "/tmp/calculator/"}


def test_add_allow_rule_write_file_nested_path(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "write_file", {"path": "/tmp/project/src/lib/utils.py"})
    permissions = load_permissions(tmp_path)
    assert permissions["allow_rules"][0]["pattern"] == "/tmp/project/src/lib/"


def test_add_allow_rule_write_file_bare_filename(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "write_file", {"path": "calc.py"})
    permissions = load_permissions(tmp_path)
    rule = permissions["allow_rules"][0]
    assert rule["tool"] == "write_file"
    assert rule["pattern"] == ""


def test_add_allow_rule_run_command_derives_first_word(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "pytest tests/ -v"})
    permissions = load_permissions(tmp_path)
    assert permissions["allow_rules"][0] == {"tool": "run_command", "pattern": "pytest*"}


def test_add_allow_rule_run_command_single_word(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "ls"})
    permissions = load_permissions(tmp_path)
    assert permissions["allow_rules"][0] == {"tool": "run_command", "pattern": "ls*"}


def test_add_allow_rule_persists_and_is_visible(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "pytest tests/"})
    add_allow_rule(tmp_path, "write_file", {"path": "/tmp/out.py"})
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 2


def test_add_allow_rule_no_duplicate(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "pytest tests/"})
    add_allow_rule(tmp_path, "run_command", {"command": "pytest -v"})
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 1


def test_remove_allow_rule(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "pytest tests/"})
    add_allow_rule(tmp_path, "write_file", {"path": "/tmp/out.py"})
    remove_allow_rule(tmp_path, 0)
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 1
    assert permissions["allow_rules"][0]["tool"] == "write_file"


def test_remove_allow_rule_invalid_index(tmp_path):
    load_permissions(tmp_path)
    remove_allow_rule(tmp_path, 99)
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 0


def test_clear_allow_rules(tmp_path):
    load_permissions(tmp_path)
    add_allow_rule(tmp_path, "run_command", {"command": "pytest"})
    add_allow_rule(tmp_path, "write_file", {"path": "/tmp/out.py"})
    clear_allow_rules(tmp_path)
    permissions = load_permissions(tmp_path)
    assert len(permissions["allow_rules"]) == 0
    assert permissions["read_file"] == "allow"
    assert permissions["run_command"] == "ask"
