import json
import os
from pathlib import Path
from tools import clean_arguments, execute_tool, TOOL_DEFINITIONS, MAX_OUTPUT_CHARS


def test_clean_arguments_strips_gemma4_delimiters():
    raw = '{"path": "<|\\"|>/tmp/test<|\\"|>"}'
    # After stripping <|"|> tokens, this should parse as valid JSON
    raw_with_tokens = '{"path": "<|\\"|>/tmp/test<|\\"|>"}'
    result = clean_arguments(raw_with_tokens)
    assert result == {"path": "/tmp/test"}


def test_clean_arguments_handles_normal_json():
    raw = '{"path": "/tmp/test", "recursive": true}'
    result = clean_arguments(raw)
    assert result == {"path": "/tmp/test", "recursive": True}


def test_clean_arguments_empty_object():
    result = clean_arguments("{}")
    assert result == {}


def test_clean_arguments_nested_braces():
    raw = '{"content": "<|\\"|>{\\n  \\"key\\": \\"value\\"\\n}<|\\"|>"}'
    result = clean_arguments(raw)
    assert "key" in result["content"]


def test_tool_definitions_are_valid():
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) == 5
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert names == {"read_file", "write_file", "list_directory", "run_command", "search_files"}
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        assert "parameters" in tool["function"]


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = execute_tool("read_file", {"path": str(f)})
    assert result == "hello world"


def test_read_file_not_found():
    result = execute_tool("read_file", {"path": "/nonexistent/file.txt"})
    assert result.startswith("Error:")
    assert "FileNotFoundError" in result or "No such file" in result


def test_write_file(tmp_path):
    f = tmp_path / "output.txt"
    result = execute_tool("write_file", {"path": str(f), "content": "written"})
    assert "wrote" in result.lower() or "written" in result.lower() or str(f) in result
    assert f.read_text() == "written"


def test_write_file_creates_parent_dirs(tmp_path):
    f = tmp_path / "deep" / "nested" / "file.txt"
    result = execute_tool("write_file", {"path": str(f), "content": "deep"})
    assert f.exists()
    assert f.read_text() == "deep"


def test_list_directory_flat(tmp_path):
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "subdir").mkdir()
    result = execute_tool("list_directory", {"path": str(tmp_path)})
    assert "a.txt" in result
    assert "b.txt" in result
    assert "subdir" in result


def test_list_directory_recursive(tmp_path):
    (tmp_path / "top.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").touch()
    result = execute_tool("list_directory", {"path": str(tmp_path), "recursive": True})
    assert "top.txt" in result
    assert "nested.txt" in result


def test_list_directory_not_found():
    result = execute_tool("list_directory", {"path": "/nonexistent/dir"})
    assert result.startswith("Error:")


def test_search_files(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def test_hello():\n    pass\n\ndef test_world():\n    pass\n")
    result = execute_tool("search_files", {"pattern": "def test", "path": str(tmp_path)})
    assert "test_hello" in result
    assert "test_world" in result


def test_search_files_no_matches(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("# nothing here\n")
    result = execute_tool("search_files", {"pattern": "def nonexistent", "path": str(tmp_path)})
    # No matches — could be empty or a "no matches" message, but not an error
    assert not result.startswith("Error:")


def test_unknown_tool():
    result = execute_tool("nonexistent_tool", {})
    assert result.startswith("Error:")
    assert "Unknown tool" in result


def test_output_truncation(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * (MAX_OUTPUT_CHARS + 1000))
    result = execute_tool("read_file", {"path": str(f)})
    assert len(result) <= MAX_OUTPUT_CHARS + 100  # allow room for [truncated] suffix
    assert "[truncated]" in result
