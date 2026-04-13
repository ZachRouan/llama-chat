import json
from tools import clean_arguments, TOOL_DEFINITIONS


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
