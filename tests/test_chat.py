import json
from pathlib import Path
from chat import ChatSession


def test_init():
    session = ChatSession("You are helpful.", 4096, 2048)
    assert session.system_prompt == "You are helpful."
    assert session.is_empty()


def test_add_message():
    session = ChatSession("system", 4096, 2048)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    assert not session.is_empty()
    assert len(session.messages) == 2
    assert session.messages[0] == {"role": "user", "content": "hello"}
    assert session.messages[1] == {"role": "assistant", "content": "hi"}


def test_get_messages_for_api():
    session = ChatSession("Be concise.", 4096, 2048)
    session.add_message("user", "hello")
    msgs = session.get_messages_for_api()
    assert msgs[0] == {"role": "system", "content": "Be concise."}
    assert msgs[1] == {"role": "user", "content": "hello"}
    assert len(msgs) == 2


def test_set_system_prompt():
    session = ChatSession("old prompt", 4096, 2048)
    session.add_message("user", "hello")
    session.set_system_prompt("new prompt")
    assert session.system_prompt == "new prompt"
    assert len(session.messages) == 1
    assert session.get_messages_for_api()[0]["content"] == "new prompt"


def test_clear():
    session = ChatSession("system", 4096, 2048)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    old = session.clear()
    assert session.is_empty()
    assert len(old) == 2
    assert session.system_prompt == "system"


def test_remove_last_message():
    session = ChatSession("system", 4096, 2048)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    removed = session.remove_last_message()
    assert removed == {"role": "assistant", "content": "hi"}
    assert len(session.messages) == 1


def test_remove_last_message_empty():
    session = ChatSession("system", 4096, 2048)
    assert session.remove_last_message() is None


def test_estimate_tokens():
    session = ChatSession("system prompt", 4096, 2048)
    assert session.estimate_tokens() == 3  # 13 chars / 4
    session.add_message("user", "a" * 400)
    assert session.estimate_tokens() == 103  # 413 / 4


def test_truncate_drops_oldest_pairs():
    session = ChatSession("sys", 100, 50)
    for i in range(10):
        session.add_message("user", "x" * 40)
        session.add_message("assistant", "y" * 40)
    dropped = session.truncate_if_needed()
    assert dropped > 0
    assert session.estimate_tokens() + 50 <= 100


def test_truncate_preserves_system_prompt():
    session = ChatSession("important system prompt", 50, 25)
    session.add_message("user", "x" * 200)
    session.add_message("assistant", "y" * 200)
    session.truncate_if_needed()
    assert session.system_prompt == "important system prompt"


def test_truncate_noop_when_fits():
    session = ChatSession("sys", 4096, 2048)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    dropped = session.truncate_if_needed()
    assert dropped == 0
    assert len(session.messages) == 2


def test_save_and_load(tmp_path):
    session = ChatSession("Be helpful.", 4096, 2048)
    session.add_message("user", "hello")
    session.add_message("assistant", "hi there!")

    path = tmp_path / "session.json"
    session.save(path, "localhost:8082", "test-model")

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["server"] == "localhost:8082"
    assert data["model"] == "test-model"
    assert data["system_prompt"] == "Be helpful."
    assert data["context_length"] == 4096
    assert "started_at" in data
    assert "updated_at" in data
    assert len(data["messages"]) == 2

    loaded, meta = ChatSession.load(path, 2048)
    assert meta["server"] == "localhost:8082"
    assert meta["model"] == "test-model"
    assert loaded.system_prompt == "Be helpful."
    assert len(loaded.messages) == 2
    assert loaded.messages[0]["content"] == "hello"


def test_save_timestamps_are_iso8601(tmp_path):
    session = ChatSession("sys", 4096, 2048)
    session.add_message("user", "hi")
    path = tmp_path / "session.json"
    session.save(path, "localhost:8082", "model")

    data = json.loads(path.read_text())
    assert data["started_at"].endswith("Z")
    assert "T" in data["started_at"]
    assert data["updated_at"].endswith("Z")


def test_archive(tmp_path):
    session_path = tmp_path / "session.json"
    history_dir = tmp_path / "history"

    data = {
        "server": "localhost:8082",
        "model": "test",
        "system_prompt": "test",
        "context_length": 4096,
        "started_at": "2026-04-12T22:15:00.000Z",
        "updated_at": "2026-04-12T22:15:30.123Z",
        "messages": [{"role": "user", "content": "hi"}],
    }
    session_path.write_text(json.dumps(data))

    ChatSession.archive(session_path, history_dir)

    assert not session_path.exists()
    assert history_dir.exists()
    archived = list(history_dir.iterdir())
    assert len(archived) == 1
    assert archived[0].name == "2026-04-12T22-15-30-123.json"


def test_archive_nonexistent_is_noop(tmp_path):
    session_path = tmp_path / "nonexistent.json"
    history_dir = tmp_path / "history"
    ChatSession.archive(session_path, history_dir)
    assert not history_dir.exists()


def test_add_message_with_tool_calls():
    session = ChatSession("system", 4096, 2048)
    tool_calls = [{"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'}}]
    session.add_message("assistant", content=None, tool_calls=tool_calls)
    msg = session.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] is None  # OpenAI API requires content: null with tool_calls
    assert msg["tool_calls"] == tool_calls


def test_add_message_with_tool_result():
    session = ChatSession("system", 4096, 2048)
    session.add_message("tool", "file contents here", tool_call_id="call_1")
    msg = session.messages[0]
    assert msg["role"] == "tool"
    assert msg["content"] == "file contents here"
    assert msg["tool_call_id"] == "call_1"


def test_get_messages_for_api_includes_tool_messages():
    session = ChatSession("system", 4096, 2048)
    session.add_message("user", "read /tmp/test")
    session.add_message("assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}])
    session.add_message("tool", "file contents", tool_call_id="c1")
    session.add_message("assistant", "Here's what I found.")
    msgs = session.get_messages_for_api()
    assert len(msgs) == 5  # system + 4 messages
    assert msgs[2]["role"] == "assistant"
    assert msgs[2].get("tool_calls") is not None
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "c1"


def test_estimate_tokens_with_none_content():
    session = ChatSession("system", 4096, 2048)
    session.add_message("assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'}}])
    # Should not raise
    estimate = session.estimate_tokens()
    assert estimate > 0


def test_truncate_drops_complete_tool_turns():
    """Truncation drops entire turns including tool call/result sequences."""
    session = ChatSession("sys", 120, 50)
    # Turn 1: user + assistant(tool_call) + tool + assistant(text)
    session.add_message("user", "x" * 80)
    session.add_message("assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'}}])
    session.add_message("tool", "y" * 80, tool_call_id="c1")
    session.add_message("assistant", "z" * 80)
    # Turn 2: user + assistant
    session.add_message("user", "a" * 40)
    session.add_message("assistant", "b" * 40)

    dropped = session.truncate_if_needed()
    assert dropped > 0
    # First remaining message should be "user" (start of turn 2)
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "a" * 40


def test_truncate_does_not_orphan_tool_results():
    """After truncation, no tool result should appear without its tool call."""
    session = ChatSession("sys", 300, 50)
    # Turn 1 with tool calls
    session.add_message("user", "x" * 100)
    session.add_message("assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}])
    session.add_message("tool", "y" * 100, tool_call_id="c1")
    session.add_message("assistant", "done")
    # Turn 2
    session.add_message("user", "short")
    session.add_message("assistant", "ok")

    session.truncate_if_needed()
    # Check: no "tool" message appears before an assistant message with tool_calls
    roles = [m["role"] for m in session.messages]
    for i, role in enumerate(roles):
        if role == "tool":
            # There must be an assistant with tool_calls before this
            found = False
            for j in range(i - 1, -1, -1):
                if session.messages[j]["role"] == "assistant" and "tool_calls" in session.messages[j]:
                    found = True
                    break
                if session.messages[j]["role"] == "user":
                    break
            assert found, f"Orphaned tool result at index {i}"
