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
