# chat.py
"""Chat session management — message history, context window, persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ChatSession:
    """Manages conversation message history and context window."""

    def __init__(self, system_prompt: str, context_length: int, max_tokens: int):
        self._system_prompt = system_prompt
        self._context_length = context_length
        self._max_tokens = max_tokens
        self._messages: list[dict] = []
        self._started_at = datetime.now(timezone.utc)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def rollback_to(self, count: int) -> None:
        """Remove all messages after position `count`."""
        self._messages = self._messages[:count]

    def add_message(self, role: str, content: str | None = None, **kwargs) -> None:
        """Add a message to history.

        For regular messages: add_message("user", "hello")
        For tool calls: add_message("assistant", content=None, tool_calls=[...])
        For tool results: add_message("tool", "result text", tool_call_id="abc123")
        """
        message: dict = {"role": role}
        if content is not None:
            message["content"] = content
        message.update(kwargs)
        self._messages.append(message)

    def remove_last_message(self) -> dict | None:
        if self._messages:
            return self._messages.pop()
        return None

    def get_messages_for_api(self) -> list[dict]:
        """Prepend system prompt to conversation messages."""
        return [{"role": "system", "content": self._system_prompt}] + list(self._messages)

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_context_length(self, length: int) -> None:
        """Update the context window size (e.g., after switching models)."""
        self._context_length = length

    def clear(self) -> list[dict]:
        old = self._messages
        self._messages = []
        return old

    def estimate_tokens(self) -> int:
        """Estimate total token count using chars/4 approximation."""
        total_chars = len(self._system_prompt)
        for message in self._messages:
            content = message.get("content") or ""
            total_chars += len(str(content))
            if "tool_calls" in message:
                total_chars += len(json.dumps(message["tool_calls"]))
        return total_chars // 4

    def truncate_if_needed(self, token_count: int | None = None) -> int:
        """Drop oldest complete turns until conversation fits in context.

        A turn starts at a "user" message and includes everything up to
        (but not including) the next "user" message. This prevents orphaning
        tool call/result sequences.
        """
        turns_dropped = 0
        current_tokens = token_count if token_count is not None else self.estimate_tokens()

        while (
            current_tokens + self._max_tokens > self._context_length
            and len(self._messages) >= 2
        ):
            # Find the start of the second turn
            next_turn = 1
            while next_turn < len(self._messages) and self._messages[next_turn]["role"] != "user":
                next_turn += 1

            if next_turn >= len(self._messages):
                break  # only one turn left

            removed = self._messages[:next_turn]
            self._messages = self._messages[next_turn:]
            turns_dropped += 1

            removed_chars = 0
            for message in removed:
                content = message.get("content") or ""
                removed_chars += len(str(content))
                if "tool_calls" in message:
                    removed_chars += len(json.dumps(message["tool_calls"]))
            current_tokens -= removed_chars // 4

        return turns_dropped

    def save(self, path: Path, server: str, model: str) -> None:
        """Save session to a JSON file."""
        now = datetime.now(timezone.utc)
        data = {
            "server": server,
            "model": model,
            "system_prompt": self._system_prompt,
            "context_length": self._context_length,
            "started_at": self._started_at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{self._started_at.microsecond // 1000:03d}Z",
            "updated_at": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
            "messages": list(self._messages),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, max_tokens: int) -> tuple[ChatSession, dict]:
        """Load a session from JSON. Returns (session, metadata_dict)."""
        data = json.loads(path.read_text())
        session = cls(
            system_prompt=data["system_prompt"],
            context_length=data["context_length"],
            max_tokens=max_tokens,
        )
        session._messages = list(data["messages"])
        session._started_at = datetime.fromisoformat(
            data["started_at"].replace("Z", "+00:00")
        )
        meta = {
            "server": data["server"],
            "model": data["model"],
            "system_prompt": data["system_prompt"],
            "context_length": data["context_length"],
            "started_at": data["started_at"],
        }
        return session, meta

    @staticmethod
    def archive(session_path: Path, history_dir: Path) -> None:
        """Move session file to history/ with timestamped filename."""
        if not session_path.exists():
            return
        data = json.loads(session_path.read_text())
        updated = data.get(
            "updated_at",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )
        stem = updated.rstrip("Z").replace(":", "-").replace(".", "-")
        filename = f"{stem}.json"
        history_dir.mkdir(parents=True, exist_ok=True)
        session_path.rename(history_dir / filename)
