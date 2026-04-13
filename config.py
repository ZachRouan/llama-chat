"""Configuration loading from ~/.config/llama-chat/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Fixed paths (XDG Base Directory)
CONFIG_DIR = Path.home() / ".config" / "llama-chat"
ENV_PATH = CONFIG_DIR / ".env"
DATA_DIR = Path.home() / ".local" / "share" / "llama-chat"
HISTORY_DIR = DATA_DIR / "history"
SESSION_PATH = DATA_DIR / "session.json"

AGENT_SYSTEM_PROMPT = """You are a coding agent. You have tools to read files, write files, run commands, list directories, and search files.

When given a task:
1. Read relevant files to understand the current code
2. Plan your changes
3. Make changes file by file
4. Verify your changes work (run tests, check syntax)

Work step by step. Only call one or two tools at a time. After writing a file, verify it looks correct by reading it back or running a relevant command."""


@dataclass
class ServerConfig:
    """A single llama.cpp server endpoint."""

    host: str
    port: int
    context_length: int | None = None


@dataclass
class AppConfig:
    """Application configuration."""

    servers: list[ServerConfig]
    system_prompt: str
    max_tokens: int
    temperature: float
    context_length: int  # global fallback


def parse_server(entry: str) -> ServerConfig:
    """Parse a server entry like 'host:port' or 'host:port:context_length'.

    Raises:
        ValueError: If the entry is not in a recognised format or contains
                    non-numeric port/context_length values.
    """
    entry = entry.strip()
    parts = entry.split(":")
    if len(parts) == 2:
        host, port_str = parts
        try:
            return ServerConfig(host=host, port=int(port_str))
        except ValueError:
            raise ValueError(
                f"Invalid server entry: {entry!r}. Port must be an integer."
            )
    elif len(parts) == 3:
        host, port_str, ctx_str = parts
        try:
            return ServerConfig(host=host, port=int(port_str), context_length=int(ctx_str))
        except ValueError:
            raise ValueError(
                f"Invalid server entry: {entry!r}. Port and context_length must be integers."
            )
    else:
        raise ValueError(
            f"Invalid server entry: {entry!r}. Expected host:port or host:port:context_length"
        )


def load_config() -> AppConfig:
    """Load configuration from environment variables (after loading .env).

    Environment variables take precedence over values in the .env file.
    Falls back to sensible defaults when variables are not set.
    """
    load_dotenv(ENV_PATH)

    servers_str = os.getenv("LLAMA_SERVERS", "localhost:8082,localhost:8081")
    servers = [parse_server(s) for s in servers_str.split(",")]

    return AppConfig(
        servers=servers,
        system_prompt=os.getenv("LLAMA_SYSTEM_PROMPT", "You are a helpful assistant."),
        max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "8192")),
        temperature=float(os.getenv("LLAMA_TEMPERATURE", "1.0")),
        context_length=int(os.getenv("LLAMA_CONTEXT_LENGTH", "4096")),
    )


def _prompt_servers_manually() -> list[str]:
    """Prompt user to enter server addresses one at a time."""
    servers: list[str] = []
    print("  Enter server addresses (host:port). Empty line to finish.\n")
    i = 1
    while True:
        entry = input(f"  Server {i}: ").strip()
        if not entry:
            if servers:
                break
            print("  At least one server is required.")
            continue
        servers.append(entry)
        i += 1
    return servers


def _probe_server(host: str, port: int) -> str | None:
    """Probe a server for its model name. Returns model ID or None."""
    import httpx

    try:
        resp = httpx.get(f"http://{host}:{port}/v1/models", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except (httpx.ConnectError, httpx.ConnectTimeout, Exception):
        pass
    return None


def run_first_time_setup() -> AppConfig:
    """Interactive first-run setup. Uses plain input()/print() — no rich.

    Probes default servers to show what's running, then lets the user
    confirm or manually configure.
    """
    print("Welcome to llama-chat! Let's set up your configuration.\n")

    # Probe default servers
    defaults = [("localhost", 8082), ("localhost", 8081)]
    print("Scanning for llama.cpp servers...\n")

    detected: list[tuple[str, int, str | None]] = []
    for host, port in defaults:
        model = _probe_server(host, port)
        detected.append((host, port, model))

    # Show what we found
    any_online = any(m is not None for _, _, m in detected)
    if any_online:
        print("  Detected servers:")
        for host, port, model in detected:
            if model:
                print(f"    {model} on {host}:{port}")
            else:
                print(f"    {host}:{port} (offline)")

        use_detected = input("\n  Use these servers? (y/n) [y]: ").strip().lower()
        if use_detected in ("", "y", "yes"):
            servers_list = [f"{h}:{p}" for h, p, _ in detected]
        else:
            servers_list = _prompt_servers_manually()
    else:
        print("  No servers detected on default ports (8082, 8081).")
        print("  Enter your server addresses manually.\n")
        servers_list = _prompt_servers_manually()

    # Allow adding more
    while True:
        more = input("  Add another server? (y/n) [n]: ").strip().lower()
        if more in ("y", "yes"):
            entry = input("  Server address (host:port): ").strip()
            if entry:
                servers_list.append(entry)
        else:
            break

    servers_str = ",".join(servers_list)

    # Collect other settings
    system_prompt = (
        input("\nSystem prompt [You are a helpful assistant.]: ").strip()
        or "You are a helpful assistant."
    )

    temp_str = input("Temperature (0.0-2.0) [1.0]: ").strip() or "1.0"
    max_tokens_str = input("Max tokens per response [8192]: ").strip() or "8192"
    context_str = input("Fallback context length [4096]: ").strip() or "4096"

    # Write .env
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"LLAMA_SERVERS={servers_str}",
        f"LLAMA_SYSTEM_PROMPT={system_prompt}",
        f"LLAMA_MAX_TOKENS={max_tokens_str}",
        f"LLAMA_TEMPERATURE={temp_str}",
        f"LLAMA_CONTEXT_LENGTH={context_str}",
    ]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nConfig saved to {ENV_PATH}")

    return load_config()
