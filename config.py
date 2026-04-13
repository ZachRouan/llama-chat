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
        max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "2048")),
        temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.7")),
        context_length=int(os.getenv("LLAMA_CONTEXT_LENGTH", "4096")),
    )


def run_first_time_setup() -> AppConfig:
    """Interactive first-run setup. Uses plain input()/print() — no rich.

    Prompts the user for server addresses and settings, writes a .env file
    to CONFIG_DIR, then returns the resulting AppConfig.
    """
    print("Welcome to llama-chat! Let's set up your configuration.\n")

    # Collect servers
    servers_list: list[str] = []
    print("Enter your llama.cpp server addresses (host:port).")
    print("Press Enter to accept the default for the first two.\n")

    defaults = ["localhost:8082", "localhost:8081"]
    for i, default in enumerate(defaults):
        entry = input(f"  Server {i + 1} [{default}]: ").strip()
        servers_list.append(entry if entry else default)

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

    temp_str = input("Temperature (0.0-2.0) [0.7]: ").strip() or "0.7"
    max_tokens_str = input("Max tokens per response [2048]: ").strip() or "2048"
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
