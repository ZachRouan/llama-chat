import os
from config import ServerConfig, AppConfig, parse_server, load_config, ENV_PATH


def test_parse_server_host_port():
    result = parse_server("localhost:8082")
    assert result == ServerConfig(host="localhost", port=8082, context_length=None)


def test_parse_server_host_port_context():
    result = parse_server("localhost:8081:100000")
    assert result == ServerConfig(host="localhost", port=8081, context_length=100000)


def test_parse_server_strips_whitespace():
    result = parse_server("  localhost:8082  ")
    assert result == ServerConfig(host="localhost", port=8082, context_length=None)


def test_parse_server_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_server("invalid")
    with pytest.raises(ValueError):
        parse_server("host:notanumber")


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVERS", "localhost:8082, localhost:8081:100000")
    monkeypatch.setenv("LLAMA_SYSTEM_PROMPT", "Be helpful.")
    monkeypatch.setenv("LLAMA_MAX_TOKENS", "1024")
    monkeypatch.setenv("LLAMA_TEMPERATURE", "0.5")
    monkeypatch.setenv("LLAMA_CONTEXT_LENGTH", "8192")

    config = load_config()

    assert len(config.servers) == 2
    assert config.servers[0] == ServerConfig("localhost", 8082, None)
    assert config.servers[1] == ServerConfig("localhost", 8081, 100000)
    assert config.system_prompt == "Be helpful."
    assert config.max_tokens == 1024
    assert config.temperature == 0.5
    assert config.context_length == 8192


def test_load_config_defaults(monkeypatch):
    for key in ["LLAMA_SERVERS", "LLAMA_SYSTEM_PROMPT", "LLAMA_MAX_TOKENS",
                "LLAMA_TEMPERATURE", "LLAMA_CONTEXT_LENGTH"]:
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert len(config.servers) == 2
    assert config.servers[0].host == "localhost"
    assert config.servers[0].port == 8082
    assert config.system_prompt == "You are a helpful assistant."
    assert config.max_tokens == 2048
    assert config.temperature == 0.7
    assert config.context_length == 4096


def test_env_path_is_xdg():
    from pathlib import Path
    expected = Path.home() / ".config" / "llama-chat" / ".env"
    assert ENV_PATH == expected
