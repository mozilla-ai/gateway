import os
from pathlib import Path

import pytest

import gateway.core.config as config_module
from gateway.core.config import load_config


def test_load_config_loads_provider_env_vars_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\nGATEWAY_MASTER_KEY=gateway-master\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_MASTER_KEY", raising=False)

    config = load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "from-dotenv"
    assert config.master_key == "gateway-master"


def test_load_config_does_not_override_existing_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")

    load_config()

    assert os.getenv("ANTHROPIC_API_KEY") == "already-set"


def test_load_config_prefers_dotenv_near_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("ANTHROPIC_API_KEY=from-config-dir\n", encoding="utf-8")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-cwd\n", encoding="utf-8")
    config_file = config_dir / "gateway.yml"
    config_file.write_text("providers:\n  anthropic:\n    api_key: ${ANTHROPIC_API_KEY}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = load_config(str(config_file))

    assert os.getenv("ANTHROPIC_API_KEY") == "from-config-dir"
    assert config.providers["anthropic"]["api_key"] == "from-config-dir"


def test_load_config_skips_duplicate_dotenv_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GATEWAY_MASTER_KEY=gateway-master\n", encoding="utf-8")
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    calls: list[Path] = []

    def _fake_load_dotenv(*, dotenv_path: Path, override: bool) -> None:
        calls.append(dotenv_path)
        assert override is False

    monkeypatch.setattr(config_module, "load_dotenv", _fake_load_dotenv)

    load_config(str(config_file))

    assert calls == [env_file]


def test_load_config_platform_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: platform\n", encoding="utf-8")

    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "gw_test_token")
    monkeypatch.setenv("PLATFORM_BASE_URL", "http://localhost:8100/api/v1")
    monkeypatch.setenv("PLATFORM_RESOLVE_TIMEOUT_MS", "1234")
    monkeypatch.setenv("PLATFORM_USAGE_TIMEOUT_MS", "2345")
    monkeypatch.setenv("PLATFORM_USAGE_MAX_RETRIES", "7")

    config = load_config(str(config_file))

    assert config.is_platform_mode
    assert config.platform["base_url"] == "http://localhost:8100/api/v1"
    assert config.platform["resolve_timeout_ms"] == 1234
    assert config.platform["usage_timeout_ms"] == 2345
    assert config.platform["usage_max_retries"] == 7


def test_load_config_rejects_platform_mode_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: platform\n", encoding="utf-8")
    monkeypatch.delenv("ANY_LLM_PLATFORM_TOKEN", raising=False)

    with pytest.raises(ValueError, match="ANY_LLM_PLATFORM_TOKEN"):
        load_config(str(config_file))


def test_load_config_prefers_platform_mode_when_token_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mode: standalone\n", encoding="utf-8")
    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "gw_test_token")

    config = load_config(str(config_file))

    assert config.mode == "standalone"
    assert config.is_platform_mode
