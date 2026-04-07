import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "X-AnyLLM-Key"


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite:///./any-llm-gateway.db",
        description="Database connection URL (SQLite default for local use; PostgreSQL recommended for production)",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Automatically run database migrations on startup",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")  # noqa: S104
    port: int = Field(default=8000, description="Port to bind the server to")
    master_key: str | None = Field(default=None, description="Master key for protecting management endpoints")
    rate_limit_rpm: int | None = Field(
        default=None, ge=1, description="Maximum requests per minute per user (None disables rate limiting)"
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="Allowed CORS origins (empty list disables CORS)"
    )
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Pre-configured provider credentials"
    )
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description=(
            "Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})"
        ),
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint at /metrics",
    )
    bootstrap_api_key: bool = Field(
        default=True,
        description="Create a first-use API key on startup when no API keys exist",
    )
    mode: str = Field(default="standalone", description="Gateway operating mode: standalone or platform")
    platform: dict[str, Any] = Field(default_factory=dict, description="Platform integration settings")

    @property
    def platform_token(self) -> str | None:
        token = os.getenv("ANY_LLM_PLATFORM_TOKEN", "").strip()
        return token if token else None

    @property
    def effective_mode(self) -> str:
        if self.platform_token:
            return "platform"
        return "standalone"

    @property
    def is_platform_mode(self) -> bool:
        return self.effective_mode == "platform"

    def validate_mode_selection(self) -> None:
        configured_mode = self.mode.strip().lower()
        if configured_mode not in {"standalone", "platform"}:
            msg = "Invalid GATEWAY_MODE value. Expected 'standalone' or 'platform'."
            raise ValueError(msg)

        token_present = self.platform_token is not None
        if configured_mode == "platform" and not token_present:
            msg = "GATEWAY_MODE=platform requires ANY_LLM_PLATFORM_TOKEN to be set."
            raise ValueError(msg)

        if configured_mode == "standalone" and token_present:
            msg = (
                "ANY_LLM_PLATFORM_TOKEN is set but GATEWAY_MODE=standalone. "
                "Remove the token or switch to platform mode."
            )
            raise ValueError(msg)


def load_config(config_path: str | None = None) -> GatewayConfig:
    """Load configuration from file and environment variables.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        GatewayConfig instance with merged configuration

    """
    _load_dotenv(config_path)

    config_dict: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = _resolve_env_vars(yaml_config)

    _apply_platform_env_overrides(config_dict)

    config = GatewayConfig(**config_dict)
    config.validate_mode_selection()
    return config


def _apply_platform_env_overrides(config: dict[str, Any]) -> None:
    platform = config.get("platform")
    if not isinstance(platform, dict):
        platform = {}

    env_mappings: dict[str, tuple[str, type[Any]]] = {
        "PLATFORM_BASE_URL": ("base_url", str),
        "PLATFORM_RESOLVE_TIMEOUT_MS": ("resolve_timeout_ms", int),
        "PLATFORM_USAGE_TIMEOUT_MS": ("usage_timeout_ms", int),
        "PLATFORM_USAGE_MAX_RETRIES": ("usage_max_retries", int),
    }

    for env_name, (field_name, caster) in env_mappings.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        platform[field_name] = caster(value)

    if platform:
        config["platform"] = platform


def _load_dotenv(config_path: str | None = None) -> None:
    """Load .env files into process environment without overriding existing vars."""
    candidate_paths: list[Path] = [Path.cwd() / ".env"]
    if config_path:
        candidate_paths.insert(0, Path(config_path).resolve().parent / ".env")

    seen: set[Path] = set()
    for dotenv_path in candidate_paths:
        if dotenv_path in seen or not dotenv_path.exists():
            continue
        seen.add(dotenv_path)
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve environment variable references in config.

    Supports ${VAR_NAME} syntax in string values.

    Raises:
        ValueError: If an environment variable reference cannot be resolved

    """
    if isinstance(config, dict):
        return {key: _resolve_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:

        def _replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                msg = f"Environment variable '{env_var}' is not set (referenced in config as '${{{env_var}}}')"
                raise ValueError(msg)
            return value

        return re.sub(r"\$\{([^}]+)}", _replace, config)
    return config
