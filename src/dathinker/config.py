"""
Configuration management for DaThinker.

Supports loading configuration from:
1. ~/.config/dathinker/config.toml (recommended for API keys)
2. Environment variables
3. Programmatic configuration

Security notes:
- Never print API keys in logs or output
- Never log HTTP headers containing authorization
- Prefer config file over shell history exposure
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

# Configure logging to never expose sensitive data
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DaThinkerConfig:
    """Configuration for DaThinker."""

    # API Configuration
    openrouter_api_key: Optional[str] = None

    # Model Configuration
    default_model: str = "balanced"
    fast_model: str = "google/gemma-2-9b-it"
    balanced_model: str = "meta-llama/llama-3.1-8b-instruct"
    reasoning_model: str = "openai/gpt-4o-mini"

    # Router Configuration
    default_router: str = "heuristic"  # heuristic, llm, hybrid

    # Session Configuration
    show_routing_traces: bool = True
    max_agents_per_turn: int = 3

    # Security
    redact_api_key_in_logs: bool = True

    @property
    def api_key_set(self) -> bool:
        """Check if API key is configured (without exposing it)."""
        return bool(self.openrouter_api_key)

    @property
    def api_key_preview(self) -> str:
        """Get a safe preview of the API key for debugging."""
        if not self.openrouter_api_key:
            return "(not set)"
        key = self.openrouter_api_key
        if len(key) > 8:
            return f"{key[:4]}...{key[-4:]}"
        return "****"


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Follow XDG Base Directory Specification
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "dathinker"

    return Path.home() / ".config" / "dathinker"


def get_config_file() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / "config.toml"


def load_toml_config(config_path: Path) -> dict:
    """Load configuration from TOML file."""
    if not config_path.exists():
        return {}

    try:
        # Try tomllib (Python 3.11+) first
        try:
            import tomllib
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            pass

        # Fall back to toml package
        try:
            import toml
            return toml.load(config_path)
        except ImportError:
            pass

        # Manual parsing for simple cases
        config = {}
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    config[key] = value
        return config

    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def load_config() -> DaThinkerConfig:
    """
    Load configuration from all sources.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file (~/.config/dathinker/config.toml)
    3. Default values
    """
    config = DaThinkerConfig()

    # Load from config file first
    config_file = get_config_file()
    file_config = load_toml_config(config_file)

    if file_config:
        # API key from config file
        if "openrouter_api_key" in file_config:
            config.openrouter_api_key = file_config["openrouter_api_key"]
        # Also check nested [api] section
        if "api" in file_config and isinstance(file_config["api"], dict):
            if "openrouter_key" in file_config["api"]:
                config.openrouter_api_key = file_config["api"]["openrouter_key"]

        # Model settings
        if "default_model" in file_config:
            config.default_model = file_config["default_model"]
        if "default_router" in file_config:
            config.default_router = file_config["default_router"]
        if "show_routing_traces" in file_config:
            config.show_routing_traces = str(file_config["show_routing_traces"]).lower() == "true"

    # Environment variables override config file
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        config.openrouter_api_key = env_key

    env_model = os.environ.get("DATHINKER_MODEL")
    if env_model:
        config.default_model = env_model

    env_router = os.environ.get("DATHINKER_ROUTER")
    if env_router:
        config.default_router = env_router

    return config


def create_default_config_file() -> Path:
    """Create a default configuration file with comments."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = get_config_file()

    default_config = '''# DaThinker Configuration
# This file is auto-generated. Edit as needed.

# API Configuration
# Store your API key here instead of environment variables for better security
# (avoids exposure in shell history)
# openrouter_api_key = "your-api-key-here"

# Default model tier: fast, balanced, reasoning
default_model = "balanced"

# Default router: heuristic, llm, hybrid
default_router = "heuristic"

# Show routing traces in adaptive mode
show_routing_traces = true

# [api]
# Alternative nested format for API keys
# openrouter_key = "your-api-key-here"
'''

    if not config_file.exists():
        with open(config_file, "w") as f:
            f.write(default_config)
        # Set restrictive permissions (owner read/write only)
        config_file.chmod(0o600)
        logger.info(f"Created config file: {config_file}")

    return config_file


def get_api_key() -> Optional[str]:
    """
    Get the API key from config or environment.

    Security: This function exists to centralize API key access.
    Never log or print the return value.
    """
    config = load_config()
    return config.openrouter_api_key


# Singleton config instance
_config: Optional[DaThinkerConfig] = None


def get_config() -> DaThinkerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the global configuration (for testing)."""
    global _config
    _config = None
