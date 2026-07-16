"""Configuration loading module."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils import get_project_root


class Config:
    """Global configuration class that wraps config.yaml and .env loading."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize configuration.

        Args:
            config_path: Path to config.yaml, defaults to config/config.yaml.

        """
        root = get_project_root()
        self._root = root

        # Load .env
        dotenv_path = root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        # Load config.yaml
        if config_path is None:
            config_path = root / "config" / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f)

    @property
    def api_key(self) -> str:
        """DeepSeek API Key."""
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key or key == "sk-your-deepseek-api-key-here":
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
            "Please configure a valid API Key in the .env file."
            )
        return key

    @property
    def api_base_url(self) -> str:
        """DeepSeek API base URL."""
        return self._raw["api"]["base_url"]

    @property
    def api_model(self) -> str:
        """DeepSeek model name."""
        return self._raw["api"]["model"]

    @property
    def api_temperature(self) -> float:
        """LLM temperature setting."""
        return self._raw["api"]["temperature"]

    @property
    def api_max_tokens(self) -> int:
        """Max tokens per API response."""
        return self._raw["api"]["max_tokens"]

    @property
    def data_input(self) -> str:
        """Input CSV file path."""
        return str(self._root / self._raw["data"]["input"])

    @property
    def data_output(self) -> str:
        """Output CSV file path."""
        return str(self._root / self._raw["data"]["output"])

    @property
    def data_checkpoint(self) -> str:
        """Checkpoint JSON file path."""
        return str(self._root / self._raw["data"]["checkpoint"])

    @property
    def id_column(self) -> str:
        """Name of the ID column in input CSV."""
        return self._raw["data"]["id_column"]

    @property
    def text_column(self) -> str:
        """Name of the text column in input CSV."""
        return self._raw["data"]["text_column"]

    @property
    def batch_size(self) -> int:
        """Number of records per batch."""
        return self._raw["batch"]["size"]

    @property
    def rate_limit(self) -> float:
        """Max API requests per second."""
        return self._raw["batch"]["rate_limit"]

    @property
    def min_interval(self) -> float:
        """Minimum seconds between API calls."""
        return self._raw["batch"]["min_interval"]

    @property
    def retry_max_attempts(self) -> int:
        """Max retry attempts on API failure."""
        return self._raw["retry"]["max_attempts"]

    @property
    def retry_min_wait(self) -> int:
        """Minimum retry wait in seconds."""
        return self._raw["retry"]["min_wait"]

    @property
    def retry_max_wait(self) -> int:
        """Maximum retry wait in seconds."""
        return self._raw["retry"]["max_wait"]

    @property
    def logging_level(self) -> str:
        """Log level (DEBUG/INFO/WARNING/ERROR)."""
        return self._raw["logging"]["level"]

    @property
    def logging_file(self) -> str:
        """Log file path."""
        return str(self._root / self._raw["logging"]["file"])

    @property
    def logging_format(self) -> str:
        """Log format string."""
        return self._raw["logging"]["format"]

    @property
    def complaint_types(self) -> list[str]:
        """List of predefined complaint categories."""
        return self._raw["rules"]["complaint_types"]

    @property
    def sentiment_levels(self) -> dict[int, str]:
        """Sentiment score level definitions."""
        raw = self._raw["rules"]["sentiment_levels"]
        return {int(k): v for k, v in raw.items()}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path.

        Args:
            key: Config key path, e.g. "api.base_url".
            default: Default value if key is not found.

        Returns:
            Config value.

        """
        keys = key.split(".")
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
