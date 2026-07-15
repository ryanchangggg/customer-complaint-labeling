"""配置加载模块"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils import get_project_root


class Config:
    """全局配置类，封装 config.yaml 和 .env 的加载。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """初始化配置。

        Args:
            config_path: config.yaml 路径，默认为 config/config.yaml。
        """
        root = get_project_root()
        self._root = root

        # 加载 .env
        dotenv_path = root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        # 加载 config.yaml
        if config_path is None:
            config_path = root / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f)

    @property
    def api_key(self) -> str:
        """DeepSeek API Key。"""
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key or key == "sk-your-deepseek-api-key-here":
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中配置有效的 API Key。"
            )
        return key

    @property
    def api_base_url(self) -> str:
        return self._raw["api"]["base_url"]

    @property
    def api_model(self) -> str:
        return self._raw["api"]["model"]

    @property
    def api_temperature(self) -> float:
        return self._raw["api"]["temperature"]

    @property
    def api_max_tokens(self) -> int:
        return self._raw["api"]["max_tokens"]

    @property
    def data_input(self) -> str:
        return str(self._root / self._raw["data"]["input"])

    @property
    def data_output(self) -> str:
        return str(self._root / self._raw["data"]["output"])

    @property
    def data_checkpoint(self) -> str:
        return str(self._root / self._raw["data"]["checkpoint"])

    @property
    def id_column(self) -> str:
        return self._raw["data"]["id_column"]

    @property
    def text_column(self) -> str:
        return self._raw["data"]["text_column"]

    @property
    def batch_size(self) -> int:
        return self._raw["batch"]["size"]

    @property
    def rate_limit(self) -> float:
        return self._raw["batch"]["rate_limit"]

    @property
    def min_interval(self) -> float:
        return self._raw["batch"]["min_interval"]

    @property
    def retry_max_attempts(self) -> int:
        return self._raw["retry"]["max_attempts"]

    @property
    def retry_min_wait(self) -> int:
        return self._raw["retry"]["min_wait"]

    @property
    def retry_max_wait(self) -> int:
        return self._raw["retry"]["max_wait"]

    @property
    def logging_level(self) -> str:
        return self._raw["logging"]["level"]

    @property
    def logging_file(self) -> str:
        return str(self._root / self._raw["logging"]["file"])

    @property
    def logging_format(self) -> str:
        return self._raw["logging"]["format"]

    @property
    def complaint_types(self) -> list[str]:
        return self._raw["rules"]["complaint_types"]

    @property
    def sentiment_levels(self) -> dict[int, str]:
        raw = self._raw["rules"]["sentiment_levels"]
        return {int(k): v for k, v in raw.items()}

    def get(self, key: str, default: Any = None) -> Any:
        """通过点号分隔的路径获取配置值。

        Args:
            key: 配置键路径，如 "api.base_url"。
            default: 未找到时的默认值。

        Returns:
            配置值。
        """
        keys = key.split(".")
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
