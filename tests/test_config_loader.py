"""配置加载模块的单元测试"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config_loader import Config


@pytest.fixture
def sample_config() -> dict:
    """提供有效的测试配置。"""
    return {
        "api": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.1,
            "max_tokens": 512,
        },
        "data": {
            "input": "data/sample_chat.csv",
            "output": "output/results.csv",
            "checkpoint": "output/checkpoint.json",
            "id_column": "id",
            "text_column": "text",
        },
        "batch": {
            "size": 20,
            "rate_limit": 5.0,
            "min_interval": 0.2,
        },
        "retry": {
            "max_attempts": 3,
            "min_wait": 2,
            "max_wait": 30,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/labeling.log",
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        },
        "rules": {
            "complaint_types": ["退款/退货", "物流/配送"],
            "sentiment_levels": {
                "0": "满意",
                "2": "一般",
                "5": "有点不满",
                "8": "投诉",
                "10": "极端愤怒",
            },
        },
    }


@pytest.fixture
def config_file(sample_config: dict) -> str:
    """创建临时配置文件并返回路径。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(sample_config, f, allow_unicode=True)
        return f.name


def test_config_basic(config_file: str) -> None:
    """测试基本配置加载。"""
    # 需要确保 .env 中设置了 API Key
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.api_base_url == "https://api.deepseek.com"
    assert config.api_model == "deepseek-chat"
    assert config.api_temperature == 0.1
    assert config.api_max_tokens == 512


def test_config_data_paths(config_file: str) -> None:
    """测试数据路径属性。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert "data/sample_chat.csv" in config.data_input
    assert "output/results.csv" in config.data_output
    assert "output/checkpoint.json" in config.data_checkpoint


def test_config_batch_settings(config_file: str) -> None:
    """测试批处理配置。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.batch_size == 20
    assert config.rate_limit == 5.0
    assert config.min_interval == 0.2


def test_config_rules(config_file: str) -> None:
    """测试业务规则配置。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert "退款/退货" in config.complaint_types
    assert config.sentiment_levels[0] == "满意"
    assert config.sentiment_levels[8] == "投诉"


def test_config_get_method(config_file: str) -> None:
    """测试 get 方法。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.get("api.base_url") == "https://api.deepseek.com"
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("nonexistent.key") is None


def test_config_api_key_missing(config_file: str) -> None:
    """测试 API Key 缺失时的错误。"""
    # 确保环境变量中不存在 API Key
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]

    config = Config(config_file)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = config.api_key


def test_config_file_not_found() -> None:
    """测试配置文件不存在的错误。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    with pytest.raises(FileNotFoundError):
        Config("/nonexistent/path/config.yaml")


# 清理临时文件
@pytest.fixture(autouse=True)
def cleanup(request: pytest.FixtureRequest) -> None:
    """测试后清理临时文件。"""
    def _cleanup() -> None:
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
    request.addfinalizer(_cleanup)
