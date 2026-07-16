"""Unit tests for the configuration loader module."""

import os
import tempfile

import pytest
import yaml

from src.config_loader import Config


@pytest.fixture
def sample_config() -> dict:
    """Provide a valid test configuration."""
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
            "complaint_types": ["Course/Teaching Quality", "Investment Advice/Losses"],
            "sentiment_levels": {
                "0": "Satisfied",
                "2": "Neutral",
                "5": "Slightly Dissatisfied",
                "8": "Complaint",
                "10": "Extreme Anger",
            },
        },
    }


@pytest.fixture
def config_file(sample_config: dict) -> str:
    """Create a temporary config file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(sample_config, f, allow_unicode=True)
        return f.name


def test_config_basic(config_file: str) -> None:
    """Test basic configuration loading."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.api_base_url == "https://api.deepseek.com"
    assert config.api_model == "deepseek-chat"
    assert config.api_temperature == 0.1
    assert config.api_max_tokens == 512


def test_config_data_paths(config_file: str) -> None:
    """Test data path properties."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert "data/sample_chat.csv" in config.data_input
    assert "output/results.csv" in config.data_output
    assert "output/checkpoint.json" in config.data_checkpoint


def test_config_batch_settings(config_file: str) -> None:
    """Test batch processing configuration."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.batch_size == 20
    assert config.rate_limit == 5.0
    assert config.min_interval == 0.2


def test_config_rules(config_file: str) -> None:
    """Test business rules configuration."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert "Course/Teaching Quality" in config.complaint_types
    assert config.sentiment_levels[0] == "Satisfied"
    assert config.sentiment_levels[8] == "Complaint"


def test_config_get_method(config_file: str) -> None:
    """Test the get method."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config(config_file)

    assert config.get("api.base_url") == "https://api.deepseek.com"
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("nonexistent.key") is None


def test_config_api_key_missing(config_file: str) -> None:
    """Test error when API Key is missing."""
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]

    config = Config(config_file)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = config.api_key


def test_config_file_not_found() -> None:
    """Test error when config file does not exist."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    with pytest.raises(FileNotFoundError):
        Config("/nonexistent/path/config.yaml")


@pytest.fixture(autouse=True)
def cleanup(request: pytest.FixtureRequest) -> None:
    """Clean up temporary files after tests."""
    def _cleanup() -> None:
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
    request.addfinalizer(_cleanup)
