"""Unit tests for the API client module"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.api_client import DeepSeekClient
from src.config_loader import Config


@pytest.fixture
def config() -> Config:
    """Provide a minimal test configuration."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    cfg = Config()
    return cfg


@pytest.fixture
def valid_json_response() -> str:
    """Valid API JSON response content."""
    return json.dumps({
        "keywords": ["Course Quality", "Complaint"],
        "sentiment_score": 8,
        "reason": "User is very dissatisfied with the course content quality",
    }, ensure_ascii=False)


@pytest.fixture
def markdown_json_response() -> str:
    """API response wrapped in a markdown code block."""
    content = json.dumps({
        "keywords": ["Slow Updates"],
        "sentiment_score": 5,
        "reason": "User complains about slow data updates",
    }, ensure_ascii=False)
    return f"```json\n{content}\n```"


def test_client_init(config: Config) -> None:
    """Test client initialization."""
    client = DeepSeekClient(config)
    assert client._config is not None
    assert client._client is not None


@patch("src.api_client.OpenAI")
def test_analyze_success(
    mock_openai: MagicMock, config: Config, valid_json_response: str
) -> None:
    """Test a successful analyze call."""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = valid_json_response
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    result = client.analyze("Analyze this text: The service was terrible")

    assert result["keywords"] == ["Course Quality", "Complaint"]
    assert result["sentiment_score"] == 8
    assert "course content" in result["reason"]


@patch("src.api_client.OpenAI")
def test_analyze_markdown_response(
    mock_openai: MagicMock, config: Config, markdown_json_response: str
) -> None:
    """Test handling of markdown-wrapped API responses."""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = markdown_json_response
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    result = client.analyze("Analyze this text")

    assert result["keywords"] == ["Slow Updates"]
    assert result["sentiment_score"] == 5


@patch("src.api_client.OpenAI")
def test_analyze_missing_fields(
    mock_openai: MagicMock, config: Config
) -> None:
    """Test fallback when response is missing fields."""
    incomplete_json = json.dumps({
        "keywords": ["Complaint"],
    })

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = incomplete_json
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    result = client.analyze("Analyze this text")

    assert result["keywords"] == ["Complaint"]
    assert result["sentiment_score"] == 0  # default value
    assert result["reason"] == ""  # default value


@patch("src.api_client.OpenAI")
def test_analyze_invalid_json(
    mock_openai: MagicMock, config: Config
) -> None:
    """Test behavior when JSON parsing fails."""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "This is not valid JSON"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    with pytest.raises(ValueError, match="could not be parsed as JSON"):
        client.analyze("Analyze this text")


@patch("src.api_client.OpenAI")
def test_analyze_batch(
    mock_openai: MagicMock, config: Config, valid_json_response: str
) -> None:
    """Test batch analysis."""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = valid_json_response
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    texts = ["Text 1", "Text 2", "Text 3"]
    prompts = [f"Analyze: {t}" for t in texts]
    results = client.analyze_batch(prompts)

    assert len(results) == 3
    for r in results:
        assert r["keywords"] == ["Course Quality", "Complaint"]
        assert r["sentiment_score"] == 8


@pytest.fixture(autouse=True)
def cleanup() -> None:
    yield
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]
