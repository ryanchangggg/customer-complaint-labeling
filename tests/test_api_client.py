"""API 客户端模块的单元测试"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.api_client import DeepSeekClient
from src.config_loader import Config


@pytest.fixture
def config() -> Config:
    """提供最小化测试配置。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    # 直接用默认配置文件的路径，测试需要 mock API 调用
    cfg = Config()
    return cfg


@pytest.fixture
def valid_json_response() -> str:
    """有效的 API JSON 返回内容。"""
    return json.dumps({
        "keywords": ["退款", "投诉"],
        "sentiment_score": 8,
        "reason": "用户对于退款速度非常不满",
    }, ensure_ascii=False)


@pytest.fixture
def markdown_json_response() -> str:
    """带 markdown 代码块的 API 返回内容。"""
    content = json.dumps({
        "keywords": ["物流慢"],
        "sentiment_score": 5,
        "reason": "用户抱怨物流速度慢",
    }, ensure_ascii=False)
    return f"```json\n{content}\n```"


def test_client_init(config: Config) -> None:
    """测试客户端初始化。"""
    client = DeepSeekClient(config)
    assert client._config is not None
    assert client._client is not None


@patch("src.api_client.OpenAI")
def test_analyze_success(
    mock_openai: MagicMock, config: Config, valid_json_response: str
) -> None:
    """测试成功调用 analyze。"""
    # Mock API 返回
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
    result = client.analyze("请分析文本：服务太差了")

    assert result["keywords"] == ["退款", "投诉"]
    assert result["sentiment_score"] == 8
    assert "退款速度" in result["reason"]


@patch("src.api_client.OpenAI")
def test_analyze_markdown_response(
    mock_openai: MagicMock, config: Config, markdown_json_response: str
) -> None:
    """测试处理带 markdown 标记的 API 返回。"""
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
    result = client.analyze("请分析文本")

    assert result["keywords"] == ["物流慢"]
    assert result["sentiment_score"] == 5


@patch("src.api_client.OpenAI")
def test_analyze_missing_fields(
    mock_openai: MagicMock, config: Config
) -> None:
    """测试缺少字段时的容错。"""
    incomplete_json = json.dumps({
        "keywords": ["投诉"],
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
    result = client.analyze("请分析文本")

    assert result["keywords"] == ["投诉"]
    assert result["sentiment_score"] == 0  # 默认值
    assert result["reason"] == ""  # 默认值


@patch("src.api_client.OpenAI")
def test_analyze_invalid_json(
    mock_openai: MagicMock, config: Config
) -> None:
    """测试 JSON 解析失败。"""
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "这不是合法的 JSON"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client_instance

    client = DeepSeekClient(config)
    with pytest.raises(ValueError, match="无法解析为 JSON"):
        client.analyze("请分析文本")


@patch("src.api_client.OpenAI")
def test_analyze_batch(
    mock_openai: MagicMock, config: Config, valid_json_response: str
) -> None:
    """测试批量分析。"""
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
    texts = ["文本1", "文本2", "文本3"]
    prompts = [f"请分析：{t}" for t in texts]
    results = client.analyze_batch(prompts)

    assert len(results) == 3
    for r in results:
        assert r["keywords"] == ["退款", "投诉"]
        assert r["sentiment_score"] == 8


# 清理环境变量
@pytest.fixture(autouse=True)
def cleanup() -> None:
    yield
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]
