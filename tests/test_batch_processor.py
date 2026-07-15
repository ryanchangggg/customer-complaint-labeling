"""批量处理模块的单元测试"""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.batch_processor import BatchProcessor
from src.config_loader import Config
from src.prompt_manager import PromptManager


@pytest.fixture
def sample_csv() -> str:
    """创建临时测试 CSV 文件。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "text"])
        writer.writerow(["1", "为什么一直没人处理我的退款？"])
        writer.writerow(["2", "谢谢客服，很满意。"])
        writer.writerow(["3", "你们服务太差了。"])
        writer.writerow(["4", "物流太慢，等了一个星期。"])
        writer.writerow(["5", "东西质量不错，好评。"])
        return f.name


@pytest.fixture
def temp_dir() -> str:
    """创建临时输出目录。"""
    path = tempfile.mkdtemp()
    return path


@pytest.fixture
def mock_config(sample_csv: str, temp_dir: str) -> Config:
    """提供 mock 配置。"""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config()
    config._raw["data"]["input"] = sample_csv
    config._raw["data"]["output"] = os.path.join(temp_dir, "results.csv")
    config._raw["data"]["checkpoint"] = os.path.join(temp_dir, "checkpoint.json")
    config._raw["batch"]["size"] = 2
    return config


@pytest.fixture
def mock_api_client() -> MagicMock:
    """提供 mock API 客户端，按顺序返回 5 条预设结果。"""
    client = MagicMock()
    results = [
        {"keywords": ["退款", "投诉"], "sentiment_score": 8, "reason": "用户对退款速度不满"},
        {"keywords": ["满意"], "sentiment_score": 0, "reason": "用户表达满意"},
        {"keywords": ["服务差"], "sentiment_score": 8, "reason": "用户抱怨服务差"},
        {"keywords": ["物流慢"], "sentiment_score": 5, "reason": "用户抱怨物流慢"},
        {"keywords": ["满意"], "sentiment_score": 0, "reason": "用户对质量满意"},
    ]
    client.analyze.side_effect = results
    return client


@pytest.fixture
def prompt_mgr() -> PromptManager:
    """提供 Prompt 管理器实例。"""
    return PromptManager()


def test_load_checkpoint_empty(mock_config: Config) -> None:
    """测试空检查点。"""
    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    results, processed_ids = processor.load_checkpoint("/nonexistent/checkpoint.json")
    assert results == []
    assert processed_ids == set()


def test_load_checkpoint_some(mock_config: Config, temp_dir: str) -> None:
    """测试存在检查点。"""
    checkpoint_path = mock_config.data_checkpoint
    checkpoint_data = [
        {"id": "1", "text": "test", "keywords": "退款", "sentiment_score": 8, "sentiment_reason": "reason"},
        {"id": "2", "text": "test2", "keywords": "满意", "sentiment_score": 0, "sentiment_reason": "good"},
    ]
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)

    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    results, processed_ids = processor.load_checkpoint(checkpoint_path)
    assert len(results) == 2
    assert "1" in processed_ids
    assert "2" in processed_ids


def test_run_basic(
    mock_config: Config, mock_api_client: MagicMock, prompt_mgr: PromptManager,
) -> None:
    """测试基本运行流程。"""
    processor = BatchProcessor(mock_config, mock_api_client, prompt_mgr)
    output_path = processor.run()
    assert os.path.exists(output_path)

    import pandas as pd
    df = pd.read_csv(output_path)
    assert len(df) == 5
    assert list(df.columns) == ["id", "text", "keywords", "sentiment_score", "sentiment_reason"]

    row1 = df[df["id"].astype(str) == "1"].iloc[0]
    assert "退款" in str(row1["keywords"])
    assert row1["sentiment_score"] == 8

    row2 = df[df["id"].astype(str) == "2"].iloc[0]
    assert row2["sentiment_score"] == 0


def test_run_checkpoint_resume(
    mock_config: Config, prompt_mgr: PromptManager, temp_dir: str,
) -> None:
    """测试断点续跑功能。
    
    检查点已有 ID 1,2 的结果，mock 只提供 3 条结果对应 ID 3,4,5。
    """
    # 创建指定结果的 mock，匹配待处理的 3 条
    api_client = MagicMock()
    api_client.analyze.side_effect = [
        {"keywords": ["服务差"], "sentiment_score": 8, "reason": "用户抱怨服务差"},
        {"keywords": ["物流慢"], "sentiment_score": 5, "reason": "用户抱怨物流慢"},
        {"keywords": ["满意"], "sentiment_score": 0, "reason": "用户对质量满意"},
    ]

    checkpoint_path = mock_config.data_checkpoint
    checkpoint_data = [
        {"id": "1", "text": "为什么一直没人处理我的退款？", "keywords": "退款;投诉",
         "sentiment_score": 8, "sentiment_reason": "用户对退款速度不满"},
        {"id": "2", "text": "谢谢客服，很满意。", "keywords": "满意",
         "sentiment_score": 0, "sentiment_reason": "用户表达满意"},
    ]
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)

    processor = BatchProcessor(mock_config, api_client, prompt_mgr)
    output_path = processor.run()

    import pandas as pd
    df = pd.read_csv(output_path)

    # 输出应该包含全部 5 条
    assert len(df) == 5

    # 前2条从检查点恢复
    row1 = df[df["id"].astype(str) == "1"].iloc[0]
    assert "退款;投诉" in str(row1["keywords"])
    assert row1["sentiment_score"] == 8

    # 后3条是新处理的
    row3 = df[df["id"].astype(str) == "3"].iloc[0]
    assert "服务差" in str(row3["keywords"])
    assert row3["sentiment_score"] == 8

    row4 = df[df["id"].astype(str) == "4"].iloc[0]
    assert "物流慢" in str(row4["keywords"])
    assert row4["sentiment_score"] == 5


def test_run_empty_input(mock_config: Config) -> None:
    """测试空输入文件。"""
    empty_csv = mock_config.data_input
    with open(empty_csv, "w", encoding="utf-8") as f:
        f.write("id,text\n")

    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="输入文件为空"):
        processor.run()


@pytest.fixture(autouse=True)
def cleanup_env() -> None:
    yield
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]
