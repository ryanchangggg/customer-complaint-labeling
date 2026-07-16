"""Unit tests for the batch processing module."""

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
    """Create a temporary test CSV file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "text"])
        writer.writerow(["1", "Why hasn't anyone handled my refund?"])
        writer.writerow(["2", "Thank you, very satisfied."])
        writer.writerow(["3", "Your service is terrible."])
        writer.writerow(["4", "The data updates are too slow, waited a week."])
        writer.writerow(["5", "Course quality is decent, good review."])
        return f.name


@pytest.fixture
def temp_dir() -> str:
    """Create a temporary output directory."""
    path = tempfile.mkdtemp()
    return path


@pytest.fixture
def mock_config(sample_csv: str, temp_dir: str) -> Config:
    """Provide a mock configuration."""
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
    config = Config()
    config._raw["data"]["input"] = sample_csv
    config._raw["data"]["output"] = os.path.join(temp_dir, "results.csv")
    config._raw["data"]["checkpoint"] = os.path.join(temp_dir, "checkpoint.json")
    config._raw["batch"]["size"] = 2
    return config


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Provide a mock API client that returns 5 preset results in order."""
    client = MagicMock()
    results = [
        {
            "keywords": ["Refund", "Complaint"],
            "sentiment_score": 8,
            "reason": "User is dissatisfied with refund speed",
        },
        {
            "keywords": ["Satisfied"],
            "sentiment_score": 0,
            "reason": "User expresses satisfaction",
        },
        {
            "keywords": ["Poor Service"],
            "sentiment_score": 8,
            "reason": "User complains about poor service",
        },
        {
            "keywords": ["Slow Updates"],
            "sentiment_score": 5,
            "reason": "User complains about slow updates",
        },
        {
            "keywords": ["Satisfied"],
            "sentiment_score": 0,
            "reason": "User is satisfied with quality",
        },
    ]
    client.analyze.side_effect = results
    return client


@pytest.fixture
def prompt_mgr() -> PromptManager:
    """Provide a PromptManager instance."""
    return PromptManager()


def test_load_checkpoint_empty(mock_config: Config) -> None:
    """Test loading an empty checkpoint."""
    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    results, processed_ids = processor.load_checkpoint("/nonexistent/checkpoint.json")
    assert results == []
    assert processed_ids == set()


def test_load_checkpoint_some(mock_config: Config, temp_dir: str) -> None:
    """Test loading a checkpoint with existing data."""
    checkpoint_path = mock_config.data_checkpoint
    checkpoint_data = [
        {
            "id": "1",
            "text": "test",
            "keywords": "Refund",
            "sentiment_score": 8,
            "sentiment_reason": "reason",
        },
        {
            "id": "2",
            "text": "test2",
            "keywords": "Satisfied",
            "sentiment_score": 0,
            "sentiment_reason": "good",
        },
    ]
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)

    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    results, processed_ids = processor.load_checkpoint(checkpoint_path)
    assert len(results) == 2
    assert "1" in processed_ids
    assert "2" in processed_ids


def test_run_basic(
    mock_config: Config,
    mock_api_client: MagicMock,
    prompt_mgr: PromptManager,
) -> None:
    """Test the basic run flow."""
    processor = BatchProcessor(mock_config, mock_api_client, prompt_mgr)
    output_path = processor.run()
    assert os.path.exists(output_path)

    import pandas as pd

    df = pd.read_csv(output_path)
    assert len(df) == 5
    assert list(df.columns) == [
        "id",
        "text",
        "keywords",
        "sentiment_score",
        "sentiment_reason",
        "complaint_type",
    ]

    row1 = df[df["id"].astype(str) == "1"].iloc[0]
    assert "Refund" in str(row1["keywords"])
    assert row1["sentiment_score"] == 8

    row2 = df[df["id"].astype(str) == "2"].iloc[0]
    assert row2["sentiment_score"] == 0


def test_run_checkpoint_resume(
    mock_config: Config,
    prompt_mgr: PromptManager,
    temp_dir: str,
) -> None:
    """Test checkpoint resume functionality.

    Checkpoint already has results for IDs 1,2. Mock provides 3 results for IDs 3,4,5.
    """
    api_client = MagicMock()
    api_client.analyze.side_effect = [
        {
            "keywords": ["Poor Service"],
            "sentiment_score": 8,
            "reason": "User complains about poor service",
        },
        {
            "keywords": ["Slow Updates"],
            "sentiment_score": 5,
            "reason": "User complains about slow updates",
        },
        {
            "keywords": ["Satisfied"],
            "sentiment_score": 0,
            "reason": "User is satisfied with quality",
        },
    ]

    checkpoint_path = mock_config.data_checkpoint
    checkpoint_data = [
        {
            "id": "1",
            "text": "Why hasn't anyone handled my refund?",
            "keywords": "Refund;Complaint",
            "sentiment_score": 8,
            "sentiment_reason": "User is dissatisfied with refund speed",
        },
        {
            "id": "2",
            "text": "Thank you, very satisfied.",
            "keywords": "Satisfied",
            "sentiment_score": 0,
            "sentiment_reason": "User expresses satisfaction",
        },
    ]
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)

    processor = BatchProcessor(mock_config, api_client, prompt_mgr)
    output_path = processor.run()

    import pandas as pd

    df = pd.read_csv(output_path)

    # Output should contain all 5 records
    assert len(df) == 5

    # First 2 records restored from checkpoint
    row1 = df[df["id"].astype(str) == "1"].iloc[0]
    assert "Refund;Complaint" in str(row1["keywords"])
    assert row1["sentiment_score"] == 8

    # Last 3 records are newly processed
    row3 = df[df["id"].astype(str) == "3"].iloc[0]
    assert "Poor Service" in str(row3["keywords"])
    assert row3["sentiment_score"] == 8

    row4 = df[df["id"].astype(str) == "4"].iloc[0]
    assert "Slow Updates" in str(row4["keywords"])
    assert row4["sentiment_score"] == 5


def test_run_empty_input(mock_config: Config) -> None:
    """Test handling of an empty input file."""
    empty_csv = mock_config.data_input
    with open(empty_csv, "w", encoding="utf-8") as f:
        f.write("id,text\n")

    processor = BatchProcessor(mock_config, MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="Input file is empty"):
        processor.run()


@pytest.fixture(autouse=True)
def cleanup_env() -> None:
    """Clean up environment variables after tests."""
    yield
    if "DEEPSEEK_API_KEY" in os.environ:
        del os.environ["DEEPSEEK_API_KEY"]
