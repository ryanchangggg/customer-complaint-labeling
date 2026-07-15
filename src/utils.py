"""工具函数模块"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """配置并返回一个 logger 实例。

    Args:
        name: Logger 名称。
        level: 日志级别。
        log_file: 日志文件路径，为 None 则仅输出到控制台。
        fmt: 日志格式。

    Returns:
        配置完成的 Logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    Args:
        path: 目录路径。

    Returns:
        目录的 Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, **kwargs: Any) -> None:
    """保存 JSON 文件。

    Args:
        data: 要保存的数据。
        path: 文件路径。
        kwargs: 传递给 json.dump 的额外参数。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, **kwargs)


def load_json(path: str | Path) -> Any:
    """加载 JSON 文件。

    Args:
        path: 文件路径。

    Returns:
        解析后的数据。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_project_root() -> Path:
    """返回项目根目录 (src/ 的父目录)。

    Returns:
        项目根目录的 Path 对象。
    """
    return Path(__file__).resolve().parent.parent
