"""DeepSeek API 客户端模块"""

import json
import time
from typing import Any

from openai import APIError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import Config


class DeepSeekClient:
    """DeepSeek API 客户端，封装 API 调用和重试逻辑。"""

    def __init__(self, config: Config) -> None:
        """初始化客户端。

        Args:
            config: 全局配置对象。
        """
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
        )
        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """速率限制：确保调用间隔不小于配置值。"""
        elapsed = time.time() - self._last_request_time
        min_gap = self._config.min_interval
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIError, TimeoutError)),
        reraise=True,
    )
    def analyze(self, prompt: str, logger: Any = None) -> dict[str, Any]:
        """调用 DeepSeek API 分析单条文本。

        Args:
            prompt: 完整 Prompt 文本。
            logger: Logger 实例，用于记录重试日志。

        Returns:
            解析后的 JSON 响应，包含 keywords、sentiment_score、reason。

        Raises:
            ValueError: API 返回格式异常。
            APIError: API 调用失败。
        """
        self._rate_limit_wait()

        response = self._client.chat.completions.create(
            model=self._config.api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.api_temperature,
            max_tokens=self._config.api_max_tokens,
        )

        self._last_request_time = time.time()

        # 解析返回内容
        content = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块标记
        if content.startswith("```"):
            lines = content.splitlines()
            # 去掉第一行 (```json / ```)
            lines = lines[1:]
            # 去掉最后一行 (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            result: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"API 返回内容无法解析为 JSON: {content[:200]}"
            ) from exc

        # 验证必要字段
        if "keywords" not in result:
            result["keywords"] = []
        if "sentiment_score" not in result:
            result["sentiment_score"] = 0
        if "reason" not in result:
            result["reason"] = ""

        return result

    def analyze_batch(
        self, batch_prompts: list[str], logger: Any = None
    ) -> list[dict[str, Any]]:
        """批量分析多条文本（顺序调用，受速率限制）。

        Args:
            batch_prompts: Prompt 列表。
            logger: Logger 实例。

        Returns:
            分析结果列表。
        """
        results: list[dict[str, Any]] = []
        for prompt in batch_prompts:
            try:
                result = self.analyze(prompt, logger)
                results.append(result)
            except Exception as exc:
                if logger:
                    logger.error(f"API 调用失败: {exc}")
                # 返回默认值标记失败
                results.append({
                    "keywords": [],
                    "sentiment_score": -1,
                    "reason": f"API 调用失败: {exc}",
                })
        return results
