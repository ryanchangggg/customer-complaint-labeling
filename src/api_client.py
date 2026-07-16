"""DeepSeek API client module."""

import json
import time
from typing import Any

from openai import APIError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import Config


class DeepSeekClient:
    """DeepSeek API client that wraps API calls and retry logic."""

    def __init__(self, config: Config) -> None:
        """Initialize the client.

        Args:
            config: Global configuration object.

        """
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
        )
        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between API calls."""
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
        """Call the DeepSeek API to analyze a single text.

        Args:
            prompt: The complete prompt text.
            logger: Logger instance for recording retry logs.

        Returns:
            Parsed JSON response containing keywords, sentiment_score, reason.

        Raises:
            ValueError: API response format is abnormal.
            APIError: API call failed.

        """
        self._rate_limit_wait()

        response = self._client.chat.completions.create(
            model=self._config.api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.api_temperature,
            max_tokens=self._config.api_max_tokens,
        )

        self._last_request_time = time.time()

        # Parse response content
        content = (response.choices[0].message.content or "").strip()

        # Clean possible markdown code block markers
        if content.startswith("```"):
            lines = content.splitlines()
            # Remove first line (```json / ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            result: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"API response could not be parsed as JSON: {content[:200]}"
            ) from exc

        # Validate required fields
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
        """Analyze multiple texts in batch (sequential calls, rate-limited).

        Args:
            batch_prompts: List of prompts.
            logger: Logger instance.

        Returns:
            List of analysis results.

        """
        results: list[dict[str, Any]] = []
        for prompt in batch_prompts:
            try:
                result = self.analyze(prompt, logger)
                results.append(result)
            except Exception as exc:
                if logger:
                    logger.error(f"API call failed: {exc}")
                # Return default values to mark the failure
                results.append(
                    {
                        "keywords": [],
                        "sentiment_score": -1,
                        "reason": f"API call failed: {exc}",
                    }
                )
        return results
