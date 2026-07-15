"""Prompt 管理模块"""

from pathlib import Path

from src.utils import get_project_root


class PromptManager:
    """管理 Prompt 模板的加载和渲染。"""

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        """初始化。

        Args:
            prompt_path: prompt.txt 路径，默认为 config/prompt.txt。
        """
        if prompt_path is None:
            prompt_path = get_project_root() / "config" / "prompt.txt"
        self._prompt_path = Path(prompt_path)
        self._template = self._load_template()

    def _load_template(self) -> str:
        """从文件加载 Prompt 模板。"""
        with open(self._prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def render(self, text: str) -> str:
        """将 {{TEXT}} 替换为实际文本。

        Args:
            text: 要分析的客服文本。

        Returns:
            替换后的完整 Prompt。
        """
        return self._template.replace("{{TEXT}}", text)

    def reload(self) -> None:
        """重新加载 Prompt 文件（热更新用）。"""
        self._template = self._load_template()
