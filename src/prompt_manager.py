"""Prompt management module."""

from pathlib import Path

from src.utils import get_project_root


class PromptManager:
    """Manages prompt template loading and rendering."""

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        """Initialize the prompt manager.

        Args:
            prompt_path: Path to prompt.txt, defaults to config/prompt.txt.

        """
        if prompt_path is None:
            prompt_path = get_project_root() / "config" / "prompt.txt"
        self._prompt_path = Path(prompt_path)
        self._template = self._load_template()

    def _load_template(self) -> str:
        """Load the prompt template from file."""
        with open(self._prompt_path, encoding="utf-8") as f:
            return f.read().strip()

    def render(self, text: str) -> str:
        """Replace {{TEXT}} with actual text.

        Args:
            text: The customer service text to analyze.

        Returns:
            The completed prompt with text substituted.

        """
        return self._template.replace("{{TEXT}}", text)

    def reload(self) -> None:
        """Reload the prompt file (for hot-reloading)."""
        self._template = self._load_template()
