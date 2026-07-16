"""Batch processing module — supports checkpoint resumption and progress bar"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.api_client import DeepSeekClient
from src.classifier import VALID_TYPES, classify_keywords
from src.config_loader import Config
from src.prompt_manager import PromptManager
from src.reporter import write_report


class BatchProcessor:
    """Batch processing controller that manages CSV reading, batched API calls,
    checkpoint resumption, and result output."""

    def __init__(
        self,
        config: Config,
        api_client: DeepSeekClient,
        prompt_manager: PromptManager,
        logger: Any = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            config: Global configuration.
            api_client: DeepSeek API client.
            prompt_manager: Prompt manager.
            logger: Logger instance.
        """
        self.config = config
        self.client = api_client
        self.prompt_mgr = prompt_manager
        self.logger = logger

    def load_checkpoint(self, path: str) -> tuple[list[dict[str, Any]], set[str]]:
        """Load a checkpoint, returning existing results and processed ID set.

        The checkpoint stores only {id, keywords, sentiment_score,
        sentiment_reason, complaint_type} — text is reconstructed from the
        original CSV at output time.

        Args:
            path: Checkpoint file path.

        Returns:
            (existing results list, processed ID set).
        """
        if not os.path.exists(path):
            return [], set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            processed_ids = {str(r[self.config.id_column]) for r in data}
            if self.logger:
                self.logger.info(
                    f"Checkpoint found, {len(processed_ids)} results already processed"
                )
            return data, processed_ids
        except (json.JSONDecodeError, KeyError) as exc:
            if self.logger:
                self.logger.warning(
                    f"Checkpoint file corrupted, restarting from scratch: {exc}"
                )
            return [], set()

    def save_checkpoint(
        self, path: str, results: list[dict[str, Any]]
    ) -> None:
        """Save a checkpoint — stores only metadata, no raw text.

        Args:
            path: Checkpoint file path.
            results: Results list (id + analysis fields only).
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Strip text column if present (backward compatibility)
        compact = []
        text_col = self.config.text_column
        for r in results:
            entry = {k: v for k, v in r.items() if k != text_col}
            compact.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)

    def _label_row(
        self, row_id: str, text: str
    ) -> dict[str, Any]:
        """Analyze a single row of text via the LLM, then apply fallback
        classification.

        Args:
            row_id: Record ID.
            text: Text content.

        Returns:
            Labeled result with complaint_type.
        """
        prompt = self.prompt_mgr.render(text)
        result = self.client.analyze(prompt, self.logger)

        kw_list = result.get("keywords", [])
        if isinstance(kw_list, str):
            kw_list = [kw_list]

        # Extract complaint_type from LLM response
        ctype = (result.get("complaint_type") or "").strip()
        if ctype not in VALID_TYPES:
            # Fallback: use keyword classifier
            fallback = classify_keywords(kw_list)
            if self.logger and ctype:
                self.logger.debug(
                    f"LLM returned invalid complaint_type={ctype!r}, "
                    f"falling back to keyword classifier → {fallback}"
                )
            ctype = fallback or ""

        return {
            self.config.id_column: row_id,
            "keywords": ";".join(kw_list),
            "sentiment_score": result.get("sentiment_score", -1),
            "sentiment_reason": result.get("reason", ""),
            "complaint_type": ctype,
        }

    def _build_output_rows(
        self,
        all_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge analysis results with the original CSV to reconstruct text.

        Checkpoint no longer stores raw text; this method reads the original
        CSV and joins on id.

        Args:
            all_results: Combined checkpoint + newly processed results.

        Returns:
            Rows ready for CSV output, each containing id, text, and
            all analysis fields.
        """
        # Read the original source data
        df_source = pd.read_csv(self.config.data_input)

        # Build a lookup by id
        result_map: dict[str, dict[str, Any]] = {}
        for r in all_results:
            result_map[str(r[self.config.id_column])] = r

        id_col = self.config.id_column
        text_col = self.config.text_column

        output: list[dict[str, Any]] = []
        for _, row in df_source.iterrows():
            rid = str(row[id_col])
            result = result_map.get(rid, {})
            output.append({
                id_col: rid,
                text_col: row[text_col],
                "keywords": result.get("keywords", ""),
                "sentiment_score": result.get("sentiment_score", -1),
                "sentiment_reason": result.get("sentiment_reason", ""),
                "complaint_type": result.get("complaint_type", ""),
            })
        return output

    def run(self) -> str:
        """Execute the full batch processing pipeline.

        Returns:
            Output CSV file path.
        """
        # Read input data
        input_path = self.config.data_input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)
        if df.empty:
            raise ValueError("Input file is empty")

        # Load checkpoint
        ckpt_path = self.config.data_checkpoint
        existing_results, processed_ids = self.load_checkpoint(ckpt_path)
        new_results: list[dict[str, Any]] = []

        # Filter already processed records
        df_filtered = df[
            ~df[self.config.id_column].astype(str).isin(processed_ids)
        ]

        if self.logger:
            self.logger.info(
                f"Total {len(df)} records, {len(processed_ids)} already processed, "
                f"{len(df_filtered)} pending"
            )

        if df_filtered.empty:
            if self.logger:
                self.logger.info("All records have been processed")
            all_results = existing_results
            output_rows = self._build_output_rows(all_results)
            self._write_output(self.config.data_output, output_rows)
            self._print_report()
            return self.config.data_output

        # Process in batches
        batch_size = self.config.batch_size
        total = len(df_filtered)

        with tqdm(total=total, desc="Processing", unit="records") as pbar:
            for start in range(0, total, batch_size):
                batch = df_filtered.iloc[start : start + batch_size]
                batch_results: list[dict[str, Any]] = []

                for _, row in batch.iterrows():
                    row_id = str(row[self.config.id_column])
                    text = str(row[self.config.text_column])
                    try:
                        labeled = self._label_row(row_id, text)
                        batch_results.append(labeled)
                        pbar.set_postfix_str(f"Latest: {text[:20]}...")
                    except Exception as exc:
                        if self.logger:
                            self.logger.error(
                                f"Failed to process ID={row_id}: {exc}"
                            )
                        batch_results.append({
                            self.config.id_column: row_id,
                            "keywords": "",
                            "sentiment_score": -1,
                            "sentiment_reason": f"Processing failed: {exc}",
                            "complaint_type": "",
                        })
                    pbar.update(1)

                # Save this batch's results
                new_results.extend(batch_results)
                all_ckpt = existing_results + new_results
                self.save_checkpoint(ckpt_path, all_ckpt)

        # Merge results with original CSV and write output
        all_results = existing_results + new_results
        output_rows = self._build_output_rows(all_results)
        output_path = self._write_output(
            self.config.data_output, output_rows
        )

        if self.logger:
            self.logger.info(
                f"Processing complete! Total {len(all_results)} records"
            )
            self.logger.info(f"Output file: {output_path}")

        # Generate summary report
        self._print_report(output_path)

        return output_path

    def _write_output(
        self, path: str, rows: list[dict[str, Any]]
    ) -> str:
        """Write result rows to CSV.

        Args:
            path: Output path.
            rows: Fully merged rows (id, text, keywords, score, reason, type).

        Returns:
            Output file path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            quoting=1,  # csv.QUOTE_ALL
        )
        return str(output_path)

    def _print_report(self, results_path: str | None = None) -> None:
        """Generate and log the summary report."""
        if results_path is None:
            results_path = self.config.data_output
        try:
            write_report(results_path)
            if self.logger:
                self.logger.info("Summary report: output/report.txt")
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Could not generate report: {exc}")
