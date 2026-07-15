"""批量处理模块——支持断点续跑、进度条"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.api_client import DeepSeekClient
from src.config_loader import Config
from src.prompt_manager import PromptManager


class BatchProcessor:
    """批量处理控制器，管理 CSV 读取、分批调用、断点续跑和结果输出。"""

    def __init__(
        self,
        config: Config,
        api_client: DeepSeekClient,
        prompt_manager: PromptManager,
        logger: Any = None,
    ) -> None:
        """初始化。

        Args:
            config: 全局配置。
            api_client: DeepSeek API 客户端。
            prompt_manager: Prompt 管理器。
            logger: Logger 实例。
        """
        self.config = config
        self.client = api_client
        self.prompt_mgr = prompt_manager
        self.logger = logger

    def load_checkpoint(self, path: str) -> tuple[list[dict[str, Any]], set[str]]:
        """加载检查点，返回已有结果和已处理 ID 集合。

        Args:
            path: 检查点文件路径。

        Returns:
            (已有结果列表, 已处理 ID 集合)。
        """
        if not os.path.exists(path):
            return [], set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            processed_ids = {str(r[self.config.id_column]) for r in data}
            if self.logger:
                self.logger.info(f"找到检查点，已有 {len(processed_ids)} 条结果")
            return data, processed_ids
        except (json.JSONDecodeError, KeyError) as exc:
            if self.logger:
                self.logger.warning(f"检查点文件损坏，将重新开始: {exc}")
            return [], set()

    def save_checkpoint(
        self, path: str, results: list[dict[str, Any]]
    ) -> None:
        """保存检查点。

        Args:
            path: 检查点文件路径。
            results: 结果列表。
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _label_row(
        self, row_id: str, text: str
    ) -> dict[str, Any]:
        """单行文本分析。

        Args:
            row_id: 记录 ID。
            text: 文本内容。

        Returns:
            标签化结果。
        """
        prompt = self.prompt_mgr.render(text)
        result = self.client.analyze(prompt, self.logger)

        return {
            self.config.id_column: row_id,
            self.config.text_column: text,
            "keywords": ";".join(result.get("keywords", [])),
            "sentiment_score": result.get("sentiment_score", -1),
            "sentiment_reason": result.get("reason", ""),
        }

    def run(self) -> str:
        """执行完整批量处理流程。

        Returns:
            输出 CSV 文件路径。
        """
        # 读取输入数据
        input_path = self.config.data_input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        df = pd.read_csv(input_path)
        if df.empty:
            raise ValueError("输入文件为空")

        # 加载检查点
        ckpt_path = self.config.data_checkpoint
        existing_results, processed_ids = self.load_checkpoint(ckpt_path)
        new_results: list[dict[str, Any]] = []

        # 过滤已处理的记录
        df_filtered = df[~df[self.config.id_column].astype(str).isin(processed_ids)]

        if self.logger:
            self.logger.info(
                f"总计 {len(df)} 条，已处理 {len(processed_ids)} 条，"
                f"待处理 {len(df_filtered)} 条"
            )

        if df_filtered.empty:
            if self.logger:
                self.logger.info("所有记录已处理完毕")
            self._write_output(
                self.config.data_output, existing_results
            )
            return self.config.data_output

        # 分批处理
        batch_size = self.config.batch_size
        total = len(df_filtered)

        with tqdm(total=total, desc="处理进度", unit="条") as pbar:
            for start in range(0, total, batch_size):
                batch = df_filtered.iloc[start : start + batch_size]
                batch_results: list[dict[str, Any]] = []

                for _, row in batch.iterrows():
                    row_id = str(row[self.config.id_column])
                    text = str(row[self.config.text_column])
                    try:
                        labeled = self._label_row(row_id, text)
                        batch_results.append(labeled)
                        pbar.set_postfix_str(
                            f"最新: {text[:20]}..."
                        )
                    except Exception as exc:
                        if self.logger:
                            self.logger.error(
                                f"处理 ID={row_id} 失败: {exc}"
                            )
                        batch_results.append({
                            self.config.id_column: row_id,
                            self.config.text_column: text,
                            "keywords": "",
                            "sentiment_score": -1,
                            "sentiment_reason": f"处理失败: {exc}",
                        })
                    pbar.update(1)

                # 保存该批次结果
                new_results.extend(batch_results)
                all_results = existing_results + new_results
                self.save_checkpoint(ckpt_path, all_results)

        # 合并结果并输出
        all_results = existing_results + new_results
        output_path = self._write_output(
            self.config.data_output, all_results
        )

        if self.logger:
            self.logger.info(
                f"处理完成！共 {len(all_results)} 条结果"
            )
            self.logger.info(f"输出文件: {output_path}")

        return output_path

    def _write_output(
        self, path: str, results: list[dict[str, Any]]
    ) -> str:
        """将结果写入 CSV。

        Args:
            path: 输出路径。
            results: 结果列表。

        Returns:
            输出文件路径。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_out = pd.DataFrame(results)
        df_out.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            quoting=1,  # csv.QUOTE_ALL
        )
        return str(output_path)
