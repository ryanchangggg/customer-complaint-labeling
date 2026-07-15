"""客服投诉标签系统——主入口"""

import argparse
import sys

from src.api_client import DeepSeekClient
from src.batch_processor import BatchProcessor
from src.config_loader import Config
from src.prompt_manager import PromptManager
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="客服投诉标签系统 - 基于 DeepSeek API 的文本标记工具"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config/config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 CSV 文件路径 (覆盖 config.yaml 中的配置)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 文件路径 (覆盖 config.yaml 中的配置)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小 (覆盖 config.yaml 中的配置)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (覆盖 config.yaml 中的配置)",
    )
    return parser.parse_args()


def main() -> None:
    """主函数。"""
    args = parse_args()

    # 加载配置
    config = Config(args.config)
    logger = setup_logger(
        "labeling",
        level=args.log_level or config.logging_level,
        log_file=config.logging_file,
        fmt=config.logging_format,
    )

    logger.info("=" * 50)
    logger.info("客服投诉标签系统启动")
    logger.info("=" * 50)

    # 命令行参数覆盖
    if args.input:
        config._raw["data"]["input"] = args.input
    if args.output:
        config._raw["data"]["output"] = args.output
    if args.batch_size:
        config._raw["batch"]["size"] = args.batch_size

    logger.info(f"输入文件: {config.data_input}")
    logger.info(f"输出文件: {config.data_output}")
    logger.info(f"批大小: {config.batch_size}")
    logger.info(f"API模型: {config.api_model}")

    # 初始化组件
    try:
        prompt_mgr = PromptManager()
        logger.info("Prompt 加载成功")
    except Exception as exc:
        logger.error(f"Prompt 加载失败: {exc}")
        sys.exit(1)

    try:
        api_client = DeepSeekClient(config)
        logger.info("API 客户端初始化成功")
    except ValueError as exc:
        logger.error(f"API 客户端初始化失败: {exc}")
        logger.error("请先在 .env 文件中设置 DEEPSEEK_API_KEY")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"API 客户端初始化失败: {exc}")
        sys.exit(1)

    # 执行批量处理
    processor = BatchProcessor(config, api_client, prompt_mgr, logger)
    try:
        output_path = processor.run()
        logger.info(f"处理完成，结果已保存至: {output_path}")
    except FileNotFoundError as exc:
        logger.error(f"输入文件不存在: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"处理过程发生异常: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
