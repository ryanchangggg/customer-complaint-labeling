"""Customer Complaint Labeling System - Main Entry Point"""

import argparse
import sys

from src.api_client import DeepSeekClient
from src.batch_processor import BatchProcessor
from src.config_loader import Config
from src.prompt_manager import PromptManager
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Customer Complaint Labeling System - DeepSeek API based text labeling tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file path (default: config/config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file path (overrides config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (overrides config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load configuration
    config = Config(args.config)
    logger = setup_logger(
        "labeling",
        level=args.log_level or config.logging_level,
        log_file=config.logging_file,
        fmt=config.logging_format,
    )

    logger.info("=" * 50)
    logger.info("Customer Complaint Labeling System Started")
    logger.info("=" * 50)

    # Command line argument overrides
    if args.input:
        config._raw["data"]["input"] = args.input
    if args.output:
        config._raw["data"]["output"] = args.output
    if args.batch_size:
        config._raw["batch"]["size"] = args.batch_size

    logger.info(f"Input file: {config.data_input}")
    logger.info(f"Output file: {config.data_output}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"API model: {config.api_model}")

    # Initialize components
    try:
        prompt_mgr = PromptManager()
        logger.info("Prompt loaded successfully")
    except Exception as exc:
        logger.error(f"Prompt loading failed: {exc}")
        sys.exit(1)

    try:
        api_client = DeepSeekClient(config)
        logger.info("API client initialized successfully")
    except ValueError as exc:
        logger.error(f"API client initialization failed: {exc}")
        logger.error("Please set DEEPSEEK_API_KEY in the .env file")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"API client initialization failed: {exc}")
        sys.exit(1)

    # Execute batch processing
    processor = BatchProcessor(config, api_client, prompt_mgr, logger)
    try:
        output_path = processor.run()
        logger.info(f"Processing complete. Results saved to: {output_path}")
    except FileNotFoundError as exc:
        logger.error(f"Input file not found: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Processing failed with exception: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
