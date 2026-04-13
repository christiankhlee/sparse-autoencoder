#!/usr/bin/env python3
"""
Build and save the factual-recall prompt families dataset.

Usage:
    python scripts/run_dataset.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root regardless of where the script is called.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_dataset import build_and_save_dataset
from src.utils.io import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the factual-recall prompt families dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    build_and_save_dataset(config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
