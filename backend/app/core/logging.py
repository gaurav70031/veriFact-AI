"""
Structured logging configuration.
Call configure_logging() once during application startup.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """Set up root logger with a clean format."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Quieten noisy libraries
    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "filelock",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
