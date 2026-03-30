"""
Logging configuration for NS-SE.

Uses loguru for structured logging with support for:
- Console output with colors
- File rotation
- JSON format for analysis
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure logging for the NS-SE framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files. If None, only console output.
        json_logs: If True, also write JSON-formatted logs for analysis.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
    )

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        logger.add(
            log_dir / "ns_gge_{time:YYYY-MM-DD}.log",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="7 days",
            compression="gz",
        )

        if json_logs:
            # JSON handler for structured analysis
            logger.add(
                log_dir / "ns_gge_{time:YYYY-MM-DD}.json",
                level=level,
                format="{message}",
                serialize=True,
                rotation="1 day",
                retention="7 days",
            )


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Logger instance bound with the given name
    """
    return logger.bind(name=name)


# Convenience: expose the main logger
__all__ = ["setup_logging", "get_logger", "logger"]
