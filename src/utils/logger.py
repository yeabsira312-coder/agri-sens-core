"""Logging Module for AGRI-SENS-CORE.

Provides colorized console output and daily rotating file logging configured for the system.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
from typing import Optional

# ANSI Color Codes for Console Output
COLOR_CODES = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;31m",  # Bold Red
    "RESET": "\033[0m",  # Reset
}


class ColoredConsoleFormatter(logging.Formatter):
    """Custom formatter providing ANSI color-coded console logs."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        """Initialize ColoredConsoleFormatter."""
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with level-specific color codes."""
        levelname = record.levelname
        color = COLOR_CODES.get(levelname, COLOR_CODES["RESET"])
        reset = COLOR_CODES["RESET"]

        # Temporarily colorize levelname for output
        record.levelname = f"{color}{levelname}{reset}"
        formatted = super().format(record)
        record.levelname = levelname  # Restore original
        return formatted


def setup_logger(
    name: str = "agri_sens",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger instance with console and file handlers.

    Args:
        name: Name of the logger instance.
        log_dir: Directory path to store log files. Defaults to 'logs' directory at project root.
        log_level: Base logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Resolve log directory
    if log_dir is None:
        # Base path relative to agri_sens_core root
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_log_dir = project_root / "logs"
    else:
        resolved_log_dir = Path(log_dir)

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = resolved_log_dir / "agri_sens.log"

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 1. Console Handler (Colorized)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredConsoleFormatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(console_handler)

    # 2. Daily Rotating File Handler
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file_path),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    return logger


# Default system logger instance
default_logger = setup_logger()


def get_logger(name: str = "agri_sens") -> logging.Logger:
    """Retrieve logger instance by name.

    Args:
        name: Name of the logger.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


def info(msg: str, *args: object, **kwargs: object) -> None:
    """Log an INFO level message using the default logger."""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args: object, **kwargs: object) -> None:
    """Log a WARNING level message using the default logger."""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args: object, **kwargs: object) -> None:
    """Log an ERROR level message using the default logger."""
    default_logger.error(msg, *args, **kwargs)


def debug(msg: str, *args: object, **kwargs: object) -> None:
    """Log a DEBUG level message using the default logger."""
    default_logger.debug(msg, *args, **kwargs)


def critical(msg: str, *args: object, **kwargs: object) -> None:
    """Log a CRITICAL level message using the default logger."""
    default_logger.critical(msg, *args, **kwargs)
