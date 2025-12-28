"""Shared logging configuration for all config modules."""

import sys

import logfire
from loguru import logger

# Track if logging has been configured
_logging_configured = False


def configure_logging() -> None:
    """Configure Logfire and loguru for console-only output (no API key needed)."""
    global _logging_configured
    if _logging_configured:
        return

    # Configure Logfire for console-only output (no API key needed)
    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()

    # Remove default loguru handler and add custom one
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )

    _logging_configured = True
