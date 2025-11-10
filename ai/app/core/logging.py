"""Logging configuration."""

import logging
import sys
from typing import Any

from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set third-party loggers to WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def mask_pii(text: str) -> str:
    """Mask potentially sensitive information in logs."""
    # Simple PII masking - can be enhanced
    import re

    # Mask SSN patterns
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "XXX-XX-XXXX", text)
    # Mask EIN patterns
    text = re.sub(r"\b\d{2}-\d{7}\b", "XX-XXXXXXX", text)
    return text


