"""
Structlog-based JSON logging setup.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List

import structlog


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared: List[Any] = [
        structlog.contextvars.merge_contextvars,
        timestamper,
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
    ]

    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    structlog.configure(
        processors=shared + [structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
