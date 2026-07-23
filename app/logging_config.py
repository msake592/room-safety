from __future__ import annotations

import logging
import os


_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    requested_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, requested_level, logging.INFO)

    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _CONFIGURED = True
