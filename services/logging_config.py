from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging() -> None:
    """
    Configure application logging once.

    Respects LOG_LEVEL (default INFO). If the root logger already has
    handlers (e.g. Chainlit configured it), only the level is adjusted;
    otherwise a plain stream handler is installed. stdout is switched to
    UTF-8 on a best-effort basis so a legacy Windows console cannot raise
    UnicodeEncodeError while emitting log lines.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        logging.basicConfig(level=level, format=_FORMAT)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass
