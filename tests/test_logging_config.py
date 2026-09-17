from __future__ import annotations

import logging

from services.logging_config import configure_logging


def test_invalid_log_level_falls_back_to_info(monkeypatch):
    root = logging.getLogger()
    original_level = root.level
    try:
        monkeypatch.setenv("LOG_LEVEL", "BOGUS")
        configure_logging()
        assert root.level == logging.INFO
    finally:
        root.setLevel(original_level)
