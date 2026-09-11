"""Structured logging + redaction. Human-readable lines + JSON event payload."""
from __future__ import annotations

import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

SECRET_KEYS = ("API_KEY", "API_SECRET", "SECRET", "TOKEN", "PASSWORD")


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: "***" if any(s in str(k).upper() for s in SECRET_KEYS) else redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def setup_logging(level: str = "INFO", log_file: str = "logs/trading.log") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger("trading-goat")


def log_event(logger: logging.Logger, level: int, event: str, message: str, **fields: Any) -> None:
    payload = {"event": event, "message": message, **redact(fields)}
    logger.log(level, "%s | %s", event, json.dumps(payload, default=str))
