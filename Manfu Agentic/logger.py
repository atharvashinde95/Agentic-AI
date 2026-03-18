# core/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_FMT = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def _build(name: str, filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(_FMT)
    logger.addHandler(ch)

    # rotating file — 5 MB max, keep 3 backups
    fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, filename), maxBytes=5_242_880, backupCount=3
    )
    fh.setFormatter(_FMT)
    logger.addHandler(fh)
    return logger


agent_logger = _build("agent",  "agent.log")
alert_logger = _build("alerts", "alerts.log")
error_logger = _build("errors", "errors.log", logging.ERROR)
