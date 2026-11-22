from __future__ import annotations

import atexit
import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

LOGGER_NAME: str = "ai-func-kit"
logger: logging.Logger = logging.getLogger(LOGGER_NAME)

_listener: Optional[QueueListener] = None


def _normalize_log_level(
    verbose: Optional[bool] = None,
    log_level: Optional[str | int] = None,
) -> tuple[str, int]:
    """Normalize the user-facing log API into a level name and numeric logging level."""
    if log_level is not None:
        if isinstance(log_level, str):
            value = log_level.strip().lower()
            if value in {"silent", "none", "off", "disabled", "0"}:
                return "silent", logging.CRITICAL + 10
            if value in {"critical", "fatal", "error", "1"}:
                return "critical", logging.CRITICAL
            if value in {"warning", "warn", "2"}:
                return "warning", logging.WARNING
            if value in {"all", "debug", "verbose", "info", "3"}:
                return "all", logging.DEBUG
            raise ValueError(f"Unsupported log_level: {log_level!r}")
        if isinstance(log_level, int):
            if log_level <= 0:
                return "silent", logging.CRITICAL + 10
            if log_level == 1:
                return "critical", logging.CRITICAL
            if log_level == 2:
                return "warning", logging.WARNING
            return "all", logging.DEBUG
        raise TypeError("log_level must be a string or int")

    if verbose is None:
        return "all", logging.DEBUG
    return ("all", logging.DEBUG) if verbose else ("critical", logging.CRITICAL)


def _stop_listener() -> None:
    """Stop the queued logging listener if it is active."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def configure_logging(
    verbose: Optional[bool] = None,
    log_to_file: Optional[str] = "ai-func-kit.log",
    log_level: Optional[str | int] = None,
) -> None:
    """Attach a queued console handler and optional file handler to the runtime logger."""
    global _listener

    _stop_listener()

    root_logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    root_logger.handlers.clear()
    root_logger.propagate = False
    root_logger.disabled = False

    level_name, effective_level = _normalize_log_level(verbose=verbose, log_level=log_level)

    if level_name == "silent":
        root_logger.disabled = True
        root_logger.setLevel(effective_level)
        return

    console_formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter: logging.Formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(effective_level)
    handlers.append(console_handler)

    if log_to_file is not None:
        try:
            file_handler: logging.FileHandler = logging.FileHandler(log_to_file, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(effective_level)
            handlers.append(file_handler)
        except OSError as exc:
            logging.getLogger(LOGGER_NAME).error(
                "Failed to open log file '%s': %s",
                log_to_file,
                exc,
            )

    root_logger.setLevel(effective_level)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler: QueueHandler = QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    _listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    _listener.start()

    logger.info(
        "Logging configured (log_level=%s, log_to_file=%s)",
        level_name,
        log_to_file,
    )


atexit.register(_stop_listener)
