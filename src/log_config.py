from __future__ import annotations

import atexit
import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

LOGGER_NAME: str = "ai-func-kit"
logger: logging.Logger = logging.getLogger(LOGGER_NAME)

_listener: Optional[QueueListener] = None


def _stop_listener() -> None:
    """Stop the queued logging listener if it is active."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def configure_logging(
    verbose: bool = True,
    log_to_file: Optional[str] = "ai-func-kit.log",
) -> None:
    """Attach a queued console handler and optional file handler to the runtime logger."""
    global _listener

    _stop_listener()

    root_logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    root_logger.handlers.clear()
    root_logger.propagate = False

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
    console_handler.setLevel(logging.INFO if verbose else logging.CRITICAL)
    handlers.append(console_handler)

    if log_to_file is not None:
        try:
            file_handler: logging.FileHandler = logging.FileHandler(log_to_file, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            handlers.append(file_handler)
        except OSError as exc:
            logging.getLogger(LOGGER_NAME).error(
                "Failed to open log file '%s': %s",
                log_to_file,
                exc,
            )

    if verbose or log_to_file is not None:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.CRITICAL)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler: QueueHandler = QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    _listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    _listener.start()

    logger.info(
        "Logging configured (verbose=%s, log_to_file=%s)",
        verbose,
        log_to_file,
    )


atexit.register(_stop_listener)
