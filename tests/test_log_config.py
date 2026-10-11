from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import log_config as lc


@pytest.fixture
def isolated_logger() -> None:
    root = logging.getLogger(lc.LOGGER_NAME)
    old_handlers = root.handlers[:]
    old_level = root.level
    old_propagate = root.propagate
    yield
    lc._stop_listener()
    root.handlers.clear()
    root.propagate = old_propagate
    root.setLevel(old_level)
    for handler in old_handlers:
        root.addHandler(handler)


def test_configure_logging_creates_queue_listener_and_file(tmp_path: Path, isolated_logger: None) -> None:
    log_file = tmp_path / "swift_core_ai.log"

    lc.configure_logging(verbose=True, log_to_file=str(log_file))

    assert lc._listener is not None
    assert len(lc._listener.handlers) == 2
    assert any(isinstance(handler, logging.StreamHandler) for handler in lc._listener.handlers)
    assert any(isinstance(handler, logging.FileHandler) for handler in lc._listener.handlers)
    assert log_file.exists()

    root_logger = logging.getLogger(lc.LOGGER_NAME)
    assert root_logger.level == logging.DEBUG


def test_configure_logging_quiet_mode_sets_critical_level(isolated_logger: None) -> None:
    lc.configure_logging(verbose=False, log_to_file=None)

    root_logger = logging.getLogger(lc.LOGGER_NAME)
    assert root_logger.level == logging.CRITICAL
    assert lc._listener is not None
    assert root_logger.handlers and isinstance(root_logger.handlers[0], logging.handlers.QueueHandler)

    lc._stop_listener()
    root_logger.handlers.clear()
