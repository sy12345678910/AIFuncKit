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


def test_configure_logging_all_level_enables_debug_and_file(tmp_path: Path, isolated_logger: None) -> None:
    log_file = tmp_path / "swift_core_ai.log"

    lc.configure_logging(log_level="all", log_to_file=str(log_file))

    assert lc._listener is not None
    assert len(lc._listener.handlers) == 2
    assert any(isinstance(handler, logging.StreamHandler) for handler in lc._listener.handlers)
    assert any(isinstance(handler, logging.FileHandler) for handler in lc._listener.handlers)
    assert log_file.exists()

    root_logger = logging.getLogger(lc.LOGGER_NAME)
    assert root_logger.level == logging.DEBUG


def test_configure_logging_warning_level_filters_out_info(isolated_logger: None) -> None:
    lc.configure_logging(log_level="warning", log_to_file=None)

    root_logger = logging.getLogger(lc.LOGGER_NAME)
    assert root_logger.level == logging.WARNING
    assert lc._listener is not None
    assert root_logger.handlers and isinstance(root_logger.handlers[0], logging.handlers.QueueHandler)

    lc._stop_listener()
    root_logger.handlers.clear()


def test_configure_logging_silent_level_disables_logging(isolated_logger: None) -> None:
    lc.configure_logging(log_level="silent", log_to_file=None)

    root_logger = logging.getLogger(lc.LOGGER_NAME)
    assert root_logger.disabled is True
    assert lc._listener is None or not lc._listener.handlers
