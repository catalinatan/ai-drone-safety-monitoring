"""Unit tests for src.logger."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.logger import setup_logger, get_logger


def test_setup_logger_console_only():
    logger = setup_logger("test_console", file_output=False)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_file_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger("test_file", log_dir=tmpdir, file_output=True, console_output=False)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        log_files = list(Path(tmpdir).glob("*.log"))
        assert len(log_files) == 1
        logger.handlers.clear()


def test_setup_logger_with_prefix():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger(
            "test_prefix", log_dir=tmpdir, log_prefix="myapp", console_output=False
        )
        log_files = list(Path(tmpdir).glob("*myapp.log"))
        assert len(log_files) == 1
        logger.handlers.clear()


def test_setup_logger_custom_format():
    logger = setup_logger("test_fmt", format_str="%(message)s", file_output=False)
    assert logger.handlers[0].formatter._fmt == "%(message)s"


def test_setup_logger_clears_old_handlers():
    logger = setup_logger("test_dup", file_output=False)
    n1 = len(logger.handlers)
    logger = setup_logger("test_dup", file_output=False)
    assert len(logger.handlers) == n1


def test_get_logger_creates_new():
    logger = get_logger("test_get_new", file_output=False)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) > 0


def test_get_logger_reuses_existing():
    name = "test_get_reuse"
    logger1 = get_logger(name, file_output=False)
    logger2 = get_logger(name, file_output=False)
    assert logger1 is logger2
