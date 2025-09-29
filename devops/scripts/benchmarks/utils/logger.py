# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys
from typing import Optional, Type, Any


# Define log level mapping as a module-level function
def _get_log_level(level_str: str) -> int:
    """Convert a string log level to a logging module level constant."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return level_map.get(level_str.lower(), logging.INFO)


class BenchmarkLogger:
    """Logger for the Benchmark Runner.

    This logger provides different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    that can be controlled via command-line arguments. Call initialize() with the
    appropriate parameters after parsing command line arguments.
    """

    _instance: Optional["BenchmarkLogger"] = None

    def __new__(cls: Type["BenchmarkLogger"]) -> "BenchmarkLogger":
        if cls._instance is None:
            cls._instance = super(BenchmarkLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Create logger but don't configure it until initialize() is called."""
        self._logger: logging.Logger = logging.getLogger("ur_benchmarks")

    def initialize(self, verbose: bool = False, log_level: str = "info") -> None:
        """Configure the logger with the appropriate log level.

        Args:
            verbose: If True, sets the log level to DEBUG regardless of log_level
            log_level: One of "debug", "info", "warning", "error", "critical"

        Note:
            This method will only initialize the logger once. Subsequent calls will be ignored.
        """
        # Return early if logger is already initialized (has handlers)
        if self._logger.handlers:
            return

        console_handler = logging.StreamHandler(sys.stdout)

        level = logging.DEBUG if verbose else _get_log_level(log_level)
        self._logger.setLevel(level)
        console_handler.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        self._logger.addHandler(console_handler)

    def debug(self, message: Any) -> None:
        """Log a debug message."""
        if self._logger.handlers:
            self._logger.debug(message)

    def info(self, message: Any) -> None:
        """Log an info message."""
        if self._logger.handlers:
            self._logger.info(message)

    def warning(self, message: Any) -> None:
        """Log a warning message."""
        if self._logger.handlers:
            self._logger.warning(message)

    def error(self, message: Any) -> None:
        """Log an error message."""
        if self._logger.handlers:
            self._logger.error(message)

    def critical(self, message: Any) -> None:
        """Log a critical message."""
        if self._logger.handlers:
            self._logger.critical(message)


# Global logger instance
log = BenchmarkLogger()
