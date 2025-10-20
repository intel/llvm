# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys


log = logging.getLogger("ur_benchmarks")

def initialize_logger(verbose: bool = False, log_level: str = "info") -> None:
    """Configure the logger with the appropriate log level.

    Args:
        verbose: If True, sets the log level to DEBUG regardless of log_level
        log_level: One of "debug", "info", "warning", "error", "critical"

    Note:
        This method will only initialize the logger once. Subsequent calls will be ignored.
    """
    # Return early if logger is already initialized (has handlers)

    if log.handlers:
        return

    console_handler = logging.StreamHandler(sys.stdout)

    level = (
        logging.DEBUG
        if verbose
        else dict(
            debug=logging.DEBUG,
            info=logging.INFO,
            warning=logging.WARNING,
            error=logging.ERROR,
            critical=logging.CRITICAL,
        ).get(log_level.lower(), logging.INFO)
    )

    log.setLevel(level)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s <%(filename)s:%(lineno)d>"
    )
    console_handler.setFormatter(formatter)

    log.addHandler(console_handler)
