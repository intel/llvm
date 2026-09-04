#!/usr/bin/env python3
"""
Copyright (C) 2025 Intel Corporation
Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Helper script for CI test validation - generates different test outcomes
"""

import sys
import time


def main():
    if len(sys.argv) < 2:
        print("Usage: test_helper.py <test_type>")
        sys.exit(1)

    test_type = sys.argv[1]

    if test_type == "pass":
        print("Test passed: result = 42")
        sys.exit(0)
    elif test_type == "fail":
        print("Test failed: expected 42, got 41")
        sys.exit(1)
    elif test_type == "timeout":
        print("Test starting infinite loop...")
        while True:
            time.sleep(1)
    elif test_type == "crash":
        print("Test about to crash...")
        sys.exit(137)  # SIGKILL
    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
