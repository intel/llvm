#!/usr/bin/env python3
"""Unified Runtime test summary processing for GitHub Actions CI."""

import sys
from ur_test_tools.cli import main_test_summary

if __name__ == "__main__":
    sys.exit(main_test_summary())
