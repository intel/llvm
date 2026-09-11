"""GitHub Actions-specific output formatting."""

import os
import sys


def print_error(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)


def print_warning(message: str) -> None:
    print(f"::warning::{message}", file=sys.stderr)


def set_output(name: str, value: str) -> None:
    """Write name=value to the GITHUB_OUTPUT file for later steps."""
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        print(f"{name}={value}", file=f)
