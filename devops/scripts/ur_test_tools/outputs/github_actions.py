"""GitHub Actions-specific output formatting."""

import sys


class GitHubActionsOutput:
    """Format output for GitHub Actions."""

    @staticmethod
    def print_error(message: str) -> None:
        print(f"::error::{message}", file=sys.stderr)

    @staticmethod
    def print_warning(message: str) -> None:
        print(f"::warning::{message}", file=sys.stderr)

    @staticmethod
    def set_output(name: str, value: str) -> None:
        print(f"{name}={value}", flush=True)
        sys.stdout.flush()
