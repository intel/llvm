"""GitHub Actions-specific output formatting."""
import sys


class GitHubActionsOutput:
    """Format output for GitHub Actions."""

    @staticmethod
    def print_error(message: str) -> None:
        """Print ::error:: annotation visible in GitHub Actions UI.

        Args:
            message: Error message to display.
        """
        print(f"::error::{message}", file=sys.stderr)

    @staticmethod
    def print_warning(message: str) -> None:
        """Print ::warning:: annotation visible in GitHub Actions UI.

        Args:
            message: Warning message to display.
        """
        print(f"::warning::{message}", file=sys.stderr)

    @staticmethod
    def set_output(name: str, value: str) -> None:
        """Set GitHub Actions output variable.

        Args:
            name: Output variable name (kebab-case).
            value: Output value.
        """
        print(f"{name}={value}", flush=True)
        sys.stdout.flush()
