"""Outputs package - Console and GitHub Actions output generation."""

from .console import ConsoleOutput
from .github_actions import GitHubActionsOutput

__all__ = [
    "ConsoleOutput",
    "GitHubActionsOutput",
]
