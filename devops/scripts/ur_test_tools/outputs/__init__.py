"""Outputs package - Console and GitHub Actions output generation."""

from .console import filter_log_for_display
from .github_actions import GitHubActionsOutput

__all__ = [
    "filter_log_for_display",
    "GitHubActionsOutput",
]
