"""UR Test Tools - Modular test orchestration and summary generation."""

from .models import (
    # Configuration
    TestConfig,
    SummaryConfigFromLines,
    # Normalized result model
    TestStatus,
    TestResult,
    TestRunResult,
    # Legacy (to be removed)
    TimingSummary,
)
from .test_runner import TestRunner
from .summary_generator import SummaryReporter
from .validation import (
    PathValidator,
)
from .outputs import (
    ConsoleOutput,
    GitHubActionsOutput,
)

__all__ = [
    "__version__",
    "__author__",
    "TestConfig",
    "SummaryConfigFromLines",
    "TestStatus",
    "TestResult",
    "TestRunResult",
    "TimingSummary",  # Legacy
    "TestRunner",
    "SummaryReporter",
    "PathValidator",
    "ConsoleOutput",
    "GitHubActionsOutput",
]
