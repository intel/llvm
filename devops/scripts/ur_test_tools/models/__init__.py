"""Models package - Data structures for UR test tools."""

from .test_data import (
    TestLists,
    TestCounts,
    TimingSummary,
    SkippedTestsResult,
    ExcludedTestsResult,
)
from .config import (
    TestConfig,
    TestExecutionContext,
    SummaryConfigFromLines,
)
from .test_results import (
    TestStatus,
    TestResult,
    TestRunResult,
)

__all__ = [
    # Legacy models (to be deprecated)
    "TestLists",
    "TestCounts",
    "TimingSummary",
    "SkippedTestsResult",
    "ExcludedTestsResult",
    # Configuration
    "TestConfig",
    "TestExecutionContext",
    "SummaryConfigFromLines",
    # Normalized result model
    "TestStatus",
    "TestResult",
    "TestRunResult",
]
