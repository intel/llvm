"""Canonical test result model - independent of data sources.

This module defines the normalized representation of test results that is used
throughout the system. Parsers translate source-specific formats (LIT log,
JUnit XML) into this model, and consumers (summary, validation, future DB)
operate only on these normalized types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TestStatus(str, Enum):
    """Canonical test result status.

    Maps to LIT result codes. The string values match LIT's internal codes,
    not the human-readable output labels.

    LIT Output Label -> TestStatus:
        "Passed"              -> PASS
        "Passed With Retry"   -> FLAKYPASS
        "Expectedly Failed"   -> XFAIL
        "Unexpectedly Passed" -> XPASS
        "Failed"              -> FAIL
        "Unresolved"          -> UNRESOLVED
        "Unsupported"         -> UNSUPPORTED
        "Timed Out"           -> TIMEOUT
        "Skipped"             -> SKIPPED
        "Excluded"            -> EXCLUDED
    """

    PASS = "PASS"
    FLAKYPASS = "FLAKYPASS"
    XFAIL = "XFAIL"
    XPASS = "XPASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"
    UNSUPPORTED = "UNSUPPORTED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    EXCLUDED = "EXCLUDED"

    @property
    def is_failure(self) -> bool:
        """Check if this status represents a test failure.

        LIT treats FAIL, XPASS, UNRESOLVED, and TIMEOUT as failures.
        """
        return self in (
            TestStatus.FAIL,
            TestStatus.XPASS,
            TestStatus.UNRESOLVED,
            TestStatus.TIMEOUT,
        )

    @property
    def display_label(self) -> str:
        """Get human-readable label for display."""
        return {
            TestStatus.PASS: "Passed",
            TestStatus.FLAKYPASS: "Passed With Retry",
            TestStatus.XFAIL: "Expectedly Failed",
            TestStatus.XPASS: "Unexpectedly Passed",
            TestStatus.FAIL: "Failed",
            TestStatus.UNRESOLVED: "Unresolved",
            TestStatus.UNSUPPORTED: "Unsupported",
            TestStatus.TIMEOUT: "Timed Out",
            TestStatus.SKIPPED: "Skipped",
            TestStatus.EXCLUDED: "Excluded",
        }[self]


@dataclass
class TestResult:
    """Normalized result for a single test.

    Represents test outcome independent of where the information was obtained.
    """

    name: str
    status: TestStatus
    duration_ms: Optional[float] = None


@dataclass
class TestRunResult:
    """Complete normalized result of a test run.

    Contains all test results and run-level metadata.
    """

    tests: List[TestResult]
    total_discovered: Optional[int] = None
    testing_time_ms: Optional[float] = None

    def count_by_status(self, status: TestStatus) -> int:
        """Count tests with given status."""
        return sum(1 for test in self.tests if test.status == status)

    def tests_by_status(self, status: TestStatus) -> List[TestResult]:
        """Get all tests with given status."""
        return [test for test in self.tests if test.status == status]

    def group_by_status(self) -> Dict[TestStatus, List[TestResult]]:
        """Group all tests by their status."""
        groups: Dict[TestStatus, List[TestResult]] = {}
        for test in self.tests:
            groups.setdefault(test.status, []).append(test)
        return groups
