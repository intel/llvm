"""Test result models."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TestStatus(str, Enum):
    """LIT test status."""

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
        return self in (
            TestStatus.FAIL,
            TestStatus.XPASS,
            TestStatus.UNRESOLVED,
            TestStatus.TIMEOUT,
        )

    @property
    def display_label(self) -> str:
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
    """Result for a single test."""

    name: str
    status: TestStatus
    duration_ms: Optional[float] = None


@dataclass
class TestRunResult:
    """Results for a test run."""

    tests: List[TestResult]
    total_discovered: Optional[int] = None
    testing_time_ms: Optional[float] = None

    def count_by_status(self, status: TestStatus) -> int:
        return sum(1 for test in self.tests if test.status == status)

    def tests_by_status(self, status: TestStatus) -> List[TestResult]:
        return [test for test in self.tests if test.status == status]

    def group_by_status(self) -> Dict[TestStatus, List[TestResult]]:
        groups: Dict[TestStatus, List[TestResult]] = {}
        for test in self.tests:
            groups.setdefault(test.status, []).append(test)
        return groups
