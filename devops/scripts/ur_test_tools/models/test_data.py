"""Type definitions for test data structures."""

from typing import TypedDict, List


class TestLists(TypedDict, total=False):
    """Type definition for test list dictionary."""

    Passed: List[str]
    Failed: List[str]
    Skipped: List[str]
    Unsupported: List[str]
    Excluded: List[str]
    Unresolved: List[str]


class TestCounts(TypedDict, total=False):
    """Type definition for test count dictionary."""

    Passed: int
    Failed: int
    Skipped: int
    Unsupported: int
    Excluded: int
    Unresolved: int


class TimingSummary(TypedDict):
    """Type definition for test timing summary."""

    slowest: List[str]
    histogram: List[str]


class SkippedTestsResult(TypedDict):
    """Result of skipped tests analysis."""

    tests: List[str]
    count: int
    source: str
    note: str


class ExcludedTestsResult(TypedDict):
    """Result of excluded tests analysis."""

    tests: List[str]
    count: int
    source: str
    note: str
