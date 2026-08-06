"""Type definitions for test data structures."""
from typing import TypedDict, List


class TestLists(TypedDict, total=False):
    """Type definition for test list dictionary.
    
    Each key represents a test category, and the value is a list of test names
    belonging to that category.
    """

    Passed: List[str]
    Failed: List[str]
    Skipped: List[str]
    Unsupported: List[str]
    Excluded: List[str]
    Unresolved: List[str]


class TestCounts(TypedDict, total=False):
    """Type definition for test count dictionary.
    
    Each key represents a test category, and the value is the count of tests
    in that category.
    """

    Passed: int
    Failed: int
    Skipped: int
    Unsupported: int
    Excluded: int
    Unresolved: int


class TimingSummary(TypedDict):
    """Type definition for test timing summary.
    
    Contains information about test execution times.
    """

    slowest: List[str]
    histogram: List[str]


class SkippedTestsResult(TypedDict):
    """Result of skipped tests analysis.
    
    Attributes:
        tests: List of skipped test names.
        count: Total count of skipped tests.
        source: Source of the data ('log', 'xml', 'stats', or 'none').
        note: Human-readable explanation or warning message.
    """

    tests: List[str]
    count: int
    source: str
    note: str


class ExcludedTestsResult(TypedDict):
    """Result of excluded tests analysis.
    
    Attributes:
        tests: List of excluded test names.
        count: Total count of excluded tests.
        source: Source of the data ('log', 'xml', 'stats', or 'none').
        note: Human-readable explanation or warning message.
    """

    tests: List[str]
    count: int
    source: str
    note: str
