"""Parser-specific data models - intermediate observations before normalization."""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

from ..models.test_results import TestStatus


class ParsedTestObservation(NamedTuple):
    """A single test observation from a parser (log or XML)."""

    name: str
    status: TestStatus
    duration_ms: Optional[float] = None


@dataclass
class ParsedLogData:
    """Raw observations and statistics from LIT text output."""

    tests: List[ParsedTestObservation] = field(default_factory=list)
    # Declared counts from "Category Tests (N):" headers
    declared_counts: Dict[TestStatus, int] = field(default_factory=dict)
    # Statistics from "Stat Name: N" lines
    statistics: Dict[str, int] = field(default_factory=dict)
    # Raw error/failure details
    error_details: List[str] = field(default_factory=list)
    # Timing summary sections
    slowest_tests: List[str] = field(default_factory=list)
    time_histogram: List[str] = field(default_factory=list)


@dataclass
class ParsedXMLData:
    """Observations from JUnit XML output."""

    tests: List[ParsedTestObservation] = field(default_factory=list)
    # XML may have additional metadata we can extract in the future
    total_tests: Optional[int] = None
    total_time_seconds: Optional[float] = None


# Mapping from LIT output labels to canonical status
LIT_OUTPUT_TO_STATUS = {
    # Category headers
    "Passed": TestStatus.PASS,
    "Passed With Retry": TestStatus.FLAKYPASS,
    "Expectedly Failed": TestStatus.XFAIL,
    "Unexpectedly Passed": TestStatus.XPASS,
    "Failed": TestStatus.FAIL,
    "Unresolved": TestStatus.UNRESOLVED,
    "Unsupported": TestStatus.UNSUPPORTED,
    "Timed Out": TestStatus.TIMEOUT,
    "Skipped": TestStatus.SKIPPED,
    "Excluded": TestStatus.EXCLUDED,
    # Statistics labels (may differ slightly)
    "Expected Passes": TestStatus.PASS,
}


# Reverse mapping for display
STATUS_TO_LIT_LABEL = {
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
}
