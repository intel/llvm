"""Parser result models."""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

from ..models.test_results import TestStatus


class ParsedTestObservation(NamedTuple):
    name: str
    status: TestStatus
    duration_ms: Optional[float] = None


@dataclass
class ParsedLogData:
    tests: List[ParsedTestObservation] = field(default_factory=list)
    declared_counts: Dict[TestStatus, int] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)
    statistics_lines: List[str] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)
    slowest_tests: List[str] = field(default_factory=list)
    time_histogram: List[str] = field(default_factory=list)
    testing_time_ms: Optional[float] = None


@dataclass
class ParsedXMLData:
    tests: List[ParsedTestObservation] = field(default_factory=list)
    total_tests: Optional[int] = None
    total_time_seconds: Optional[float] = None


LIT_CATEGORY_TO_STATUS = {
    "Passed": TestStatus.PASS,
    "Passed With Retry": TestStatus.FLAKYPASS,
    "Passed After Update": TestStatus.FIXED,
    "Expectedly Failed": TestStatus.XFAIL,
    "Unexpectedly Passed": TestStatus.XPASS,
    "Failed": TestStatus.FAIL,
    "Unresolved": TestStatus.UNRESOLVED,
    "Unsupported": TestStatus.UNSUPPORTED,
    "Timed Out": TestStatus.TIMEOUT,
    "Skipped": TestStatus.SKIPPED,
    "Excluded": TestStatus.EXCLUDED,
}


LIT_STAT_TO_STATUS = {
    **LIT_CATEGORY_TO_STATUS,
    "Expected Passes": TestStatus.PASS,
}
