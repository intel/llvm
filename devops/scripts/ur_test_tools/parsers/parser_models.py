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
    error_details: List[str] = field(default_factory=list)
    slowest_tests: List[str] = field(default_factory=list)
    time_histogram: List[str] = field(default_factory=list)


@dataclass
class ParsedXMLData:
    tests: List[ParsedTestObservation] = field(default_factory=list)
    total_tests: Optional[int] = None
    total_time_seconds: Optional[float] = None


LIT_OUTPUT_TO_STATUS = {
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
    "Expected Passes": TestStatus.PASS,
}


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
