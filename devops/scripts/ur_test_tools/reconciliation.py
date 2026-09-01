"""Combine parsed test results."""

import sys
from typing import List, Optional

from .models.config import SummaryConfigFromLines
from .models.test_results import TestResult, TestRunResult, TestStatus
from .parsers.log_parser import LITLogParser
from .parsers.parser_models import LIT_STAT_TO_STATUS, ParsedLogData, ParsedXMLData
from .parsers.xml_parser import JUnitXMLParser

XML_FALLBACK_STATUSES = {
    TestStatus.EXCLUDED,
    TestStatus.SKIPPED,
    TestStatus.UNSUPPORTED,
}


def build_test_run_result(config: SummaryConfigFromLines) -> TestRunResult:
    log_data = LITLogParser(config.log_lines).parse_to_observations()
    xml_data = None
    if config.xml_file:
        xml_data = JUnitXMLParser(config.xml_file).parse_to_observations()
    return reconcile_test_results(log_data, xml_data)


def reconcile_test_results(
    log_data: ParsedLogData, xml_data: Optional[ParsedXMLData] = None
) -> TestRunResult:
    """Prefer complete log lists and fill missing results from XML."""
    log_by_name = {obs.name: obs for obs in log_data.tests}
    log_counts = {}
    for observation in log_data.tests:
        log_counts[observation.status] = log_counts.get(observation.status, 0) + 1

    complete_log_statuses = {
        status
        for status, declared_count in log_data.declared_counts.items()
        if log_counts.get(status, 0) == declared_count
    }

    xml_by_name = {}
    if xml_data:
        xml_by_name = {obs.name: obs for obs in xml_data.tests}

    results = []
    for log_obs in log_data.tests:
        xml_obs = xml_by_name.get(log_obs.name)
        duration_ms = xml_obs.duration_ms if xml_obs else log_obs.duration_ms

        results.append(
            TestResult(
                name=log_obs.name,
                status=log_obs.status,
                duration_ms=duration_ms,
            )
        )

    result_names = set(log_by_name)
    if xml_data:
        for xml_obs in xml_data.tests:
            if (
                xml_obs.status not in XML_FALLBACK_STATUSES
                or xml_obs.status in complete_log_statuses
                or xml_obs.name in result_names
            ):
                continue

            results.append(
                TestResult(
                    name=xml_obs.name,
                    status=xml_obs.status,
                    duration_ms=xml_obs.duration_ms,
                )
            )
            result_names.add(xml_obs.name)

    total_discovered = log_data.statistics.get("Total Discovered")
    testing_time_ms = log_data.testing_time_ms
    if testing_time_ms is None and xml_data and xml_data.total_time_seconds:
        testing_time_ms = xml_data.total_time_seconds * 1000.0

    _validate_counts(log_data, results)

    return TestRunResult(
        tests=results,
        total_discovered=total_discovered,
        testing_time_ms=testing_time_ms,
        statistics_lines=log_data.statistics_lines,
        slowest_tests=log_data.slowest_tests,
        time_histogram=log_data.time_histogram,
    )


def _validate_counts(log_data: ParsedLogData, results: List[TestResult]) -> None:
    actual_counts = {}
    for result in results:
        actual_counts[result.status] = actual_counts.get(result.status, 0) + 1

    for status, declared_count in log_data.declared_counts.items():
        actual_count = actual_counts.get(status, 0)
        if actual_count != declared_count:
            print(
                f"Warning: Count mismatch for {status.value}: "
                f"declared {declared_count}, found {actual_count}",
                file=sys.stderr,
            )

    for label, expected_count in log_data.statistics.items():
        status = LIT_STAT_TO_STATUS.get(label)
        if status is None:
            if label != "Total Discovered":
                print(f"Warning: Unknown test statistic '{label}'", file=sys.stderr)
            continue
        actual_count = actual_counts.get(status, 0)
        if actual_count != expected_count:
            print(
                f"Warning: Count mismatch for {status.value}: "
                f"statistics {expected_count}, found {actual_count}",
                file=sys.stderr,
            )

    if log_data.statistics.get("Total Discovered"):
        total_discovered = log_data.statistics["Total Discovered"]
        total_results = len(results)
        if total_results != total_discovered:
            print(
                f"Warning: Total test mismatch: "
                f"declared {total_discovered}, found {total_results}",
                file=sys.stderr,
            )
