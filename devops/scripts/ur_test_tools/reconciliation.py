"""Reconciliation layer - combines parser observations into canonical results."""

import sys
from typing import List, Optional

from .models.test_results import TestResult, TestRunResult, TestStatus
from .parsers.parser_models import ParsedLogData, ParsedXMLData


def reconcile_test_results(
    log_data: ParsedLogData, xml_data: Optional[ParsedXMLData] = None
) -> TestRunResult:
    """Reconcile observations from log and XML into canonical test results.

    Reconciliation policy:
    1. Test status comes from log output (authoritative for XFAIL/XPASS/FLAKYPASS)
    2. Test duration comes from XML (more precise than log timing sections)
    3. If test appears in XML but not log, include with XML status
    4. Validate counts and warn on mismatches

    Args:
        log_data: Parsed observations from LIT text output
        xml_data: Optional parsed observations from JUnit XML

    Returns:
        TestRunResult with reconciled test results
    """
    # Build index of log results (status is authoritative)
    log_by_name = {obs.name: obs for obs in log_data.tests}

    # Build index of XML results (timing is authoritative)
    xml_by_name = {}
    if xml_data:
        xml_by_name = {obs.name: obs for obs in xml_data.tests}

    # Reconcile: start with all tests from log
    results = []
    for log_obs in log_data.tests:
        # Use log status, but prefer XML timing if available
        xml_obs = xml_by_name.get(log_obs.name)
        duration_ms = xml_obs.duration_ms if xml_obs else log_obs.duration_ms

        results.append(
            TestResult(
                name=log_obs.name,
                status=log_obs.status,
                duration_ms=duration_ms,
            )
        )

    # Add any tests that appear in XML but not in log
    # This shouldn't happen normally, but handle it gracefully
    xml_only_tests = set(xml_by_name.keys()) - set(log_by_name.keys())
    if xml_only_tests:
        print(
            f"Warning: {len(xml_only_tests)} test(s) found in XML but not in log",
            file=sys.stderr,
        )
        for test_name in sorted(xml_only_tests):
            xml_obs = xml_by_name[test_name]
            results.append(
                TestResult(
                    name=xml_obs.name,
                    status=xml_obs.status,
                    duration_ms=xml_obs.duration_ms,
                )
            )

    # Extract total discovered and testing time
    total_discovered = log_data.statistics.get("Total Discovered")
    testing_time_ms = None
    if xml_data and xml_data.total_time_seconds:
        testing_time_ms = xml_data.total_time_seconds * 1000.0

    # Validate counts
    _validate_counts(log_data, results)

    return TestRunResult(
        tests=results,
        total_discovered=total_discovered,
        testing_time_ms=testing_time_ms,
    )


def _validate_counts(log_data: ParsedLogData, results: List[TestResult]) -> None:
    """Validate that reconciled results match declared counts from log.

    Warns if counts don't match but doesn't fail - this helps catch parser bugs.
    """
    # Count results by status
    actual_counts = {}
    for result in results:
        actual_counts[result.status] = actual_counts.get(result.status, 0) + 1

    # Compare with declared counts from log
    for status, declared_count in log_data.declared_counts.items():
        actual_count = actual_counts.get(status, 0)
        if actual_count != declared_count:
            print(
                f"Warning: Count mismatch for {status.value}: "
                f"declared {declared_count}, found {actual_count}",
                file=sys.stderr,
            )

    # Check total discovered if available
    if log_data.statistics.get("Total Discovered"):
        total_discovered = log_data.statistics["Total Discovered"]
        total_results = len(results)
        if total_results != total_discovered:
            print(
                f"Warning: Total test mismatch: "
                f"declared {total_discovered}, found {total_results}",
                file=sys.stderr,
            )
