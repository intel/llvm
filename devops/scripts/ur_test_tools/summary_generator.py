"""Generate test summary reports."""

from typing import List

from .models.test_results import TestResult, TestStatus, TestRunResult


class SummaryReporter:
    """Generate a test summary."""

    def __init__(self, result: TestRunResult):
        self.result = result

    def generate(self) -> None:
        self._display_statistics(self.result)
        self._display_test_groups(self.result)
        self._display_timing(self.result)

    def _display_statistics(self, result: TestRunResult) -> None:
        if result.statistics_lines:
            print("=== Test Statistics ===")
            for line in result.statistics_lines:
                print(line)
            print()

    def _display_test_groups(self, result: TestRunResult) -> None:
        grouped = result.group_by_status()

        priority_order = [
            TestStatus.FAIL,
            TestStatus.TIMEOUT,
            TestStatus.UNRESOLVED,
            TestStatus.XPASS,
            TestStatus.SKIPPED,
            TestStatus.UNSUPPORTED,
            TestStatus.EXCLUDED,
            TestStatus.XFAIL,
            TestStatus.FIXED,
            TestStatus.FLAKYPASS,
            TestStatus.PASS,
        ]

        for status in priority_order:
            tests = grouped.get(status, [])
            if not tests:
                continue

            self._print_test_group(status, tests)

    def _print_test_group(self, status: TestStatus, tests: List[TestResult]) -> None:
        print(f"::group::{status.display_label} ({len(tests)})")

        for test in tests[:100]:
            print(test.name)

        if len(tests) > 100:
            print(f"... and {len(tests) - 100} more")

        print("::endgroup::")

    def _display_timing(self, result: TestRunResult) -> None:
        if not (
            result.testing_time_ms or result.slowest_tests or result.time_histogram
        ):
            return

        print("::group::Test Timing Summary")

        if result.testing_time_ms is not None:
            print(f"Testing Time: {result.testing_time_ms / 1000.0:.2f}s")
            print()

        if result.slowest_tests:
            print("Slowest Tests:")
            print("-" * 70)
            for line in result.slowest_tests:
                print(line)
            print()

        if result.time_histogram:
            print("Test Times Distribution:")
            print("-" * 70)
            for line in result.time_histogram:
                print(line)

        print("::endgroup::")
