"""Generate test summary reports."""

from typing import List

from .models.test_results import TestResult, TestStatus, TestRunResult

# Failure-first order used when listing test groups.
_STATUS_DISPLAY_ORDER = [
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

# Statuses worth listing by name in the compact GitHub step summary.
_STEP_SUMMARY_LISTED_STATUSES = [
    TestStatus.FAIL,
    TestStatus.XPASS,
    TestStatus.TIMEOUT,
    TestStatus.UNRESOLVED,
]

_STEP_SUMMARY_MAX_LISTED_TESTS = 50


class SummaryReporter:
    def __init__(self, result: TestRunResult):
        self.result = result

    def generate(self) -> None:
        self._display_statistics()
        self._display_test_groups()
        self._display_timing()

    def _display_statistics(self) -> None:
        if self.result.statistics_lines:
            print("=== Test Statistics ===")
            for line in self.result.statistics_lines:
                print(line)
            print()

    def _display_test_groups(self) -> None:
        grouped = self.result.group_by_status()

        for status in _STATUS_DISPLAY_ORDER:
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

    def _display_timing(self) -> None:
        result = self.result
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

    def generate_github_step_summary(self) -> None:
        """Print a compact summary for $GITHUB_STEP_SUMMARY.

        Unlike generate(), this lists only statuses that need attention
        (Failed, Unexpectedly Passed, Timed Out, Unresolved) and shows
        all other statuses as counts only, to keep the workflow run page
        readable regardless of how many tests were discovered.
        """
        result = self.result
        grouped = result.group_by_status()

        print("```")

        for status in _STEP_SUMMARY_LISTED_STATUSES:
            tests = grouped.get(status, [])
            if not tests:
                continue

            print(f"{status.display_label} Tests ({len(tests)}):")
            for test in tests[:_STEP_SUMMARY_MAX_LISTED_TESTS]:
                print(test.name)
            if len(tests) > _STEP_SUMMARY_MAX_LISTED_TESTS:
                remaining = len(tests) - _STEP_SUMMARY_MAX_LISTED_TESTS
                print(f"... and {remaining} more")
            print()

        if result.testing_time_ms is not None:
            print(f"Testing Time: {result.testing_time_ms / 1000.0:.2f}s")
            print()

        total = result.total_discovered or len(result.tests)
        counted_statuses = [s for s in _STATUS_DISPLAY_ORDER if grouped.get(s)]
        if total and counted_statuses:
            print(f"Total Discovered Tests: {total}")
            label_width = max(len(s.display_label) for s in counted_statuses)
            for status in counted_statuses:
                count = len(grouped[status])
                percentage = count / total * 100
                print(
                    f"{status.display_label:<{label_width}} : "
                    f"{count} ({percentage:.2f}%)"
                )

        print("```")
