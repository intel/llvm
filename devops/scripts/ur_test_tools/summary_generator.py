"""Generate test summary reports."""

from typing import List

from .models.config import SummaryConfigFromLines
from .models.test_results import TestResult, TestStatus, TestRunResult
from .parsers.log_parser import LITLogParser
from .parsers.xml_parser import JUnitXMLParser
from .parsers.parser_models import ParsedLogData
from .reconciliation import reconcile_test_results


class SummaryReporter:
    """Generate a test summary."""

    def __init__(self, config: SummaryConfigFromLines):
        self.config = config

    def generate(self) -> None:
        log_parser = LITLogParser(self.config.log_lines)
        log_data = log_parser.parse_to_observations()

        xml_data = None
        if self.config.xml_file:
            xml_parser = JUnitXMLParser(self.config.xml_file)
            xml_data = xml_parser.parse_to_observations()

        result = reconcile_test_results(log_data, xml_data)

        self._display_statistics(log_parser)
        self._display_test_groups(result)
        self._display_timing(log_data)

    def _display_statistics(self, log_parser: LITLogParser) -> None:
        stats = log_parser.extract_statistics()
        if stats:
            print("=== Test Statistics ===")
            for stat in stats:
                print(stat.rstrip())
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

    def _display_timing(self, log_data: ParsedLogData) -> None:
        if not (log_data.slowest_tests or log_data.time_histogram):
            return

        print("::group::Test Timing Summary")

        if log_data.slowest_tests:
            print("Slowest Tests:")
            print("-" * 70)
            for line in log_data.slowest_tests:
                print(line)
            print()

        if log_data.time_histogram:
            print("Test Times Distribution:")
            print("-" * 70)
            for line in log_data.time_histogram:
                print(line)

        print("::endgroup::")
