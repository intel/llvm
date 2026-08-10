"""Generate test summary reports."""

import sys
from typing import List

from .models.config import SummaryConfigFromLines
from .models.test_data import (
    TestLists,
    TestCounts,
    SkippedTestsResult,
    ExcludedTestsResult,
)
from .parsers.log_parser import LITLogParser
from .parsers.xml_parser import JUnitXMLParser
from .parsers.stats_parser import get_count_from_stats
from .outputs.console import ConsoleOutput
from .validation.data_validator import validate_test_counts


class SummaryReporter:
    """Generate comprehensive test summary."""

    def __init__(self, config: SummaryConfigFromLines):
        self.config = config

    def generate(self) -> None:
        parser = LITLogParser(self.config.log_lines)
        stats = parser.extract_statistics()
        test_lists, declared_counts = parser.extract_test_lists()
        total_discovered = get_count_from_stats(stats, ["Total Discovered"])

        xml_parser = JUnitXMLParser(self.config.xml_file)
        parsed_xml = xml_parser.extract_tests_from_xml()
        skipped_xml = parsed_xml.skipped
        excluded_xml = parsed_xml.excluded

        ConsoleOutput.print_statistics(stats)

        skipped_result = self._analyze_skipped_tests(test_lists, stats, skipped_xml)
        self._validate_skipped_counts(skipped_result, declared_counts, stats)
        self._display_skipped_tests(skipped_result)
        if skipped_result["count"] > 0:
            self._cleanup_skipped_from_test_lists(test_lists)

        excluded_result = self._analyze_excluded_tests(test_lists, stats, excluded_xml)
        self._validate_excluded_counts(excluded_result, declared_counts, stats)
        self._display_excluded_tests(excluded_result)
        if excluded_result["count"] > 0:
            self._cleanup_excluded_from_test_lists(test_lists)

        self._display_remaining_categories(test_lists)

        validate_test_counts(
            total_discovered,
            test_lists,
            skipped_result["count"],
            excluded_result["count"],
        )

        ConsoleOutput.print_timing_summary(self.config.log_lines)

    def _analyze_skipped_tests(
        self, test_lists: TestLists, stats: List[str], skipped_xml: List[str]
    ) -> SkippedTestsResult:
        """Analyze skipped tests (priority: XML > Log > Stats)."""
        skipped_from_log = test_lists.get("Skipped", test_lists.get("Unsupported", []))
        stats_count = get_count_from_stats(stats, ["Skipped", "Unsupported"])

        # Priority 1: XML data (most reliable - structured output)
        if skipped_xml:
            return SkippedTestsResult(
                tests=skipped_xml,
                count=len(skipped_xml),
                source="xml",
                note="",
            )

        # Priority 2: Log data
        if skipped_from_log:
            return SkippedTestsResult(
                tests=skipped_from_log,
                count=len(skipped_from_log),
                source="log",
                note="",
            )

        # Priority 3: Stats only (no individual test names)
        if stats_count:
            return SkippedTestsResult(
                tests=[],
                count=stats_count,
                source="stats",
                note="Warning: Test names not available",
            )

        # No data available
        return SkippedTestsResult(tests=[], count=0, source="none", note="")

    def _validate_skipped_counts(
        self, result: SkippedTestsResult, declared_counts: TestCounts, stats: List[str]
    ) -> None:
        """Validate skipped counts (warns on mismatch)."""
        actual_count = result["count"]
        if actual_count == 0:
            return  # Nothing to validate

        declared_count = declared_counts.get(
            "Skipped", declared_counts.get("Unsupported", 0)
        )
        stats_count = get_count_from_stats(stats, ["Skipped", "Unsupported"])

        # Build list of mismatches
        mismatches = []

        if declared_count and declared_count != actual_count:
            mismatches.append(f"log header: {declared_count}")

        if stats_count and stats_count != actual_count:
            mismatches.append(f"statistics: {stats_count}")

        # Display warning only if mismatches found
        if mismatches:
            sources_str = ", ".join(mismatches)
            print(
                f"Warning: Skipped test count mismatch. "
                f"Using {actual_count} from {result['source']}, "
                f"but found {sources_str}",
                file=sys.stderr,
            )

    def _display_skipped_tests(self, result: SkippedTestsResult) -> None:
        if result["count"] > 0:
            ConsoleOutput.print_test_group(
                "Skipped Tests",
                result["tests"],
                note=result["note"],
                count=result["count"] if not result["tests"] else None,
            )

    def _cleanup_skipped_from_test_lists(self, test_lists: TestLists) -> None:
        test_lists.pop("Skipped", None)
        test_lists.pop("Unsupported", None)

    def _analyze_excluded_tests(
        self, test_lists: TestLists, stats: List[str], excluded_xml: List[str]
    ) -> ExcludedTestsResult:
        """Analyze excluded tests (priority: Log > XML > Stats)."""
        excluded_from_log = test_lists.get("Excluded", [])
        stats_count = get_count_from_stats(stats, ["Excluded"])

        # Priority 1: Log data
        if excluded_from_log:
            return ExcludedTestsResult(
                tests=excluded_from_log,
                count=len(excluded_from_log),
                source="log",
                note="",
            )

        # Priority 2: XML data
        if excluded_xml:
            return ExcludedTestsResult(
                tests=excluded_xml, count=len(excluded_xml), source="xml", note=""
            )

        # Priority 3: Stats only (no individual test names)
        if stats_count:
            return ExcludedTestsResult(
                tests=[],
                count=stats_count,
                source="stats",
                note="Warning: Test names not available",
            )

        # No data available
        return ExcludedTestsResult(tests=[], count=0, source="none", note="")

    def _validate_excluded_counts(
        self, result: ExcludedTestsResult, declared_counts: TestCounts, stats: List[str]
    ) -> None:
        """Validate excluded counts (warns on mismatch)."""
        actual_count = result["count"]
        if actual_count == 0:
            return  # Nothing to validate

        declared_count = declared_counts.get("Excluded", 0)
        stats_count = get_count_from_stats(stats, ["Excluded"])

        # Build list of mismatches
        mismatches = []

        if declared_count and declared_count != actual_count:
            mismatches.append(f"log header: {declared_count}")

        if stats_count and stats_count != actual_count:
            mismatches.append(f"statistics: {stats_count}")

        # Display warning only if mismatches found
        if mismatches:
            sources_str = ", ".join(mismatches)
            print(
                f"Warning: Excluded test count mismatch. "
                f"Using {actual_count} from {result['source']}, "
                f"but found {sources_str}",
                file=sys.stderr,
            )

    def _display_excluded_tests(self, result: ExcludedTestsResult) -> None:
        if result["count"] > 0:
            ConsoleOutput.print_test_group(
                "Excluded Tests",
                result["tests"],
                note=result["note"],
                count=result["count"] if not result["tests"] else None,
            )

    def _cleanup_excluded_from_test_lists(self, test_lists: TestLists) -> None:
        test_lists.pop("Excluded", None)

    def _display_remaining_categories(self, test_lists: TestLists) -> None:
        for category, tests in test_lists.items():
            if tests:
                ConsoleOutput.print_test_group(f"{category} Tests", tests)

