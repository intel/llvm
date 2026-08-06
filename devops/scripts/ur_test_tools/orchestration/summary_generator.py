"""Generate test summary reports."""
from typing import List

from ..models.config import SummaryConfigLegacy
from ..models.test_data import (
    TestLists,
    TestCounts,
    SkippedTestsResult,
    ExcludedTestsResult,
)
from ..parsers.log_parser import extract_statistics, extract_test_lists
from ..parsers.xml_parser import extract_tests_from_xml
from ..parsers.stats_parser import get_count_from_stats
from ..formatters.display import TestResultDisplay
from ..validation.data_validator import validate_test_counts


class SummaryGenerator:
    """Generate comprehensive test summary.
    
    Separates analysis logic from display logic for better testability.
    """

    def __init__(self, config: SummaryConfigLegacy):
        """Initialize generator with configuration.
        
        Args:
            config: Summary configuration with log lines and optional XML path.
        """
        self.config = config

    def generate(self) -> None:
        """Generate and display complete test summary."""
        stats = extract_statistics(self.config.log_lines)
        test_lists, declared_counts = extract_test_lists(self.config.log_lines)
        total_discovered = get_count_from_stats(stats, ["Total Discovered"])

        skipped_xml, excluded_xml = extract_tests_from_xml(self.config.xml_file)

        TestResultDisplay.print_statistics(stats)

        skipped_result = self._analyze_skipped_tests(
            test_lists, declared_counts, stats, skipped_xml
        )
        self._display_skipped_tests(skipped_result)
        if skipped_result["count"] > 0:
            self._cleanup_skipped_from_test_lists(test_lists)

        excluded_result = self._analyze_excluded_tests(
            test_lists, stats, excluded_xml
        )
        self._display_excluded_tests(excluded_result)
        if excluded_result["count"] > 0:
            self._cleanup_excluded_from_test_lists(test_lists)

        self._display_remaining_categories(test_lists)

        validate_test_counts(
            total_discovered,
            test_lists,
            skipped_result["count"],
            excluded_result["count"]
        )

        TestResultDisplay.print_timing_summary(self.config.log_lines)

    def _analyze_skipped_tests(
        self,
        test_lists: TestLists,
        declared_counts: TestCounts,
        stats: List[str],
        skipped_xml: List[str]
    ) -> SkippedTestsResult:
        """Analyze skipped tests from multiple sources.
        
        Pure function - no side effects, just returns result.
        
        Args:
            test_lists: Test lists from log parsing.
            declared_counts: Declared counts from log headers.
            stats: Statistics lines from log.
            skipped_xml: Skipped tests from XML.
        
        Returns:
            SkippedTestsResult with tests, count, source, and note.
        """
        skipped_from_log = test_lists.get(
            "Skipped", test_lists.get("Unsupported", [])
        )
        declared_count = declared_counts.get(
            "Skipped", declared_counts.get("Unsupported", 0)
        )
        stats_count = get_count_from_stats(stats, ["Skipped", "Unsupported"])

        # Priority 1: Log data matches declared count (most reliable)
        if skipped_from_log and declared_count:
            actual_count = len(skipped_from_log)

            if actual_count == declared_count:
                return SkippedTestsResult(
                    tests=skipped_from_log,
                    count=actual_count,
                    source="log",
                    note=""
                )

            # Mismatch detected - prefer XML if available
            if skipped_xml:
                return SkippedTestsResult(
                    tests=skipped_xml,
                    count=len(skipped_xml),
                    source="xml",
                    note=(
                        f"Note: Using XML data (log header claimed "
                        f"{declared_count}, but found {actual_count} lines)."
                    )
                )

            # No XML - use log with warning
            return SkippedTestsResult(
                tests=skipped_from_log,
                count=actual_count,
                source="log",
                note=(
                    f"Warning: Log header claimed {declared_count} skipped, "
                    f"but found {actual_count} lines)."
                )
            )

        # Priority 2: Stats count available (no individual test names)
        if stats_count:
            if skipped_xml:
                return SkippedTestsResult(
                    tests=skipped_xml,
                    count=len(skipped_xml),
                    source="xml",
                    note=""
                )

            return SkippedTestsResult(
                tests=[],
                count=stats_count,
                source="stats",
                note=(
                    f"Warning: Could not extract individual skipped test names.\n"
                    f"Statistics show {stats_count} skipped tests, but they "
                    f"are not available in the output."
                )
            )

        # No data available
        return SkippedTestsResult(tests=[], count=0, source="none", note="")

    def _display_skipped_tests(self, result: SkippedTestsResult) -> None:
        """Display skipped tests result.
        
        Pure display function - no logic, just formatting.
        
        Args:
            result: Skipped tests analysis result.
        """
        if result["count"] > 0:
            TestResultDisplay.print_test_group(
                "Skipped Tests",
                result["tests"],
                note=result["note"],
                count=result["count"] if not result["tests"] else None
            )

    def _cleanup_skipped_from_test_lists(self, test_lists: TestLists) -> None:
        """Remove skipped tests from lists (explicit side effect).
        
        Args:
            test_lists: Test lists dictionary to modify.
        """
        test_lists.pop("Skipped", None)
        test_lists.pop("Unsupported", None)

    def _analyze_excluded_tests(
        self,
        test_lists: TestLists,
        stats: List[str],
        excluded_xml: List[str]
    ) -> ExcludedTestsResult:
        """Analyze excluded tests from multiple sources.
        
        Pure function - no side effects, just returns result.
        
        Args:
            test_lists: Test lists from log parsing.
            stats: Statistics lines from log.
            excluded_xml: Excluded tests from XML.
        
        Returns:
            ExcludedTestsResult with tests, count, source, and note.
        """
        excluded_from_log = test_lists.get("Excluded", [])
        stats_count = get_count_from_stats(stats, ["Excluded"])

        # Priority 1: Log data
        if excluded_from_log:
            return ExcludedTestsResult(
                tests=excluded_from_log,
                count=len(excluded_from_log),
                source="log",
                note=""
            )

        # Priority 2: XML data
        if excluded_xml:
            return ExcludedTestsResult(
                tests=excluded_xml,
                count=len(excluded_xml),
                source="xml",
                note=""
            )

        # Priority 3: Stats only (no individual test names)
        if stats_count:
            return ExcludedTestsResult(
                tests=[],
                count=stats_count,
                source="stats",
                note="Warning: Test names not available"
            )

        # No data available
        return ExcludedTestsResult(tests=[], count=0, source="none", note="")

    def _display_excluded_tests(self, result: ExcludedTestsResult) -> None:
        """Display excluded tests result.
        
        Pure display function - no logic, just formatting.
        
        Args:
            result: Excluded tests analysis result.
        """
        if result["count"] > 0:
            TestResultDisplay.print_test_group(
                "Excluded Tests",
                result["tests"],
                note=result["note"],
                count=result["count"] if not result["tests"] else None
            )

    def _cleanup_excluded_from_test_lists(self, test_lists: TestLists) -> None:
        """Remove excluded tests from lists (explicit side effect).
        
        Args:
            test_lists: Test lists dictionary to modify.
        """
        test_lists.pop("Excluded", None)

    def _display_remaining_categories(self, test_lists: TestLists) -> None:
        """Display all remaining test categories.
        
        Args:
            test_lists: Test lists dictionary.
        """
        for category, tests in test_lists.items():
            if tests:
                TestResultDisplay.print_test_group(f"{category} Tests", tests)


# Standalone function for backward compatibility
def show_statistics_and_lists(config: SummaryConfigLegacy) -> None:
    """Display test statistics and categorized test lists.
    
    This is a convenience function that creates a generator and runs it.
    For new code, prefer using SummaryGenerator class directly.
    
    Args:
        config: Summary configuration with log lines and XML file path.
    """
    generator = SummaryGenerator(config)
    generator.generate()
