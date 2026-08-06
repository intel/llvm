"""Console output formatting for test results."""
from typing import List, Optional

from ..constants import (
    SEPARATOR_WIDTH,
    SLOWEST_TESTS_HEADER,
    STATS_PATTERN,
    TEST_CATEGORY_PATTERN,
    TEST_TIMES_HEADERS,
)
from ..models.test_data import TimingSummary
from ..parsers.log_parser import extract_time_summary


class ConsoleOutput:
    """Format test results for console output."""

    @staticmethod
    def print_test_group(
        title: str,
        tests: List[str],
        note: str = "",
        count: Optional[int] = None
    ) -> None:
        """Print a collapsible GitHub Actions group with test list.

        Args:
            title: Group title.
            tests: List of test names to display.
            note: Optional note to display before tests.
            count: Optional count to override len(tests) in title.
        """
        test_count = count if count is not None else len(tests)
        print(f"::group::{title} ({test_count})")
        if note:
            print(note)
            print()
        for test in tests:
            print(test)
        print("::endgroup::")

    @staticmethod
    def print_statistics(stats: List[str]) -> None:
        """Print statistics section.

        Args:
            stats: List of statistics lines.
        """
        if stats:
            print("=== Test Statistics ===")
            for stat in stats:
                print(stat.rstrip())
            print()

    @staticmethod
    def print_timing_summary(lines: List[str]) -> None:
        """Print timing information section.

        Args:
            lines: Log lines containing timing information.
        """
        time_info = extract_time_summary(lines)

        testing_time = None
        for line in lines:
            if line.strip().startswith("Testing Time:"):
                testing_time = line.strip()
                break

        if not (time_info["slowest"] or time_info["histogram"] or testing_time):
            return

        print("::group::Test Timing Summary")

        if testing_time:
            print(testing_time)
            print()

        if time_info["slowest"]:
            print(SLOWEST_TESTS_HEADER)
            print("-" * SEPARATOR_WIDTH)
            for line in time_info["slowest"]:
                print(line)
            print()

        if time_info["histogram"]:
            print("Test Times Distribution:")
            print("-" * SEPARATOR_WIDTH)
            for line in time_info["histogram"]:
                print(line)

        print("::endgroup::")


def filter_log_for_display(lines: List[str]) -> List[str]:
    """Filter log to remove statistics, test lists, and timing sections.

    Args:
        lines: Original log lines.

    Returns:
        Filtered log lines suitable for display.
    """
    result = []
    skip_until_empty = False
    in_timing = False

    for line in lines:
        stripped = line.strip()

        # Skip statistics lines
        if STATS_PATTERN.match(line):
            continue

        # Skip test category sections
        if TEST_CATEGORY_PATTERN.match(line):
            skip_until_empty = True
            continue

        # Skip timing sections
        if stripped == SLOWEST_TESTS_HEADER or stripped in TEST_TIMES_HEADERS:
            in_timing = True
            continue

        if in_timing and stripped.replace("*", "") == "":
            in_timing = False
            continue

        if in_timing:
            continue

        if stripped.startswith("Testing Time:"):
            continue

        if skip_until_empty:
            if not stripped:
                skip_until_empty = False
            continue

        result.append(line)

    return result
