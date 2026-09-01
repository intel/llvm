"""Console output formatting for test results."""

from typing import List

from ..constants import (
    SLOWEST_TESTS_HEADER,
    TEST_CATEGORY_PATTERN,
    TEST_TIMES_HEADERS,
)
from ..parsers.log_parser import LITLogParser


class ConsoleOutput:
    """Format test results for console output."""

    @staticmethod
    def filter_log_for_display(lines: List[str]) -> List[str]:
        """Remove statistics, test lists, and timing from log."""
        result = []
        statistics = set(LITLogParser(lines).extract_statistics())
        skip_until_empty = False
        in_timing = False

        for line in lines:
            stripped = line.strip()

            # Skip statistics lines
            if line in statistics:
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
