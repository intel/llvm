"""Parse LIT text output for test information."""
import sys
from typing import List, Tuple

from ..constants import (
    FAIL_TIMEOUT_PATTERN,
    TEST_LIST_HEADER_PATTERN,
    STATS_PATTERN,
    TEST_CATEGORY_PATTERN,
    SLOWEST_TESTS_HEADER,
    TEST_TIMES_HEADERS,
)
from ..models.test_data import TestLists, TestCounts, TimingSummary


def read_log_file(log_path: str) -> List[str]:
    """Read log file and return lines.
    
    Handles UTF-8 decoding with fallback to replacement characters.
    
    Args:
        log_path: Path to log file.
    
    Returns:
        List of lines from the file.
    
    Raises:
        OSError: If file cannot be read.
    """
    try:
        # Try strict decoding first
        with open(log_path, "r", encoding="utf-8", errors="strict") as f:
            return f.readlines()
    except UnicodeDecodeError:
        # Fallback to replacement and log warning
        print(
            f"Warning: Log contains non-UTF-8 characters, "
            f"replacing with U+FFFD",
            file=sys.stderr,
        )
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
        except OSError as e:
            raise OSError(f"Cannot read log file: {e}") from e
    except OSError as e:
        raise OSError(f"Cannot read log file: {e}") from e


class LITLogParser:
    """Parse LIT (llvm-lit) text output.
    
    Format compatibility: LLVM 15+ (stable since ~2010).
    Parses text output from LLVM LIT with flags:
    --verbose --time-tests --show-unsupported --show-pass --show-xfail
    """

    def __init__(self, lines: List[str]):
        """Initialize parser with log lines.
        
        Args:
            lines: List of log file lines.
        """
        self.lines = lines

    def extract_error_details(self) -> List[str]:
        """Extract error details from FAIL/TIMEOUT entries.
        
        Returns:
            List of lines containing error details.
        """
        result = []
        in_error = False

        for line in self.lines:
            if FAIL_TIMEOUT_PATTERN.match(line):
                in_error = True

            # Stop at test list headers or timing summaries
            if in_error and (
                TEST_LIST_HEADER_PATTERN.match(line)
                or line.strip() == SLOWEST_TESTS_HEADER
                or line.strip() in TEST_TIMES_HEADERS
            ):
                break

            if in_error:
                result.append(line)

        return result

    def extract_statistics(self) -> List[str]:
        """Extract test statistics from LIT summary.
        
        Returns:
            List of statistics lines.
        """
        return [line for line in self.lines if STATS_PATTERN.match(line)]

    def extract_time_summary(self) -> TimingSummary:
        """Extract timing from LIT --time-tests (slowest tests and histogram).
        
        Returns:
            TimingSummary with slowest tests and timing histogram.
        """
        result: TimingSummary = {"slowest": [], "histogram": []}
        current_section = None
        skip_next_hr = False

        for line in self.lines:
            stripped = line.strip()

            if stripped == SLOWEST_TESTS_HEADER:
                current_section = "slowest"
                skip_next_hr = True
                continue
            elif stripped in TEST_TIMES_HEADERS:
                current_section = "histogram"
                skip_next_hr = True
                continue

            if skip_next_hr and stripped.startswith("---"):
                skip_next_hr = False
                continue

            if current_section == "slowest":
                if not stripped:
                    current_section = None
                elif not stripped.startswith("---"):
                    result["slowest"].append(line.rstrip())
            elif current_section == "histogram":
                if not stripped:
                    current_section = None
                elif stripped.replace("*", "") == "":
                    current_section = None
                elif stripped.startswith("[") or stripped.replace("-", "") == "":
                    result["histogram"].append(line.rstrip())
                else:
                    current_section = None

        return result

    def extract_test_lists(self) -> Tuple[TestLists, TestCounts]:
        """Extract categorized test lists and counts from LIT summary.
        
        Returns:
            Tuple of (test_lists, declared_counts) where:
            - test_lists: Dictionary mapping category names to test lists
            - declared_counts: Dictionary mapping category names to declared counts
        """
        categories: TestLists = {}
        declared_counts: TestCounts = {}
        current_category = None
        current_tests = []
        current_declared_count = 0

        for line in self.lines:
            match = TEST_CATEGORY_PATTERN.match(line)
            if match:
                if current_category:
                    categories[current_category] = current_tests
                    declared_counts[current_category] = current_declared_count

                current_category = match.group(1)
                current_declared_count = int(match.group(2))
                current_tests = []
                continue

            if current_category:
                if not line.strip():
                    categories[current_category] = current_tests
                    declared_counts[current_category] = current_declared_count
                    current_category = None
                    current_tests = []
                    current_declared_count = 0
                else:
                    test_name = line.strip()
                    if test_name:
                        current_tests.append(test_name)

        if current_category:
            categories[current_category] = current_tests
            declared_counts[current_category] = current_declared_count

        return categories, declared_counts


# Standalone functions for backward compatibility
def extract_error_details(lines: List[str]) -> List[str]:
    """Extract error details from FAIL/TIMEOUT entries.
    
    Convenience function. For new code, prefer using LITLogParser class.
    """
    parser = LITLogParser(lines)
    return parser.extract_error_details()


def extract_statistics(lines: List[str]) -> List[str]:
    """Extract test statistics from LIT summary.
    
    Convenience function. For new code, prefer using LITLogParser class.
    """
    parser = LITLogParser(lines)
    return parser.extract_statistics()


def extract_time_summary(lines: List[str]) -> TimingSummary:
    """Extract timing from LIT --time-tests.
    
    Convenience function. For new code, prefer using LITLogParser class.
    """
    parser = LITLogParser(lines)
    return parser.extract_time_summary()


def extract_test_lists(lines: List[str]) -> Tuple[TestLists, TestCounts]:
    """Extract categorized test lists and counts from LIT summary.
    
    Convenience function. For new code, prefer using LITLogParser class.
    """
    parser = LITLogParser(lines)
    return parser.extract_test_lists()
