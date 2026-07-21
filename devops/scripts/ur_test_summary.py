#!/usr/bin/env python3
"""
Unified Runtime test summary processing for GitHub Actions CI.

This script processes LIT test output logs to:
- Extract error details from failures/timeouts
- Show statistics and collapsed test lists for GitHub Actions
"""

import sys
import re
from typing import List, Dict, Tuple, Optional, TypedDict
from pathlib import Path
from dataclasses import dataclass

import defusedxml.ElementTree as ET


# Type definitions for structured data
class TestLists(TypedDict, total=False):
    """Type definition for test list dictionary."""

    Passed: List[str]
    Failed: List[str]
    Skipped: List[str]
    Unsupported: List[str]
    Excluded: List[str]
    Unresolved: List[str]


class TestCounts(TypedDict, total=False):
    """Type definition for test count dictionary."""

    Passed: int
    Failed: int
    Skipped: int
    Unsupported: int
    Excluded: int
    Unresolved: int


class TimingSummary(TypedDict):
    """Type definition for test timing summary."""

    slowest: List[str]
    histogram: List[str]


@dataclass
class SummaryConfig:
    """Configuration for show_statistics_and_lists function."""

    log_lines: List[str]
    xml_file: Optional[str] = None


FAIL_TIMEOUT_PATTERN = re.compile(r"^(FAIL|TIMEOUT):")
TEST_LIST_HEADER_PATTERN = re.compile(
    r"^(Passed|Unsupported|Failed|Expectedly Failed|"
    r"Timed Out|Unexpectedly Passed|Unresolved) Tests \("
)
STATS_PATTERN = re.compile(
    r"^\s*(Total Discovered|Expected Passes|Expectedly Failed|"
    r"Excluded|Unsupported|Skipped|Passed|Passed With Retry|"
    r"Failed|Timed Out|Unexpectedly Passed|Unresolved)(\s+Tests)?\s*:"
)
TEST_CATEGORY_PATTERN = re.compile(r"^([A-Za-z]+(?: [A-Za-z]+)*) Tests \((\d+)\):")

TEST_NOT_SELECTED_MSG = "Test not selected"
SEPARATOR_WIDTH = 70


def read_log_file(log_path: str) -> List[str]:
    """Read log file and return lines."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        sys.exit(1)


def extract_error_details(lines: List[str]) -> List[str]:
    """Extract error details from FAIL/TIMEOUT entries."""
    result = []
    in_error = False

    for line in lines:
        if FAIL_TIMEOUT_PATTERN.match(line):
            in_error = True

        if in_error and TEST_LIST_HEADER_PATTERN.match(line):
            break

        if in_error:
            result.append(line)

    return result


def extract_statistics(lines: List[str]) -> List[str]:
    """Extract test statistics from LIT summary."""
    return [line for line in lines if STATS_PATTERN.match(line)]


def extract_time_summary(lines: List[str]) -> TimingSummary:
    """Extract test timing from LIT --time-tests output (slowest tests and histogram)."""
    result: TimingSummary = {"slowest": [], "histogram": []}
    current_section = None
    skip_next_hr = False

    for line in lines:
        stripped = line.strip()

        if stripped == "Slowest Tests:":
            current_section = "slowest"
            skip_next_hr = True
            continue
        elif stripped == "Tests Times:" or stripped == "Test Times:":
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


def _should_include_test(message: str, include_test_not_selected: bool) -> bool:
    has_test_not_selected = TEST_NOT_SELECTED_MSG in message
    if include_test_not_selected:
        return has_test_not_selected
    return not has_test_not_selected


def _format_test_name(classname: str, name: str) -> str:
    if classname and name:
        return f"{classname}.{name}"
    return name


def _extract_tests_from_xml_by_filter(
    xml_path: str, include_test_not_selected: bool, test_type_name: str
) -> List[str]:
    """Extract tests from LIT xunit XML by filtering on 'Test not selected' message."""
    if not xml_path or not Path(xml_path).exists():
        if xml_path and not Path(xml_path).exists():
            print(f"Note: XML file not found: {xml_path}", file=sys.stderr)
        return []

    tests = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for testcase in root.findall(".//testcase"):
            skipped_elem = testcase.find("skipped")
            if skipped_elem is None:
                continue

            message = skipped_elem.get("message", "")
            if not _should_include_test(message, include_test_not_selected):
                continue

            test_name = _format_test_name(
                testcase.get("classname", ""), testcase.get("name", "")
            )
            if test_name:
                tests.append(test_name)

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML file {xml_path}: {e}", file=sys.stderr)
    except (OSError, ValueError) as e:
        print(f"Warning: Error reading XML file {xml_path}: {e}", file=sys.stderr)

    if tests and Path(xml_path).exists():
        print(
            f"Note: Found {len(tests)} {test_type_name} tests in XML file",
            file=sys.stderr,
        )

    return tests


def extract_skipped_from_xml(xml_path: str) -> List[str]:
    """Extract skipped test names from LIT xunit XML (excludes 'Test not selected')."""
    return _extract_tests_from_xml_by_filter(
        xml_path, include_test_not_selected=False, test_type_name="skipped"
    )


def extract_excluded_from_xml(xml_path: str) -> List[str]:
    """Extract excluded test names from LIT xunit XML (marked as 'Test not selected')."""
    return _extract_tests_from_xml_by_filter(
        xml_path, include_test_not_selected=True, test_type_name="excluded"
    )


def extract_test_lists(
    lines: List[str],
) -> Tuple[TestLists, TestCounts]:
    """Extract categorized test lists and counts from LIT summary."""
    categories: TestLists = {}
    declared_counts: TestCounts = {}
    current_category = None
    current_tests = []
    current_declared_count = 0

    for line in lines:
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


def filter_log_for_display(lines: List[str]) -> List[str]:
    """Filter log to remove statistics, test lists, and timing sections."""
    result = []
    skip_until_empty = False
    in_timing = False

    for line in lines:
        stripped = line.strip()

        if STATS_PATTERN.match(line):
            continue

        if TEST_CATEGORY_PATTERN.match(line):
            skip_until_empty = True
            continue

        if stripped in ["Slowest Tests:", "Tests Times:", "Test Times:"]:
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


def _print_test_group(
    title: str, tests: List[str], note: str = None, count: int = None
) -> None:
    """Print a collapsible GitHub Actions group with test list."""
    test_count = count if count is not None else len(tests)
    print(f"::group::{title} ({test_count})")
    if note:
        print(note)
        print()
    for test in tests:
        print(test)
    print("::endgroup::")


def _display_statistics(stats: List[str]) -> None:
    if stats:
        print("=== Test Statistics ===")
        for stat in stats:
            print(stat.rstrip())
        print()


def _get_count_from_stats(stats: List[str], keywords: List[str]) -> int:
    for stat in stats:
        if any(keyword in stat for keyword in keywords):
            match = re.search(r"(\d+)", stat)
            if match:
                return int(match.group(1))
    return 0


def _display_skipped_tests(
    test_lists: TestLists,
    declared_counts: TestCounts,
    stats: List[str],
    xml_file: Optional[str],
) -> int:
    skipped_from_log = test_lists.get("Skipped", test_lists.get("Unsupported", []))
    declared_count = declared_counts.get(
        "Skipped", declared_counts.get("Unsupported", 0)
    )
    stats_count = _get_count_from_stats(stats, ["Skipped:", "Unsupported:"])

    if skipped_from_log and declared_count:
        actual_count = len(skipped_from_log)

        if actual_count == declared_count:
            _print_test_group("Skipped Tests", skipped_from_log)
            test_lists.pop("Skipped", None)
            test_lists.pop("Unsupported", None)
            return actual_count

        skipped_xml = extract_skipped_from_xml(xml_file)
        if skipped_xml:
            note = f"Note: Using XML data (log header claimed {declared_count}, but found {actual_count} lines)."
            _print_test_group("Skipped Tests", skipped_xml, note)
            count = len(skipped_xml)
        else:
            note = f"Warning: Log header claimed {declared_count} skipped, but found {actual_count} lines."
            _print_test_group("Skipped Tests", skipped_from_log, note)
            count = actual_count

        test_lists.pop("Skipped", None)
        test_lists.pop("Unsupported", None)
        return count

    elif stats_count:
        skipped_xml = extract_skipped_from_xml(xml_file)
        if skipped_xml:
            _print_test_group("Skipped Tests", skipped_xml)
            return len(skipped_xml)

        note = f"Warning: Could not extract individual skipped test names.\nStatistics show {stats_count} skipped tests, but they are not available in the output."
        _print_test_group("Skipped Tests", [], note, count=stats_count)
        return stats_count

    return 0


def _display_excluded_tests(
    test_lists: TestLists, stats: List[str], xml_file: Optional[str]
) -> int:
    if "Excluded" in test_lists:
        return 0

    excluded_xml = extract_excluded_from_xml(xml_file)
    stats_count = _get_count_from_stats(stats, ["Excluded:"])

    if excluded_xml:
        _print_test_group("Excluded Tests", excluded_xml)
        return len(excluded_xml)

    elif stats_count:
        note = f"Warning: Could not extract individual excluded test names.\nStatistics show {stats_count} excluded tests, but they are not available in the output."
        _print_test_group("Excluded Tests", [], note, count=stats_count)
        return stats_count

    return 0


def _display_remaining_categories(test_lists: TestLists) -> None:
    for category, tests in test_lists.items():
        if tests:
            _print_test_group(f"{category} Tests", tests)


def _validate_test_counts(
    total_discovered: int,
    test_lists: TestLists,
    displayed_skipped: int,
    displayed_excluded: int,
) -> None:
    if not total_discovered:
        return

    sum_categories = sum(len(tests) for tests in test_lists.values())

    if displayed_skipped > 0 and "Skipped" not in test_lists:
        sum_categories += displayed_skipped
    if displayed_excluded > 0 and "Excluded" not in test_lists:
        sum_categories += displayed_excluded

    if total_discovered != sum_categories:
        print()
        print(
            f"::warning::Test count mismatch: Total Discovered = {total_discovered}, but sum of all categories = {sum_categories}"
        )
        print(
            f"Warning: {total_discovered - sum_categories} tests are unaccounted for."
        )
        print(f"This may indicate tests in unexpected categories or parsing issues.")
        print(f"Categories found: {', '.join(test_lists.keys())}")
        if displayed_skipped > 0:
            print(f"(Plus {displayed_skipped} skipped tests displayed separately)")
        if displayed_excluded > 0:
            print(f"(Plus {displayed_excluded} excluded tests displayed separately)")
        print()


def _display_timing_summary(lines: List[str]) -> None:
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
        print("Slowest Tests:")
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


def show_statistics_and_lists(config: SummaryConfig) -> None:
    """Display test statistics and categorized test lists."""
    stats = extract_statistics(config.log_lines)
    _display_statistics(stats)

    total_discovered = _get_count_from_stats(stats, ["Total Discovered"])
    test_lists, declared_counts = extract_test_lists(config.log_lines)

    displayed_skipped = _display_skipped_tests(
        test_lists, declared_counts, stats, config.xml_file
    )
    displayed_excluded = _display_excluded_tests(test_lists, stats, config.xml_file)
    _display_remaining_categories(test_lists)
    _validate_test_counts(
        total_discovered, test_lists, displayed_skipped, displayed_excluded
    )
    _display_timing_summary(config.log_lines)





def _validate_log_path(path: str) -> None:
    if ".." in path or path.startswith("/"):
        print(
            f"Error: Invalid log file path (absolute paths and '..' not allowed): {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not Path(path).exists():
        print(f"Error: Log file not found: {path}", file=sys.stderr)
        sys.exit(1)


def _validate_optional_path(
    path: str, path_type: str, allow_absolute: bool = False
) -> str:
    if not path:
        return None
    if ".." in path:
        print(
            f"Error: Invalid {path_type} file path (path traversal): {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not allow_absolute and path.startswith("/"):
        print(
            f"Error: Invalid {path_type} file path (absolute paths not allowed): {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def main():
    if len(sys.argv) < 3:
        print(f"Error: {sys.argv[0]} <command> <log_file> [xml_file]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    log_file = sys.argv[2]
    _validate_log_path(log_file)
    lines = read_log_file(log_file)

    if command == "extract-errors":
        for line in extract_error_details(lines):
            print(line, end="")

    elif command == "filter-log":
        for line in filter_log_for_display(lines):
            print(line, end="")

    elif command == "show-summary":
        xml_file = _validate_optional_path(
            sys.argv[3] if len(sys.argv) > 3 else None, "XML", allow_absolute=True
        )
        config = SummaryConfig(log_lines=lines, xml_file=xml_file)
        show_statistics_and_lists(config)

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
