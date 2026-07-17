#!/usr/bin/env python3
"""
Unified Runtime test summary processing for GitHub Actions CI.

This script processes LIT test output logs to:
- Extract error details from failures/timeouts
- Show statistics and collapsed test lists for GitHub Actions
"""

import sys
import re
from typing import List, Dict, Tuple
from pathlib import Path

import defusedxml.ElementTree as ET

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
GTEST_SKIPPED_HEADER = re.compile(r"^\[\s+SKIPPED\s+\] \d+ tests, listed below:")
GTEST_SKIPPED_TEST = re.compile(r"^\[\s+SKIPPED\s+\] (.+)$")
GTEST_SKIPPED_FOOTER = re.compile(r"^\s*\d+ SKIPPED TESTS")
UNSUPPORTED_PATTERN = re.compile(r"^(UNSUPPORTED|SKIP):\s+(.+?)(?:\s+\(|$)")

TEST_NOT_SELECTED_MSG = "Test not selected"


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
    """
    Extract error details from FAIL/TIMEOUT entries.

    Stops before test list sections to avoid duplication
    (lists are shown in collapsed sections).
    """
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
    """
    Extract test statistics from LIT summary.

    Matches lines like:
    - Total Discovered Tests: 123
    - Passed: 100 (81.30%)
    - Failed: 2 (1.63%)
    etc.
    """
    return [line for line in lines if STATS_PATTERN.match(line)]


def extract_time_summary(lines: List[str]) -> Dict[str, List[str]]:
    """
    Extract test timing information from LIT output.

    LIT with --time-tests flag produces:
    Slowest Tests:
    ----------------------------------------------------------------------
    4.16s: test_name
    0.10s: test_name2
    ...

    Tests Times:
    ----------------------------------------------------------------------
    [Range] :: [Percentage] :: [Count]
    ----------------------------------------------------------------------
    [0.00s, 1.00s) :: [========] :: [10/20]
    ...
    ----------------------------------------------------------------------
    ********************

    Returns dictionary with 'slowest' and 'histogram' keys containing line lists.
    """
    result = {"slowest": [], "histogram": []}
    current_section = None
    skip_next_hr = False

    for line in lines:
        stripped = line.strip()

        # Detect section starts
        if stripped == "Slowest Tests:":
            current_section = "slowest"
            skip_next_hr = True
            continue
        elif stripped == "Tests Times:" or stripped == "Test Times:":
            current_section = "histogram"
            skip_next_hr = True
            continue

        # Skip horizontal rule after section header
        if skip_next_hr and stripped.startswith("---"):
            skip_next_hr = False
            continue

        # Collect lines for current section
        if current_section == "slowest":
            # Slowest section ends at empty line or next section
            if not stripped:
                current_section = None
            elif not stripped.startswith("---"):
                result["slowest"].append(line.rstrip())
        elif current_section == "histogram":
            # Histogram section ends at:
            # - Empty line
            # - Line with only asterisks (section separator)
            # - Line not starting with '[' and not only dashes
            if not stripped:
                current_section = None
            elif stripped.replace("*", "") == "":
                # Line contains only asterisks - end of section
                current_section = None
            elif stripped.startswith("[") or stripped.replace("-", "") == "":
                # Valid histogram line (header, data row, or separator)
                result["histogram"].append(line.rstrip())
            else:
                # Something else - end of histogram section
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
    xml_path: str,
    include_test_not_selected: bool,
    test_type_name: str
) -> List[str]:
    """
    Generic helper to extract tests from LIT xunit XML output.

    Args:
        xml_path: Path to XML file
        include_test_not_selected: If True, include only tests with "Test not selected" message.
                                   If False, exclude those tests.
        test_type_name: Name for debug messages ("skipped" or "excluded")

    Returns list of test names in format: classname.name
    """
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
        print(f"Note: Found {len(tests)} {test_type_name} tests in XML file", file=sys.stderr)

    return tests


def extract_skipped_from_xml(xml_path: str) -> List[str]:
    """
    Extract skipped test names from LIT xunit XML output.

    LIT generates XML with --xunit-xml-output flag:
    <testsuites>
      <testsuite name="..." tests="123" skipped="10">
        <testcase name="TestName" classname="TestClass">
          <skipped message="reason"/>
        </testcase>
      </testsuite>
    </testsuites>

    Returns list of skipped test names in format: classname.name
    Excludes tests with "Test not selected" message (those are excluded, not skipped).
    """
    return _extract_tests_from_xml_by_filter(
        xml_path,
        include_test_not_selected=False,
        test_type_name="skipped"
    )


def extract_excluded_from_xml(xml_path: str) -> List[str]:
    """
    Extract excluded test names from LIT xunit XML output.

    LIT marks excluded tests as skipped with specific message:
    <testsuites>
      <testsuite name="..." tests="123">
        <testcase name="TestName" classname="TestClass">
          <skipped message="Test not selected (--filter, --max-tests)"/>
        </testcase>
      </testsuite>
    </testsuites>

    Returns list of excluded test names in format: classname.name
    """
    return _extract_tests_from_xml_by_filter(
        xml_path,
        include_test_not_selected=True,
        test_type_name="excluded"
    )


def extract_skipped_from_gtest(lines: List[str]) -> List[str]:
    """
    Extract skipped test names from GoogleTest output.

    GoogleTest shows skipped tests in summary section:
    [  SKIPPED ] 3 tests, listed below:
    [  SKIPPED ] TestName1
    [  SKIPPED ] TestName2
    ...

    Returns list of skipped test names.
    """
    skipped = []
    in_summary = False

    for line in lines:
        # Start collecting after "[  SKIPPED ] N tests, listed below:"
        if re.match(r"^\[  SKIPPED \] \d+ tests, listed below:", line):
            in_summary = True
            continue

        # Stop at "X SKIPPED TESTS" line or empty line after list
        if in_summary and (re.match(r"^ *\d+ SKIPPED TESTS", line) or not line.strip()):
            break

        # Collect test names from summary
        if in_summary:
            match = re.match(r"^\[  SKIPPED \] (.+)$", line)
            if match:
                test_name = match.group(1).strip()
                skipped.append(test_name)

    return skipped


def extract_unsupported_from_lit_inline(lines: List[str]) -> List[str]:
    """
    Extract unsupported/skipped test names from LIT inline output.

    During test execution, LIT prints:
    UNSUPPORTED: TestName (reason)
    SKIP: TestName

    This is used for GoogleTest format where LIT doesn't generate
    "Unsupported Tests (N):" summary list, but shows "Skipped" in statistics.

    Returns list of unsupported/skipped test names (deduplicated).
    """
    unsupported = []
    seen = set()

    for line in lines:
        match = UNSUPPORTED_PATTERN.match(line)
        if match:
            test_name = match.group(2).strip()
            if test_name not in seen:
                seen.add(test_name)
                unsupported.append(test_name)

    return unsupported


def extract_test_lists(
    lines: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Extract categorized test lists from LIT summary.

    Looks for sections like:
        Passed Tests (N):
          test name 1
          test name 2

    Returns tuple of:
    - Dictionary with category names as keys and test lists as values
    - Dictionary with category names as keys and declared counts from headers
    """
    categories = {}
    declared_counts = {}
    current_category = None
    current_tests = []
    current_declared_count = 0

    for line in lines:
        match = TEST_CATEGORY_PATTERN.match(line)
        if match:
            # Save previous category
            if current_category:
                categories[current_category] = current_tests
                declared_counts[current_category] = current_declared_count

            # Start new category
            current_category = match.group(1)
            current_declared_count = int(match.group(2))
            current_tests = []
            continue

        # If we're in a category
        if current_category:
            # Empty line ends the category
            if not line.strip():
                categories[current_category] = current_tests
                declared_counts[current_category] = current_declared_count
                current_category = None
                current_tests = []
                current_declared_count = 0
            else:
                # Add test to current category (strip leading whitespace)
                test_name = line.strip()
                if test_name:
                    current_tests.append(test_name)

    # Save last category
    if current_category:
        categories[current_category] = current_tests
        declared_counts[current_category] = current_declared_count

    return categories, declared_counts


def parse_gtest_list(all_tests_file: str) -> List[str]:
    """
    Parse --gtest_list_tests output to get full test names.

    Format:
        TestSuite.
          TestName1/Param  # comment
          TestName2

    Returns list of full test names: TestSuite.TestName1/Param
    """
    if not Path(all_tests_file).exists():
        return []

    tests = []
    current_suite = None

    # Patterns to ignore (errors, warnings, GoogleTest verification messages)
    ignore_patterns = [
        "Error:",
        "Warning:",
        "Actual:",
        "Expected:",
        "Value of:",
        "Failure",
        "UninstantiatedParameterizedTestSuite",
        "/__w/",  # File paths from error messages
        "No platforms",
    ]

    with open(all_tests_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Skip error messages and warnings
            if any(pattern in line for pattern in ignore_patterns):
                continue

            # Skip GoogleTestVerification suite (contains only warnings, not real tests)
            if line == "GoogleTestVerification.":
                current_suite = None
                continue

            # Suite line ends with '.'
            if line.endswith(".") and not line.startswith(" "):
                current_suite = line
            # Test name (indented with exactly 2 spaces)
            elif line.startswith("  ") and current_suite:
                # Remove leading whitespace and comments
                test_name = line.strip().split(" #")[0].strip()
                if test_name:
                    full_name = f"{current_suite}{test_name}"
                    tests.append(full_name)

    return tests


def filter_log_for_display(lines: List[str]) -> List[str]:
    """
    Filter log to remove sections that are displayed separately.

    Removes:
    - Test statistics (Total Discovered, Passed:, Failed:, etc.)
    - Test lists with status (Passed Tests (N):, Failed Tests (N):, etc.)
    - Test timing summary (Testing Time, Slowest Tests, histogram)

    Returns filtered log lines suitable for "Show Full Log" section.
    """
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
            
        # Skip timing sections
        if stripped in ["Slowest Tests:", "Tests Times:", "Test Times:"]:
            in_timing = True
            continue
            
        # End of timing section at asterisks line
        if in_timing and stripped.replace("*", "") == "":
            in_timing = False
            continue
            
        # Skip content while in timing section
        if in_timing:
            continue
            
        # Skip Testing Time line
        if stripped.startswith("Testing Time:"):
            continue
            
        # Stop skipping at empty line after test list
        if skip_until_empty:
            if not stripped:
                skip_until_empty = False
            continue
            
        # Keep this line
        result.append(line)
    
    return result


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
    test_lists: Dict[str, List[str]],
    declared_counts: Dict[str, int],
    stats: List[str],
    xml_file: str
) -> int:
    skipped_from_log = test_lists.get("Skipped", test_lists.get("Unsupported", []))
    declared_count = declared_counts.get("Skipped", declared_counts.get("Unsupported", 0))
    stats_count = _get_count_from_stats(stats, ["Skipped:", "Unsupported:"])

    if skipped_from_log and declared_count:
        actual_count = len(skipped_from_log)
        
        if actual_count == declared_count:
            print(f"::group::Skipped Tests ({actual_count})")
            for test in skipped_from_log:
                print(test)
            print("::endgroup::")
            test_lists.pop("Skipped", None)
            test_lists.pop("Unsupported", None)
            return actual_count
        
        skipped_xml = extract_skipped_from_xml(xml_file)
        if skipped_xml:
            count = len(skipped_xml)
            print(f"::group::Skipped Tests ({count})")
            print(f"Note: Using XML data (log header claimed {declared_count}, but found {actual_count} lines).")
            print()
            for test in skipped_xml:
                print(test)
            print("::endgroup::")
        else:
            count = actual_count
            print(f"::group::Skipped Tests ({actual_count})")
            print(f"Warning: Log header claimed {declared_count} skipped, but found {actual_count} lines.")
            print()
            for test in skipped_from_log:
                print(test)
            print("::endgroup::")
        
        test_lists.pop("Skipped", None)
        test_lists.pop("Unsupported", None)
        return count
    
    elif stats_count:
        skipped_xml = extract_skipped_from_xml(xml_file)
        if skipped_xml:
            count = len(skipped_xml)
            print(f"::group::Skipped Tests ({count})")
            for test in skipped_xml:
                print(test)
            print("::endgroup::")
            return count
        
        print(f"::group::Skipped Tests ({stats_count})")
        print("Warning: Could not extract individual skipped test names.")
        print(f"Statistics show {stats_count} skipped tests, but they are not available in the output.")
        print("::endgroup::")
        return stats_count
    
    return 0


def _display_excluded_tests(
    test_lists: Dict[str, List[str]],
    stats: List[str],
    xml_file: str
) -> int:
    if "Excluded" in test_lists:
        return 0
    
    excluded_xml = extract_excluded_from_xml(xml_file)
    stats_count = _get_count_from_stats(stats, ["Excluded:"])
    
    if excluded_xml:
        count = len(excluded_xml)
        print(f"::group::Excluded Tests ({count})")
        for test in excluded_xml:
            print(test)
        print("::endgroup::")
        return count
    
    elif stats_count:
        print(f"::group::Excluded Tests ({stats_count})")
        print("Warning: Could not extract individual excluded test names.")
        print(f"Statistics show {stats_count} excluded tests, but they are not available in the output.")
        print("::endgroup::")
        return stats_count
    
    return 0


def _display_remaining_categories(test_lists: Dict[str, List[str]]) -> None:
    for category, tests in test_lists.items():
        count = len(tests)
        if count > 0:
            print(f"::group::{category} Tests ({count})")
            for test in tests:
                print(test)
            print("::endgroup::")


def _validate_test_counts(
    total_discovered: int,
    test_lists: Dict[str, List[str]],
    displayed_skipped: int,
    displayed_excluded: int
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
        print(f"::warning::Test count mismatch: Total Discovered = {total_discovered}, but sum of all categories = {sum_categories}")
        print(f"Warning: {total_discovered - sum_categories} tests are unaccounted for.")
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
        print("-" * 70)
        for line in time_info["slowest"]:
            print(line)
        print()
    
    if time_info["histogram"]:
        print("Test Times Distribution:")
        print("-" * 70)
        for line in time_info["histogram"]:
            print(line)
    
    print("::endgroup::")


def show_statistics_and_lists(
    lines: List[str], all_tests_file: str = None, xml_file: str = None
) -> None:
    stats = extract_statistics(lines)
    _display_statistics(stats)
    
    total_discovered = _get_count_from_stats(stats, ["Total Discovered"])
    test_lists, declared_counts = extract_test_lists(lines)
    
    displayed_skipped = _display_skipped_tests(test_lists, declared_counts, stats, xml_file)
    displayed_excluded = _display_excluded_tests(test_lists, stats, xml_file)
    _display_remaining_categories(test_lists)
    _validate_test_counts(total_discovered, test_lists, displayed_skipped, displayed_excluded)
    _display_timing_summary(lines)


def _print_usage() -> None:
    print(
        "Usage: ur_test_summary.py <command> <log_file> [all_tests_file] [xml_file]",
        file=sys.stderr,
    )
    print("\nCommands:", file=sys.stderr)
    print(
        "  extract-errors <log>  - Extract FAIL/TIMEOUT error details", file=sys.stderr
    )
    print(
        "  show-summary <log> [all_tests] [xml]  - Show statistics and collapsed test lists",
        file=sys.stderr,
    )
    print(
        "  filter-log <log>      - Filter log to remove test lists and timing (for Show Full Log)",
        file=sys.stderr,
    )
    print("\nArguments:", file=sys.stderr)
    print("  log          - LIT test output log file", file=sys.stderr)
    print(
        "  all_tests    - Optional: output from --gtest_list_tests (for computed skipped)",
        file=sys.stderr,
    )
    print(
        "  xml          - Optional: LIT xunit XML output (--xunit-xml-output)",
        file=sys.stderr,
    )


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
    if not path or path == "":
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
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if len(sys.argv) < 3:
        print(
            f"Error: Command '{command}' requires a log file argument", file=sys.stderr
        )
        sys.exit(1)

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
        all_tests_file = _validate_optional_path(
            sys.argv[3] if len(sys.argv) > 3 else None,
            "all_tests",
            allow_absolute=False,
        )
        xml_file = _validate_optional_path(
            sys.argv[4] if len(sys.argv) > 4 else None, "XML", allow_absolute=True
        )
        show_statistics_and_lists(lines, all_tests_file, xml_file)

    else:
        print(f"Error: Unknown command: {command}", file=sys.stderr)
        print("Valid commands: extract-errors, filter-log, show-summary", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
