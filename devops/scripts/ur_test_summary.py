#!/usr/bin/env python3
"""
Unified Runtime test summary processing for GitHub Actions CI.

This script processes LIT test output logs to:
- Extract error details from failures/timeouts
- Show statistics and collapsed test lists for GitHub Actions
"""

import sys
import re
from typing import List, Dict
from pathlib import Path


def read_log_file(log_path: str) -> List[str]:
    """Read log file and return lines."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
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

    # Test list headers that mark the end of error details
    list_headers_pattern = re.compile(
        r"^(Passed|Unsupported|Failed|Expectedly Failed|"
        r"Timed Out|Unexpectedly Passed|Unresolved) Tests \("
    )

    for line in lines:
        # Start capturing on FAIL/TIMEOUT
        if re.match(r"^(FAIL|TIMEOUT):", line):
            in_error = True

        # Stop at test list headers
        if in_error and list_headers_pattern.match(line):
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
    stats_pattern = re.compile(
        r"^\s*(Total Discovered|Expected Passes|Expectedly Failed|"
        r"Unsupported|Skipped|Passed|Failed|Timed Out|"
        r"Unexpectedly Passed|Unresolved)(\s+Tests)?\s*:"
    )

    result = []
    for line in lines:
        if stats_pattern.match(line):
            result.append(line)

    return result


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
    Extract unsupported test names from LIT inline output.

    During test execution, LIT prints:
    UNSUPPORTED: TestName (reason)

    This is used for GoogleTest format where LIT doesn't generate
    "Unsupported Tests (N):" summary list, but shows "Skipped" in statistics.

    Returns list of unsupported test names (deduplicated).
    """
    unsupported = []
    seen = set()
    # Pattern: "UNSUPPORTED: test_name" or "UNSUPPORTED: test_name (reason)"
    unsupported_pattern = re.compile(r"^UNSUPPORTED:\s+(.+?)(?:\s+\(|$)")

    for line in lines:
        match = unsupported_pattern.match(line)
        if match:
            test_name = match.group(1).strip()
            if test_name not in seen:
                seen.add(test_name)
                unsupported.append(test_name)

    return unsupported


def extract_test_lists(lines: List[str]) -> Dict[str, List[str]]:
    """
    Extract categorized test lists from LIT summary.

    Looks for sections like:
        Passed Tests (N):
          test name 1
          test name 2

    Returns dictionary with category names as keys and test lists as values.
    """
    categories = {}
    current_category = None
    current_tests = []

    # Pattern: "Category Tests (N):"
    category_pattern = re.compile(
        r"^(Passed|Unsupported|Failed|Expectedly Failed|Unresolved|Timed Out) Tests \((\d+)\):"
    )

    for line in lines:
        # Check for category header
        match = category_pattern.match(line)
        if match:
            # Save previous category
            if current_category:
                categories[current_category] = current_tests

            # Start new category
            current_category = match.group(1)
            current_tests = []
            continue

        # If we're in a category
        if current_category:
            # Empty line ends the category
            if not line.strip():
                categories[current_category] = current_tests
                current_category = None
                current_tests = []
            else:
                # Add test to current category (strip leading whitespace)
                test_name = line.strip()
                if test_name:
                    current_tests.append(test_name)

    # Save last category
    if current_category:
        categories[current_category] = current_tests

    return categories


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

    with open(all_tests_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Suite line ends with '.'
            if line.endswith(".") and not line.startswith(" "):
                current_suite = line
            # Test name (indented)
            elif line.startswith("  ") and current_suite:
                # Remove leading whitespace and comments
                test_name = line.strip().split(" #")[0].strip()
                full_name = f"{current_suite}{test_name}"
                tests.append(full_name)

    return tests


def show_statistics_and_lists(lines: List[str], all_tests_file: str = None) -> None:
    """
    Extract categorized test lists from LIT summary.

    Returns dict like:
    {
        'Passed': ['test1', 'test2', ...],
        'Failed': ['test3'],
        ...
    }
    """
    categories = {}
    current_category = None
    current_tests = []

    # Pattern for category headers: "Passed Tests (123):"
    # Note: LIT does not generate "Skipped Tests" - only Unsupported and Expectedly Failed
    header_pattern = re.compile(
        r"^(Passed|Unsupported|Failed|Expectedly Failed|"
        r"Timed Out|Unexpectedly Passed|Unresolved) Tests \((\d+)\):"
    )

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save previous category
            if current_category:
                categories[current_category] = current_tests

            # Start new category
            current_category = match.group(1)
            current_tests = []
        elif current_category:
            # Empty line ends the category
            if not line.strip():
                categories[current_category] = current_tests
                current_category = None
                current_tests = []
            else:
                # Add test to current category (strip leading whitespace)
                test_name = line.strip()
                if test_name:
                    current_tests.append(test_name)

    # Save last category
    if current_category:
        categories[current_category] = current_tests

    return categories


def show_statistics_and_lists(lines: List[str], all_tests_file: str = None) -> None:
    """
    Show test statistics and collapsed test lists for GitHub Actions.

    Output format:
    - Statistics section (always visible)
    - Collapsed sections for each test category
    """
    # Extract statistics
    stats = extract_statistics(lines)
    if stats:
        print("=== Test Statistics ===")
        for stat in stats:
            print(stat.rstrip())
        print()

    # Extract test lists in collapsed sections (LIT format)
    test_lists = extract_test_lists(lines)

    # For GoogleTest format: try to generate skipped list from all_tests_file
    if "Unsupported" not in test_lists and all_tests_file:
        all_tests = parse_gtest_list(all_tests_file)
        
        # Extract test names from ALL categories (not just Passed)
        known_tests = set()
        for category, tests in test_lists.items():
            # For each test in this category, extract the GoogleTest name
            for test in tests:
                parts = test.split("/")
                # Find test class (typically starts with 'ur' or ends with 'Test')
                for i, part in enumerate(parts):
                    if ("Test" in part or part.startswith("ur")) and i + 1 < len(parts):
                        test_class = part
                        test_case = parts[i + 1]
                        gtest_name = f"{test_class}.{test_case}"
                        # Add backend parameter if present
                        if i + 2 < len(parts) and parts[i + 2].startswith("UR_BACKEND"):
                            gtest_name += f"/{parts[i + 2]}"
                        known_tests.add(gtest_name)
                        break

        if all_tests:
            # Find skipped: all_tests - all_known_tests
            skipped = [t for t in all_tests if t not in known_tests]

            if skipped:
                count = len(skipped)
                print(f"::group::Skipped Tests ({count})")
                for test in skipped:
                    print(test)
                print("::endgroup::")

    # For GoogleTest format without all_tests_file: extract from inline output
    elif "Unsupported" not in test_lists:
        # Try GoogleTest SKIPPED first (for test fixtures that use GTEST_SKIP())
        skipped_tests = extract_skipped_from_gtest(lines)
        if skipped_tests:
            count = len(skipped_tests)
            print(f"::group::Skipped Tests ({count})")
            for test in skipped_tests:
                print(test)
            print("::endgroup::")

        # Then extract LIT UNSUPPORTED (for tests with REQUIRES:/UNSUPPORTED: markers)
        unsupported_tests = extract_unsupported_from_lit_inline(lines)
        if unsupported_tests:
            count = len(unsupported_tests)
            print(f"::group::Skipped Tests ({count})")
            for test in unsupported_tests:
                print(test)
            print("::endgroup::")

        # If no skipped tests found but GoogleTest format detected,
        # show informational message
        if not skipped_tests and not unsupported_tests:
            # Check if statistics mention skipped tests
            for stat in stats:
                if "Skipped:" in stat or "Unsupported:" in stat:
                    print(
                        "ℹ️  Skipped test list not available (GoogleTest format limitation)"
                    )
                    print(
                        "   LIT does not generate detailed skip list for GoogleTest binaries."
                    )
                    print()
                    break

    # Show remaining test categories from LIT format
    for category, tests in test_lists.items():
        count = len(tests)
        if count > 0:
            print(f"::group::{category} Tests ({count})")
            for test in tests:
                print(test)
            print("::endgroup::")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print(
            "Usage: ur_test_summary.py <command> <log_file> [all_tests_file]",
            file=sys.stderr,
        )
        print("\nCommands:", file=sys.stderr)
        print(
            "  extract-errors <log>  - Extract FAIL/TIMEOUT error details",
            file=sys.stderr,
        )
        print(
            "  show-summary <log> [all_tests]  - Show statistics and collapsed test lists",
            file=sys.stderr,
        )
        sys.exit(1)

    command = sys.argv[1]

    # All commands require log file
    if len(sys.argv) < 3:
        print(
            f"Error: Command '{command}' requires a log file argument",
            file=sys.stderr,
        )
        sys.exit(1)

    log_file = sys.argv[2]

    # Security: Validate path to prevent traversal attacks
    if ".." in log_file or log_file.startswith("/"):
        print(
            f"Error: Invalid log file path (absolute paths and '..' not allowed): {log_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    lines = read_log_file(log_file)

    if command == "extract-errors":
        result = extract_error_details(lines)
        for line in result:
            print(line, end="")

    elif command == "show-summary":
        # Optional: all tests file for GoogleTest format
        all_tests_file = sys.argv[3] if len(sys.argv) > 3 else None
        if all_tests_file and (
            ".." in all_tests_file or all_tests_file.startswith("/")
        ):
            print(
                f"Error: Invalid all_tests file path: {all_tests_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        show_statistics_and_lists(lines, all_tests_file)

    else:
        print(f"Error: Unknown command: {command}", file=sys.stderr)
        print("Valid commands: extract-errors, show-summary", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
