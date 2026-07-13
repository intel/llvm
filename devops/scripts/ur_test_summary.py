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

# Use defusedxml for secure XML parsing to prevent XML attacks
# This is required - install via: pip install defusedxml
import defusedxml.ElementTree as ET


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
        r"Excluded|Unsupported|Skipped|Passed|Passed With Retry|"
        r"Failed|Timed Out|Unexpectedly Passed|Unresolved)(\s+Tests)?\s*:"
    )

    result = []
    for line in lines:
        if stats_pattern.match(line):
            result.append(line)

    return result


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

    Returns dictionary with 'slowest' and 'histogram' keys containing line lists.
    """
    result = {"slowest": [], "histogram": []}
    current_section = None
    skip_next_hr = False

    for line in lines:
        # Detect section starts
        if line.strip() == "Slowest Tests:":
            current_section = "slowest"
            skip_next_hr = True
            continue
        elif line.strip() == "Tests Times:" or line.strip() == "Test Times:":
            current_section = "histogram"
            skip_next_hr = True
            continue

        # Skip horizontal rule after section header
        if skip_next_hr and line.strip().startswith("---"):
            skip_next_hr = False
            continue

        # Collect lines for current section
        if current_section == "slowest":
            # Slowest section ends at empty line or next section
            if not line.strip():
                current_section = None
            elif not line.strip().startswith("---"):
                result["slowest"].append(line.rstrip())
        elif current_section == "histogram":
            # Histogram section continues until empty line
            if not line.strip():
                current_section = None
            else:
                result["histogram"].append(line.rstrip())

    return result


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
    if not xml_path:
        return []

    if not Path(xml_path).exists():
        print(f"Note: XML file not found: {xml_path}", file=sys.stderr)
        return []

    skipped = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate through all testsuites and testcases
        for testsuite in root.findall(".//testsuite"):
            for testcase in testsuite.findall("testcase"):
                # Check if testcase has <skipped> child element
                skipped_elem = testcase.find("skipped")
                if skipped_elem is not None:
                    # Exclude tests with "Test not selected" - those are excluded, not skipped
                    message = skipped_elem.get("message", "")
                    if "Test not selected" in message:
                        continue

                    classname = testcase.get("classname", "")
                    name = testcase.get("name", "")

                    # Format: classname.name (match GoogleTest format)
                    if classname and name:
                        full_name = f"{classname}.{name}"
                        skipped.append(full_name)
                    elif name:
                        skipped.append(name)

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML file {xml_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error reading XML file {xml_path}: {e}", file=sys.stderr)

    if xml_path and Path(xml_path).exists():
        print(f"Note: Found {len(skipped)} skipped tests in XML file", file=sys.stderr)

    return skipped


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
    if not xml_path:
        return []

    if not Path(xml_path).exists():
        print(f"Note: XML file not found: {xml_path}", file=sys.stderr)
        return []

    excluded = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate through all testsuites and testcases
        for testsuite in root.findall(".//testsuite"):
            for testcase in testsuite.findall("testcase"):
                # Check if testcase has <skipped> child with "Test not selected" message
                skipped_elem = testcase.find("skipped")
                if skipped_elem is not None:
                    message = skipped_elem.get("message", "")
                    if "Test not selected" in message:
                        classname = testcase.get("classname", "")
                        name = testcase.get("name", "")

                        # Format: classname.name (match GoogleTest format)
                        if classname and name:
                            full_name = f"{classname}.{name}"
                            excluded.append(full_name)
                        elif name:
                            excluded.append(name)

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML file {xml_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error reading XML file {xml_path}: {e}", file=sys.stderr)

    if xml_path and Path(xml_path).exists():
        print(
            f"Note: Found {len(excluded)} excluded tests in XML file", file=sys.stderr
        )

    return excluded


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
    # Pattern: "UNSUPPORTED: test_name" or "SKIP: test_name"
    unsupported_pattern = re.compile(r"^(UNSUPPORTED|SKIP):\s+(.+?)(?:\s+\(|$)")

    for line in lines:
        match = unsupported_pattern.match(line)
        if match:
            test_name = match.group(2).strip()
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
    # Match any word followed by "Tests (N):" to catch unknown categories
    category_pattern = re.compile(r"^([A-Za-z ]+) Tests \((\d+)\):")

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


def show_statistics_and_lists(
    lines: List[str], all_tests_file: str = None, xml_file: str = None
) -> None:
    """
    Show test statistics and collapsed test lists for GitHub Actions.

    Args:
        lines: Log file lines
        all_tests_file: Optional file with --gtest_list_tests output
        xml_file: Optional LIT xunit XML output file

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

    # Validate totals match sum of categories
    total_discovered = None
    for stat in stats:
        if "Total Discovered" in stat:
            match = re.search(r"(\d+)", stat)
            if match:
                total_discovered = int(match.group(1))
                break

    # Extract test lists in collapsed sections (LIT format)
    test_lists = extract_test_lists(lines)

    # Track if we display skipped/excluded separately (for validation later)
    displayed_skipped_count = 0
    displayed_excluded_count = 0
    stats_skipped_count = None
    stats_excluded_count = None

    # For GoogleTest format: try multiple strategies to get skipped list
    # BUT: only if they're not already in test_lists (avoid duplicates)
    if "Unsupported" not in test_lists and "Skipped" not in test_lists:
        # Strategy 1: Extract from LIT xunit XML output (most complete and reliable)
        skipped_xml = extract_skipped_from_xml(xml_file)

        # Strategy 2: Extract from LIT inline output (works if LIT prints them)
        unsupported_inline = extract_unsupported_from_lit_inline(lines)

        # Strategy 3: Extract from GoogleTest SKIPPED summary (rare, only if test uses GTEST_SKIP())
        skipped_gtest = extract_skipped_from_gtest(lines)

        # Strategy 4: Compute from all_tests_file if provided
        skipped_computed = []
        if all_tests_file:
            all_tests = parse_gtest_list(all_tests_file)

            # Extract test names from ALL categories (Passed, Failed, Timed Out, etc.)
            known_tests = set()
            for category, tests in test_lists.items():
                for test in tests:
                    # Split by ' :: ' first to separate suite name from test path
                    if " :: " in test:
                        test = test.split(" :: ", 1)[1]

                    parts = test.split("/")

                    # Find binary name (ends with '-test') to locate TestClass/TestCase
                    binary_index = -1
                    for i, part in enumerate(parts):
                        if part.endswith("-test"):
                            binary_index = i
                            break

                    # After binary: TestClass/TestCase[/Params...]
                    if binary_index >= 0 and binary_index + 2 < len(parts):
                        test_class = parts[binary_index + 1]
                        test_case = parts[binary_index + 2]
                        gtest_name = f"{test_class}.{test_case}"

                        # Add ALL parameters after test case
                        if binary_index + 3 < len(parts):
                            params = "/".join(parts[binary_index + 3 :])
                            gtest_name += f"/{params}"

                        known_tests.add(gtest_name)

            if all_tests:
                # Find skipped: all_tests - all_known_tests
                skipped_computed = [t for t in all_tests if t not in known_tests]

        # Use the most complete list available
        # Prefer XML (most complete) > inline > computed > gtest summary
        if skipped_xml:
            skipped_list = skipped_xml
            source = "LIT xunit XML"
        elif unsupported_inline:
            skipped_list = unsupported_inline
            source = "inline UNSUPPORTED"
        elif skipped_computed:
            skipped_list = skipped_computed
            source = "computed from --gtest_list_tests"
        elif skipped_gtest:
            skipped_list = skipped_gtest
            source = "GoogleTest SKIPPED summary"
        else:
            skipped_list = []
            source = None

        # Extract skipped count from statistics (needed for fallback message)
        for stat in stats:
            if "Skipped:" in stat:
                match = re.search(r"(\d+)", stat)
                if match:
                    stats_skipped_count = int(match.group(1))
                    break

        if skipped_list:
            count = len(skipped_list)
            displayed_skipped_count = count

            print(f"::group::Skipped Tests ({count})")

            # If there's a mismatch, note it
            if stats_skipped_count and abs(stats_skipped_count - count) > 10:
                diff = stats_skipped_count - count
                print(
                    f"Note: Statistics show {stats_skipped_count} skipped, list contains {count} ({source})."
                )
                if diff > 0:
                    print(
                        f"Missing {diff} tests - possible causes: test discovery limitations, filtering, or device-specific tests."
                    )
                print()

            for test in skipped_list:
                print(test)
            print("::endgroup::")
        elif stats_skipped_count:
            # Statistics show skipped but we couldn't extract the list
            print(f"::group::Skipped Tests ({stats_skipped_count})")
            print(f"Warning: Could not extract individual skipped test names.")
            print(
                f"Statistics show {stats_skipped_count} skipped tests, but they are not available in the output."
            )
            print("::endgroup::")

    # Handle Excluded tests similarly
    # BUT: only if they're not already in test_lists (avoid duplicates)
    if "Excluded" not in test_lists:
        # Extract from LIT xunit XML output
        excluded_xml = extract_excluded_from_xml(xml_file)

        # Extract excluded count from statistics
        for stat in stats:
            if "Excluded:" in stat:
                match = re.search(r"(\d+)", stat)
                if match:
                    stats_excluded_count = int(match.group(1))
                    break

        if excluded_xml:
            count = len(excluded_xml)
            displayed_excluded_count = count

            print(f"::group::Excluded Tests ({count})")
            for test in excluded_xml:
                print(test)
            print("::endgroup::")
        elif stats_excluded_count:
            # Statistics show excluded but we couldn't extract the list
            print(f"::group::Excluded Tests ({stats_excluded_count})")
            print(f"Warning: Could not extract individual excluded test names.")
            print(
                f"Statistics show {stats_excluded_count} excluded tests, but they are not available in the output."
            )
            print("::endgroup::")

    # Show remaining test categories from LIT format
    for category, tests in test_lists.items():
        count = len(tests)
        if count > 0:
            print(f"::group::{category} Tests ({count})")
            for test in tests:
                print(test)
            print("::endgroup::")

    # Extract and display test timing information if available
    time_info = extract_time_summary(lines)
    if time_info["slowest"] or time_info["histogram"]:
        print("::group::⏱️ Test Timing Summary")

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

    # Validate that total discovered matches sum of all categories
    if total_discovered:
        # Sum from test_lists + any skipped/excluded we displayed separately
        # BUT: Don't double-count if they're already in test_lists
        sum_categories = sum(len(tests) for tests in test_lists.values())

        # Only add displayed counts if we displayed them separately
        if displayed_skipped_count > 0 and "Skipped" not in test_lists:
            sum_categories += displayed_skipped_count
        if displayed_excluded_count > 0 and "Excluded" not in test_lists:
            sum_categories += displayed_excluded_count

        if total_discovered != sum_categories:
            print()
            print(
                f"::warning::Test count mismatch: Total Discovered = {total_discovered}, but sum of all categories = {sum_categories}"
            )
            print(
                f"Warning: {total_discovered - sum_categories} tests are unaccounted for."
            )
            print(
                f"This may indicate tests in unexpected categories or parsing issues."
            )
            print(f"Categories found: {', '.join(test_lists.keys())}")
            if displayed_skipped_count > 0:
                print(
                    f"(Plus {displayed_skipped_count} skipped tests displayed separately)"
                )
            if displayed_excluded_count > 0:
                print(
                    f"(Plus {displayed_excluded_count} excluded tests displayed separately)"
                )
            print()


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print(
            "Usage: ur_test_summary.py <command> <log_file> [all_tests_file] [xml_file]",
            file=sys.stderr,
        )
        print("\nCommands:", file=sys.stderr)
        print(
            "  extract-errors <log>  - Extract FAIL/TIMEOUT error details",
            file=sys.stderr,
        )
        print(
            "  show-summary <log> [all_tests] [xml]  - Show statistics and collapsed test lists",
            file=sys.stderr,
        )
        print(
            "\nArguments:",
            file=sys.stderr,
        )
        print(
            "  log          - LIT test output log file",
            file=sys.stderr,
        )
        print(
            "  all_tests    - Optional: output from --gtest_list_tests (for computed skipped)",
            file=sys.stderr,
        )
        print(
            "  xml          - Optional: LIT xunit XML output (--xunit-xml-output)",
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
        # Empty string means not provided (from bash "")
        if all_tests_file == "":
            all_tests_file = None
        if all_tests_file and (
            ".." in all_tests_file or all_tests_file.startswith("/")
        ):
            print(
                f"Error: Invalid all_tests file path: {all_tests_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Optional: XML file from LIT xunit output
        xml_file = sys.argv[4] if len(sys.argv) > 4 else None
        # Empty string means not provided (from bash "")
        if xml_file == "":
            xml_file = None
        # Validate: reject path traversal but allow absolute paths (BUILD_DIR can be absolute)
        if xml_file and ".." in xml_file:
            print(
                f"Error: Invalid XML file path (path traversal): {xml_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        show_statistics_and_lists(lines, all_tests_file, xml_file)

    else:
        print(f"Error: Unknown command: {command}", file=sys.stderr)
        print("Valid commands: extract-errors, show-summary", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
