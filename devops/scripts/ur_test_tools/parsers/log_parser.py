"""Parse LIT text output for test information."""

import sys
from pathlib import Path
from typing import Dict, List

from ..constants import (
    FAIL_TIMEOUT_PATTERN,
    STAT_LINE_PATTERN,
    TEST_CATEGORY_PATTERN,
    TESTING_TIME_PATTERN,
    SLOWEST_TESTS_HEADER,
    TEST_TIMES_HEADERS,
)
from .parser_models import (
    ParsedLogData,
    ParsedTestObservation,
    LIT_CATEGORY_TO_STATUS,
)
from .stats_parser import parse_statistics


def _read_with_utf8_fallback(path: str, read_func):
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return read_func(f)
    except UnicodeDecodeError:
        print(
            f"Warning: File contains non-UTF-8 characters, replacing with U+FFFD",
            file=sys.stderr,
        )
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return read_func(f)


def read_log_file(log_path: str) -> List[str]:
    path = Path(log_path)
    file_size = path.stat().st_size

    if file_size > 10 * 1024 * 1024:  # 10 MB
        print(
            f"Large log file: {file_size / (1024 * 1024):.1f} MB. "
            f"This may indicate a test problem.",
            file=sys.stderr,
        )

    try:
        return _read_with_utf8_fallback(log_path, lambda f: f.readlines())
    except OSError as e:
        raise OSError(f"Cannot read log file: {e}") from e


class LITLogParser:
    """Parse LIT (llvm-lit) text output for test information."""

    def __init__(self, lines: List[str]):
        self.lines = lines

    def extract_error_details(self) -> List[str]:
        result = []
        in_error = False

        for line in self.lines:
            if FAIL_TIMEOUT_PATTERN.match(line):
                in_error = True

            # Stop at test list headers or timing summaries
            if in_error and (
                TEST_CATEGORY_PATTERN.match(line)
                or line.strip() == SLOWEST_TESTS_HEADER
                or line.strip() in TEST_TIMES_HEADERS
            ):
                break

            if in_error:
                result.append(line)

        return result

    def extract_statistics(self) -> List[str]:
        result = []
        in_summary = False
        for line in self.lines:
            match = STAT_LINE_PATTERN.match(line)
            if not in_summary:
                if match and match.group(1) == "Total Discovered Tests":
                    in_summary = True
                    result.append(line)
            elif match:
                result.append(line)
            elif line.strip():
                break
        return result

    def extract_time_summary(self) -> Dict[str, List[str]]:
        result = {"slowest": [], "histogram": []}
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

    def parse_to_observations(self) -> ParsedLogData:
        observations = []
        declared_counts = {}
        current_status = None

        for line in self.lines:
            match = TEST_CATEGORY_PATTERN.match(line)
            if match:
                category = match.group(1)
                current_status = LIT_CATEGORY_TO_STATUS.get(category)
                if current_status is None:
                    print(
                        f"Warning: Unknown test category '{category}'", file=sys.stderr
                    )
                    continue

                declared_counts[current_status] = declared_counts.get(
                    current_status, 0
                ) + int(match.group(2))
                continue

            if current_status and not line.strip():
                current_status = None
            elif current_status:
                observations.append(
                    ParsedTestObservation(name=line.strip(), status=current_status)
                )

        stats_lines = self.extract_statistics()
        statistics = parse_statistics(stats_lines)

        testing_time_ms = None
        for line in self.lines:
            match = TESTING_TIME_PATTERN.match(line)
            if match:
                testing_time_ms = float(match.group(1)) * 1000.0
                break

        timing = self.extract_time_summary()
        errors = self.extract_error_details()

        return ParsedLogData(
            tests=observations,
            declared_counts=declared_counts,
            statistics=statistics,
            statistics_lines=[line.rstrip() for line in stats_lines],
            error_details=errors,
            slowest_tests=timing["slowest"],
            time_histogram=timing["histogram"],
            testing_time_ms=testing_time_ms,
        )
