#!/usr/bin/env python3

# This script checks if a diff contains changes that should be inspected by
# clang-tidy.

from __future__ import annotations
import re
import sys
from typing import Iterator

EXCLUDE_PATH_RE = re.compile(r"(?:^|/)(test|test-e2e|unittests)(?:/|$)")


def is_relevant_path(path: str) -> bool:
    return EXCLUDE_PATH_RE.search(path) is None


# This iterator splits a diff  file into separate sections like:
# diff --git a/path-to/file b/path-to/file
# ... code changes ...
def iter_diff_sections(diff_content: str) -> Iterator[str]:
    section_lines: list[str] = []
    started = False

    for line in diff_content.splitlines(True):  # keep '\n'
        if line.startswith("diff --git "):
            if started and section_lines:
                yield "".join(section_lines)
                section_lines = []
            started = True

        if started:
            section_lines.append(line)

    if started and section_lines:
        yield "".join(section_lines)


def main() -> int:
    diff_path = sys.argv[1]
    with open(diff_path, "r", encoding="utf-8", errors="replace") as f:
        diff_content = f.read()

    should_run = False
    for section in iter_diff_sections(diff_content):
        lines = section.splitlines()
        # Skip removed files.
        if lines[4] == "+++ /dev/null":
            continue
        result_file = lines[3]
        # Skip non-c++ files.
        if not result_file.endswith((".cpp", ".hpp", ".h")):
            continue
        # Skip tests etc.
        if not is_relevant_path(result_file):
            continue
        # Check if any non-comment string was added.
        for line in lines[4:]:
            if line.startswith("+") and not line[1:].lstrip().startswith("//"):
                should_run = True
                break
        if should_run == True:
            break

    sys.exit(0 if should_run else 1)


if __name__ == "__main__":
    sys.exit(main())
