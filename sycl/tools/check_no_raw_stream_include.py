#!/usr/bin/env python
"""
Guard for device_only_no_iostream.cpp.

Given a `-MD` dependency file, extract every KHR-reachable SYCL header
(``.../include/sycl/*.hpp``) and fail if any of them contains a *raw*
``#include <iostream|istream|ostream|sstream>``.

This catches the real regression -- an author adding a stream include to a
KHR-reachable SYCL header -- independent of standard-library version. It is a
source grep, not a dependency-list check, so it is not fooled by old libstdc++
dragging <ostream>/<istream> in through <iterator>.

Cross-platform on purpose: the `-MD` list uses '/' on Linux and '\\' on
Windows, and this replaces a POSIX grep/xargs pipeline that does not run on the
Windows lit shell. Exits nonzero (and prints the offenders) on any violation.
"""

import argparse
import re
import sys

# Raw stream includes that must never appear in a KHR-reachable SYCL header.
STREAM_INCLUDE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*<(iostream|istream|ostream|sstream)>"
)


def sycl_headers(dep_file):
    """Yield the SYCL header paths mentioned in a -MD dependency file."""
    with open(dep_file, "r") as f:
        text = f.read()
    # Drop line-continuation backslashes and normalize '\' to '/' so the match
    # works on Windows. The ".hpp$" filter below already excludes the make
    # target ("foo.o:"), so there is no need to strip it explicitly -- doing so
    # would misfire on a Windows drive-letter colon (e.g. "C:/build/foo.o:").
    text = text.replace("\\\n", " ").replace("\\", "/")
    seen = set()
    for tok in text.split():
        if re.search(r"/include/sycl/.*\.hpp$", tok) and tok not in seen:
            seen.add(tok)
            yield tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dep_file", help="the -MD dependency (.d) file")
    args = ap.parse_args()

    violations = []
    for header in sycl_headers(args.dep_file):
        try:
            with open(header, "r") as f:
                for lineno, line in enumerate(f, 1):
                    if STREAM_INCLUDE.match(line):
                        violations.append((header, lineno, line.rstrip()))
        except OSError as e:
            print("warning: could not read %s: %s" % (header, e), file=sys.stderr)

    if violations:
        print("raw stream #include found in KHR-reachable SYCL header(s):")
        for header, lineno, line in violations:
            print("  %s:%d: %s" % (header, lineno, line))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
