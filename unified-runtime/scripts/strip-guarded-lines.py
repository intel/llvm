#!/usr/bin/env python3

# Copyright (C) 2025 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""
This script is a basic pre-processor which deals with conditionally excluding
blocks of lines based on pre and post guard marker lines, a list of guard
names to include in the output, and an input file.

- Pre and post guard marker lines will always be removed even when those guard
  names have not been specified for inclusion.
- Lines within guard blocks which are not specified for inclusion are always
  removed.
- Lines within guard blocks which are specified for inclusion will be always be
  included in the output file.
- All other lines not within guard blocks are always included in the output
  file.
"""

from argparse import ArgumentParser, FileType, RawDescriptionHelpFormatter
import re
from sys import stdout
from typing import List, Tuple, Union


def _create_guards(pre: str, post: str, names: List[str]) -> List[Tuple[str, str]]:
    guards = []
    for name in names:
        guards.append(
            (
                pre % name if "%s" in pre else pre,
                post % name if "%s" in post else post,
            )
        )
    return guards


def _is_guard(marker: str, line: str) -> bool:
    line = line.strip()
    marker = marker.replace("%s", r"[A-Za-z0-9][A-Za-z0-9_]+")
    if re.match(marker, line):
        return True
    return False


def _find_guard(
    line: str, guards: List[Tuple[str, str]]
) -> Union[Tuple[str, str], None]:
    line = line.strip()
    for guard in guards:
        if guard[0] in line or guard[1] in line:
            return guard
    return None


def strip_guarded_lines(
    inlines: List[str],
    pre: str,
    post: str,
    names: List[str],
) -> List[str]:
    guards = _create_guards(pre, post, names)
    stack = []
    outlines = []
    for line in inlines:
        if _is_guard(pre, line):
            stack.append(_find_guard(line, guards))
            continue
        elif _is_guard(post, line):
            guard = stack.pop()
            if guard:
                assert _is_guard(
                    guard[1], line
                ), f'interleaved guard found: "{guard[1]}" before "{line.strip()}"'
            continue
        else:
            if not all(stack):
                continue
        outlines.append(line)
    return outlines


def main():
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "infile", type=FileType("r"), help="input file to strip guarded lines from"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-o",
        "--outfile",
        type=FileType("w"),
        default=stdout,
        help="file to write to stripped output to, default: stdout",
    )
    group.add_argument(
        "-i", "--in-place", action="store_true", help="write to input file in-place"
    )
    parser.add_argument("--encoding", help="encoding to be used for the outfile")
    parser.add_argument(
        "--pre",
        default="#if %s",
        help='pre-guard marker where %%s is the guard name, default: "#if %%s"',
    )
    parser.add_argument(
        "--post",
        default="#endif",
        help='post-guard market where %%s is the guard name, default: "#endif"',
    )
    parser.add_argument("guards", nargs="*", help="names of guards to strip lines of")
    args = parser.parse_args()

    inlines = args.infile.readlines()
    if args.in_place:
        args.infile.close()
        args.outfile = open(args.infile.name, "w")
    if args.encoding:
        args.outfile.reconfigure(encoding=args.encoding)

    outlines = strip_guarded_lines(inlines, args.pre, args.post, args.guards)
    args.outfile.writelines(outlines)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
