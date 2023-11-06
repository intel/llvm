#!/usr/bin/env python

# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Check if all input file content matches match file content.
# Lines in a match file can contain regex inside of double curly braces {{}}.
# Regex patterns are limited to single line.
#
# List of available special tags:
# {{OPT}}    - makes content in the same line as the tag optional
# {{IGNORE}} - ignores all content until the next successfully matched line or the end of the input
# Special tags are mutually exclusive and are expected to be located at the start of a line.
#

import os
import sys
import re
from enum import Enum


## @brief print the whole content of input and match files
def print_content(input_lines, match_lines, ignored_lines):
    print("--- Input Lines " + "-" * 64)
    print("".join(input_lines).strip())
    print("--- Match Lines " + "-" * 64)
    print("".join(match_lines).strip())
    print("--- Ignored Lines " + "-" * 62)
    print("".join(ignored_lines).strip())
    print("-" * 80)


## @brief print the incorrect match line
def print_incorrect_match(match_line, present, expected):
    print("Line " + str(match_line) + " does not match")
    print("is:       " + present)
    print("expected: " + expected)


## @brief pattern matching script status values
class Status(Enum):
    INPUT_END = 1
    MATCH_END = 2
    INPUT_AND_MATCH_END = 3
    PROCESSING = 4


## @brief check matching script status
def check_status(input_lines, match_lines):
    if not input_lines and not match_lines:
        return Status.INPUT_AND_MATCH_END
    elif not input_lines:
        return Status.INPUT_END
    elif not match_lines:
        return Status.MATCH_END
    return Status.PROCESSING


## @brief pattern matching tags.
## Tags are expected to be at the start of the line.
class Tag(Enum):
    OPT = "{{OPT}}"         # makes the line optional
    IGNORE = "{{IGNORE}}"   # ignores all input until next match or end of input file


## @brief main function for the match file processing script
def main():
    if len(sys.argv) != 3:
        print("Usage: python match.py <input_file> <match_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    match_file = sys.argv[2]

    with open(input_file, 'r') as input, open(match_file, 'r') as match:
        input_lines = input.readlines()
        match_lines = match.readlines()

    ignored_lines = []

    input_idx = 0
    match_idx = 0
    tags_in_effect = []
    while True:
        # check file status
        status = check_status(input_lines[input_idx:], match_lines[match_idx:])
        if (status == Status.INPUT_AND_MATCH_END) or (status == Status.MATCH_END and Tag.IGNORE in tags_in_effect):
            # all lines matched or the last line in match file is an ignore tag
            sys.exit(0)
        elif status == Status.MATCH_END:
            print_incorrect_match(match_idx + 1, input_lines[input_idx].strip(), "");
            print_content(input_lines, match_lines, ignored_lines)
            sys.exit(1)

        input_line = input_lines[input_idx].strip() if input_idx < len(input_lines) else ""
        match_line = match_lines[match_idx]

        # check for tags
        if match_line.startswith(Tag.OPT.value):
            tags_in_effect.append(Tag.OPT)
            match_line = match_line[len(Tag.OPT.value):]
        elif match_line.startswith(Tag.IGNORE.value):
            tags_in_effect.append(Tag.IGNORE)
            match_idx += 1
            continue # line with ignore tag should be skipped

        # split into parts at {{ }}
        match_parts = re.split(r'\{{(.*?)\}}', match_line.strip())
        pattern = ""
        for j, part in enumerate(match_parts):
            if j % 2 == 0:
                pattern += re.escape(part)
            else:
                pattern += part

        # match or process tags
        if re.fullmatch(pattern, input_line):
            input_idx += 1
            match_idx += 1
            tags_in_effect = []
        elif Tag.OPT in tags_in_effect:
            match_idx += 1
            tags_in_effect.remove(Tag.OPT)
        elif Tag.IGNORE in tags_in_effect:
            ignored_lines.append(input_line + os.linesep)
            input_idx += 1
        else:
            print_incorrect_match(match_idx + 1, input_line, match_line.strip())
            print_content(input_lines, match_lines, ignored_lines)
            sys.exit(1)


if __name__ == "__main__":
    main()
