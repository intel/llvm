#!/usr/bin/env python3

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
# {{NONDETERMINISTIC}} - order of match rules isn't important - each (non OPT) input line is paired with a match line
# in any order
# Special tags are mutually exclusive and are expected to be located at the start of a line.
#

import os
import sys
import re
from enum import Enum


## @brief print a sequence of lines
def print_lines(lines, hint=None):
    counter = 1
    for l in lines:
        hint_char = " "
        if hint == counter - 1:
            hint_char = ">"
        print("{}{:4d}| {}".format(hint_char, counter, l.strip()))
        counter += 1


## @brief print the whole content of input and match files
def print_content(
    input_lines, match_lines, ignored_lines, hint_input=None, hint_match=None
):
    print("------ Input Lines " + "-" * 61)
    print_lines(input_lines, hint_input)
    print("------ Match Lines " + "-" * 61)
    print_lines(match_lines, hint_match)
    print("------ Ignored Lines " + "-" * 59)
    print_lines(ignored_lines)
    print("-" * 80)


## @brief print the incorrect match line
def print_incorrect_match(match_line, present, expected):
    print("Line " + str(match_line) + " does not match")
    print("is:       " + present)
    print("expected: " + expected)


## @brief print missing match line
def print_input_not_found(input_line, input):
    print("Input line " + str(input_line) + " has no match line")
    print("is:       " + input)


## @brief print missing input line
def print_match_not_found(match_line, input):
    print("Match line " + str(match_line) + " has no input line")
    print("is:       " + input)


## @brief print general syntax error
def print_error(text, match_line):
    print("Line " + str(match_line) + " encountered an error")
    print(text)


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
    OPT = "{{OPT}}"  # makes the line optional
    IGNORE = "{{IGNORE}}"  # ignores all input until next match or end of input file
    NONDETERMINISTIC = "{{NONDETERMINISTIC}}"  # switches on "deterministic mode"
    COMMENT = "#"  # comment - line ignored


## @brief main function for the match file processing script
def main():
    if len(sys.argv) != 3:
        print("Usage: python match.py <input_file> <match_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    match_file = sys.argv[2]

    with open(input_file, "r") as input, open(match_file, "r") as match:
        input_lines = input.readlines()
        # Filter out empty lines and comments (lines beginning with the comment
        # character, ignoring leading whitespace)
        match_lines = list(
            filter(
                lambda line: line.strip()
                and not line.lstrip().startswith(Tag.COMMENT.value),
                match.readlines(),
            )
        )

    ignored_lines = []
    matched_lines = set()

    input_idx = 0
    match_idx = 0
    tags_in_effect = []
    deterministic_mode = False
    while True:
        # check file status
        status = check_status(input_lines[input_idx:], match_lines[match_idx:])
        if deterministic_mode:
            if status == Status.INPUT_END:
                # Convert the list of seen matches to the list of unseen matches
                remaining_matches = set(range(len(match_lines))) - matched_lines
                for m in remaining_matches:
                    line = match_lines[m]
                    if line.startswith(Tag.OPT.value) or line.startswith(
                        Tag.NONDETERMINISTIC.value
                    ):
                        continue
                    print_match_not_found(m + 1, match_lines[m])
                    print_content(input_lines, match_lines, ignored_lines, hint_match=m)
                    sys.exit(1)

                sys.exit(0)
            elif status == Status.MATCH_END:
                print_input_not_found(input_idx + 1, input_lines[input_idx])
                print_content(
                    input_lines, match_lines, ignored_lines, hint_input=input_idx
                )
                sys.exit(1)
        else:
            if (status == Status.INPUT_AND_MATCH_END) or (
                status == Status.MATCH_END and Tag.IGNORE in tags_in_effect
            ):
                # all lines matched or the last line in match file is an ignore tag
                sys.exit(0)
            elif status == Status.MATCH_END:
                print_incorrect_match(input_idx + 1, input_lines[input_idx].strip(), "")
                print_content(
                    input_lines, match_lines, ignored_lines, hint_input=input_idx
                )
                sys.exit(1)
            elif status == Status.INPUT_END:
                # If we get to the end of the input, but still have pending matches,
                # then that's a failure unless all pending matches are optional -
                # otherwise we're done
                while match_idx < len(match_lines):
                    if not (
                        match_lines[match_idx].startswith(Tag.OPT.value)
                        or match_lines[match_idx].startswith(Tag.IGNORE.value)
                        or match_lines[match_idx].startswith(Tag.NONDETERMINISTIC.value)
                    ):
                        print_incorrect_match(match_idx + 1, "", match_lines[match_idx])
                        print_content(
                            input_lines,
                            match_lines,
                            ignored_lines,
                            hint_match=match_idx,
                        )
                        sys.exit(1)
                    match_idx += 1
                sys.exit(0)

        input_line = (
            input_lines[input_idx].strip() if input_idx < len(input_lines) else ""
        )
        match_line = match_lines[match_idx]

        # check for tags
        if match_line.startswith(Tag.OPT.value):
            tags_in_effect.append(Tag.OPT)
            match_line = match_line[len(Tag.OPT.value) :]
        elif (
            match_line.startswith(Tag.NONDETERMINISTIC.value) and not deterministic_mode
        ):
            deterministic_mode = True
            match_idx = 0
            input_idx = 0
            continue
        elif match_line.startswith(Tag.IGNORE.value):
            if deterministic_mode:
                print_error(r"Can't use \{{IGNORE\}} in deterministic mode")
                sys.exit(2)
            tags_in_effect.append(Tag.IGNORE)
            match_idx += 1
            continue  # line with ignore tag should be skipped

        # split into parts at {{ }}
        match_parts = re.split(r"\{{(.*?)\}}", match_line.strip())
        pattern = ""
        for j, part in enumerate(match_parts):
            if j % 2 == 0:
                pattern += re.escape(part)
            else:
                pattern += part

        # match or process tags
        if deterministic_mode:
            if re.fullmatch(pattern, input_line) and match_idx not in matched_lines:
                input_idx += 1
                matched_lines.add(match_idx)
                match_idx = 0
                tags_in_effect = []
            else:
                match_idx += 1
        else:
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
                print_content(
                    input_lines,
                    match_lines,
                    ignored_lines,
                    hint_match=match_idx,
                    hint_input=input_idx,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
