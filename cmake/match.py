#!/usr/bin/env python

# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# check if all lines match in a file
# lines in a match file can contain regex inside of double curly braces {{}}

import sys
import re

if len(sys.argv) != 3:
    print("Usage: python match.py <input_file> <match_file>")
    sys.exit(1)

input_file = sys.argv[1]
match_file = sys.argv[2]

with open(input_file, 'r') as input, open(match_file, 'r') as match:
    input_lines = input.readlines()
    match_lines = match.readlines()

if len(match_lines) < len(input_lines):
    sys.exit(f"Match length < input length (input: {len(input_lines)}, match: {len(match_lines)})")

input_idx = 0
opt = "{{OPT}}"
for i, match_line in enumerate(match_lines):
    if match_line.startswith(opt):
        optional_line = True
        match_line = match_line.removeprefix(opt)
    else:
        optional_line = False

    # split into parts at {{ }}
    match_parts = re.split(r'\{{(.*?)\}}', match_line.strip())
    pattern = ""
    for j, part in enumerate(match_parts):
        if j % 2 == 0:
            pattern += re.escape(part)
        else:
            pattern += part

    # empty input file or end of input file, from now on match file must be optional
    if not input_lines:
        if optional_line is True:
            continue
        else:
            print("End of input file or empty file.")
            print("expected: " + match_line.strip())
            sys.exit(1)

    input_line = input_lines[input_idx].strip()
    if not re.fullmatch(pattern, input_line):
        if optional_line is True:
            continue
        else:
            print("Line " + str(i+1) + " does not match")
            print("is:       " + input_line)
            print("expected: " + match_line.strip())
            print("--- Input Lines " + "-" * 64)
            print("".join(input_lines).strip())
            print("--- Match Lines " + "-" * 64)
            print("".join(match_lines).strip())
            print("-" * 80)
            sys.exit(1)
    else:
        if (input_idx == len(input_lines) - 1):
            input_lines = []
        else:
            input_idx += 1
