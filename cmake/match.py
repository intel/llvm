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

if len(match_lines) != len(input_lines):
    sys.exit(f"Line count doesn't match (is: {len(input_lines)}, expected: {len(match_lines)})")

for i, match_line in enumerate(match_lines):
    # split into parts at {{ }}
    match_parts = re.split(r'\{{(.*?)\}}', match_line.strip())
    pattern = ""
    for j, part in enumerate(match_parts):
        if j % 2 == 0:
            pattern += re.escape(part)
        else:
            pattern += part

    input_line = input_lines[i].strip()
    if not re.fullmatch(pattern, input_line):
        print(f"Line {i+1} does not match".format(i+1))
        print("is:       " + input_line)
        print("expected: " + match_line.strip())
        sys.exit(1)
