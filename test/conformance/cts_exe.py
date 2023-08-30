#! /usr/bin/env python3
"""
 Copyright (C) 2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import sys
from argparse import ArgumentParser
import subprocess  # nosec B404
import signal
import re

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--test_command", help="Ctest test case")

    args = parser.parse_args()
    result = subprocess.Popen([args.test_command, '--gtest_brief=1'], stdout = subprocess.PIPE, text = True)  # nosec B603

    pat = re.compile(r'\[( )*FAILED( )*\]')
    for line in result.stdout.readlines():
        if pat.search(line):
            test_case = line.split(" ")[5]
            test_case = test_case.rstrip(',')
            print(test_case)

    result.communicate()
    rc = result.returncode
    if rc < 0:
        print(signal.strsignal(abs(result.returncode)))
