#! /usr/bin/env python3
"""
 Copyright (C) 2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
# Printing conformance test output from gtest and checking failed tests with match files.
# The match files contain tests that are expected to fail.

import sys
from argparse import ArgumentParser
import subprocess  # nosec B404
import signal
import re

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--test_command", help="Ctest test case")
    parser.add_argument("--test_devices_count", type=str, help="Number of devices on which tests will be run")
    parser.add_argument("--test_platforms_count", type=str, help="Number of platforms on which tests will be run")
    args = parser.parse_args()
    
    result = subprocess.Popen([args.test_command, '--gtest_brief=1', f'--devices_count={args.test_devices_count}',
                               f'--platforms_count={args.test_platforms_count}'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # nosec B603

    pat = re.compile(r'\[( )*FAILED( )*\]')
    output_list = []
    for line in result.stdout:
        output_list.append(line)
        if pat.search(line):
            test_case = line.split(" ")[5]
            test_case = test_case.rstrip(',')
            print(test_case)

    rc = result.wait()
    if rc < 0:
        print(signal.strsignal(abs(rc)))

    print("#### GTEST_OUTPUT ####", file=sys.stderr)
    for output in output_list:
        print(output, file=sys.stderr)
    print("#### GTEST_OUTPUT_END ####", file=sys.stderr)
