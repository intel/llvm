#!/usr/bin/env python
"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from sys import argv
from subprocess import PIPE, Popen, STDOUT
import argparse
import os


def run(command, env):
    process = Popen(command, env=env, stdout=PIPE,
                    stderr=STDOUT, cwd=command[1])
    lines = process.communicate()[0].decode('utf-8').splitlines()
    results = {"Passed": {}, "Skipped": {}, "Failed": {}, 'Total': 0, 'Success': True}
    for line in lines:
        result_types = ['[       OK ]', '[  FAILED  ]', '[  SKIPPED ]']
        if any([x in line for x in result_types]) and line.endswith("ms)"):
            name, time = line[line.find(']') + 2:].split(' ', maxsplit=1)
            if 'OK' in line:
                results['Passed'][name] = {'time': time}
            elif 'SKIPPED' in line:
                results['Skipped'][name] = {'time': time}
            elif 'FAILED' in line:
                results['Failed'][name] = {'time': time}
        elif '[==========] Running' in line:
            # This is the start of a new test suite, get the number of tests in
            # the first line e.g: '[==========] Running 29 tests from 9 test suites.'
            total = line[line.find('g') + 2:line.find('t') - 1]
            results['Total'] += int(total)

    if process.returncode != 0:
        results['Success'] = False    

    return results


def print_results(results, result_type):
    print('[CTest Parser] {} tests: '.format(result_type))
    print("\n".join("\t{}\t{}".format(k, v['time'])
          for k, v in results.items()))


def print_summary(total, total_passed, total_failed, total_skipped, total_crashed):
    pass_rate_incl_skipped = str(round((total_passed / total) * 100, 2))
    total_excl_skipped = total - total_skipped
    pass_rate_excl_skipped = str(
        round((total_passed / total_excl_skipped) * 100, 2))

    skipped_rate = str(round((total_skipped / total) * 100, 2))
    failed_rate = str(round((total_failed / total) * 100, 2))
    crashed_rate = str(round((total_crashed / total) * 100, 2))

    print('[CTest Parser] Results:')
    print('\tTotal\t[{}]'. format(total))
    print('\tPassed\t[{}]\t({}%) - ({}% with skipped tests excluded)'.format(
        total_passed, pass_rate_incl_skipped, pass_rate_excl_skipped))
    print('\tSkipped\t[{}]\t({}%)'.format(total_skipped, skipped_rate))
    print('\tFailed\t[{}]\t({}%)'.format(total_failed, failed_rate))
    print('\tCrashed\t[{}]\t({}%)'.format(total_crashed, crashed_rate))


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
    parser = argparse.ArgumentParser(
        description='CTest Result Parser. Parses output from CTest and '
        'summarises test results. -VV argument is  always passed to '
        'CTest capture full output.')

    parser.add_argument('ctest_path', type=dir_path, nargs='?', default='.',
                        help='Optional path to test directory containing '
                        'CTestTestfile. Defaults to current directory.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Output only failed tests.')

    args = parser.parse_args()

    path = args.ctest_path
    command = ['ctest', path, '-VV']

    env = os.environ.copy()
    env['GTEST_COLOR'] = 'no'
    env['CTEST_OUTPUT_ON_FAILURE'] = '0'
    env['GTEST_BRIEF'] = '0' 
    env['GTEST_PRINT_TIME'] = '1' 
    env['GTEST_PRINT_UTF8'] = '1' 
      
    results = run(command, env)

    total = results['Total']
    total_passed = len(results['Passed'])
    total_skipped = len(results['Skipped'])
    total_failed = len(results['Failed'])
    total_crashed = total - (total_passed + total_skipped + total_failed)

    if total > 0:
        print("[CTest Parser] Preparing results...")
        if args.quiet == False:
            if total_passed > 0:
                print_results(results['Passed'], 'Passed')
            if total_skipped > 0:
                print_results(results['Skipped'], 'Skipped')
        if total_failed > 0:
            print_results(results['Failed'], 'Failed')
        
        print_summary(total, total_passed, total_failed,
                      total_skipped, total_crashed)
        if results['Success'] == False:
            exit(1)
    else:
        print("[CTest Parser] Error: no tests were run")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
