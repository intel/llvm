#!/usr/bin/env python
"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from subprocess import  Popen, DEVNULL
import argparse
import os
import json

TMP_RESULTS_FILE = "tmp-results-file.json"
CTS_TEST_SUITES = ["context", "device", "enqueue", "event", "kernel", "memory",
                   "platform", "program", "queue", "runtime", "sampler", "usm",
                   "virtual_memory"]

def percent(amount, total):
    return round((amount / total) * 100, 2)

def summarize_results(results):
    total = results['Total']
    total_passed = len(results['Passed'])
    total_skipped = len(results['Skipped'])
    total_failed = len(results['Failed'])

    pass_rate_incl_skipped = percent(total_passed, total)
    pass_rate_excl_skipped = percent(total_passed, total - total_skipped)

    skipped_rate = percent(total_skipped, total)
    failed_rate = percent(total_failed, total)

    ljust_param = len(str(total))
    
    print(
f"""[CTest Parser] Results:
    Total    [{str(total).ljust(ljust_param)}]
    Passed   [{str(total_passed).ljust(ljust_param)}]    ({pass_rate_incl_skipped}%) - ({pass_rate_excl_skipped}% with skipped tests excluded)
    Skipped  [{str(total_skipped).ljust(ljust_param)}]    ({skipped_rate}%)
    Failed   [{str(total_failed).ljust(ljust_param)}]    ({failed_rate}%)
"""
    )

def parse_results(results):
    parsed_results = {"Passed": {}, "Skipped":{}, "Failed": {}, 'Total':0, 'Success':True}
    for _, result in results.items():
        if result is None:
            parsed_results['Success'] = False
            continue

        parsed_results['Total'] += result['tests']
        for testsuite in result.get('testsuites'):
            for test in testsuite.get('testsuite'):
                test_name = f"{testsuite['name']}.{test['name']}"
                test_time = test['time']
                if 'failures' in test:
                    parsed_results['Failed'][test_name] = {'time': test_time}
                elif test['result'] == 'SKIPPED':
                    parsed_results['Skipped'][test_name] = {'time': test_time}
                else:
                    parsed_results['Passed'][test_name] = {'time': test_time}
    return parsed_results


def run(args):
    results = {}

    tmp_results_file = f"{args.ctest_path}/{TMP_RESULTS_FILE}"
    env = os.environ.copy()
    env['GTEST_OUTPUT'] = f"json:{tmp_results_file}"

    for suite in CTS_TEST_SUITES:
        ctest_path = f"{args.ctest_path}/test/conformance/{suite}"
        process = Popen(['ctest',ctest_path], env=env, cwd=ctest_path, 
                        stdout=DEVNULL if args.quiet else None, 
                        stderr=DEVNULL if args.quiet else None)
        process.wait()

        try:
            with open(tmp_results_file, 'r') as results_file:
                json_data = json.load(results_file)
                results[suite] = json_data
            os.remove(tmp_results_file)
        except FileNotFoundError:
            results[suite] = None
            print('\033[91m' + f"Conformance test suite '{suite}' : likely crashed!" + '\033[0m')
    
    return results

def dir_path(string):
    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ctest_path', type=dir_path, nargs='?', default='.',
                        help='Optional path to test directory containing '
                        'CTestTestfile. Defaults to current directory.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Output only failed tests.')
    args = parser.parse_args()

    raw_results = run(args)
    parsed_results = parse_results(raw_results)
    summarize_results(parsed_results)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
