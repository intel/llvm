#!/usr/bin/env python3
"""
 Copyright (C) 2024 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
# Printing conformance test output from gtest and checking failed tests with match files.
# The match files contain tests that are expected to fail.

import os
import sys
import argparse
import signal
import subprocess  # nosec B404


def _ci():
    return os.environ.get("CI") is not None


def _color():
    return sys.stdout.isatty() or os.environ.get("GTEST_COLOR").lower() == "yes"


def _print_header(header, *args):
    if _ci():
        # GitHub CI interprets this as a "group header" and will provide buttons to fold/unfold it
        print("##[group]{}".format(header.format(*args)))
    elif _color():
        # Inverse color
        print("\033[7m{}\033[27m".format(header.format(*args)))
    else:
        print("### {}".format(header.format(*args)))


def _print_end_header():
    if _ci():
        print("##[endgroup]")


def _print_error(header, *args):
    if _color():
        # "!!!" on a red background
        print("\033[41m!!!\033[0m {}".format(header.format(*args)))
    else:
        print("!!! {}".format(header.format(*args)))


def _print_format(msg, *args):
    print(msg.format(*args))


def _print_environ(env):
    _print_header("Environment")
    for k, v in env.items():
        _print_format("> {} = {}", k, v)
    _print_end_header()


def _print_failure(code, stdout):
    # Display some context in the CI log so the user doesn't have to click down the lines
    if _ci():
        _print_format("stdout/stderr of failure:")
        print(stdout)

    signalname = "???"
    try:
        signalname = signal.strsignal(abs(code))
    except ValueError:
        pass
    _print_format("Got error code {} ({})", code, signalname)


def _check_filter(cmd, filter):
    """
    Checks that the filter matches at least one test for the given cmd
    """
    sys.stdout.flush()
    check = subprocess.Popen(  # nosec B603
        cmd + ["--gtest_list_tests"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=(os.environ | {"GTEST_FILTER": filter}),
    )
    if not check.stdout.read(1):
        return False
    return True


def _run_cmd(cmd, comment, filter):
    _print_header("Running suite for: {}", comment)
    _print_format("### {}", " ".join(cmd))

    # Check tests are found
    if not _check_filter(cmd, filter):
        _print_end_header()
        _print_error("Could not find any tests with this filter")
        return (2, "")

    sys.stdout.flush()
    proc = subprocess.Popen(  # nosec B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=(os.environ | {"GTEST_FILTER": filter}),
    )
    stdout = proc.communicate()[0].decode("utf-8")
    returncode = proc.wait()
    print(stdout)
    _print_end_header()
    return (returncode, stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_command", help="Ctest test case")
    parser.add_argument("--failslist", type=str, help="Failure list")
    parser.add_argument("--", dest="ignored", action="store_true")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.test_command is None or args.failslist is None:
        print("Usage: cts_exe.py --test_command (test binary) --failslist (match file) -- (test arguments)")
        sys.exit(1)

    base_invocation = [args.test_command] + args.rest

    if os.environ.get("GTEST_OUTPUT") is not None:
        # We are being ran purely to generate an output file (likely for ctest_parser.py); falling back to just using
        # one test execution
        sys.exit(
            subprocess.call(  # nosec B603
                base_invocation, stdout=sys.stdout, stderr=sys.stderr
            )
        )

    _print_environ(os.environ)

    # Parse fails list
    _print_format("Loading fails from {}", args.failslist)
    fail_patterns = []
    with open(args.failslist) as f:
        for l in f:
            optional = "{{OPT}}" in l
            l = l.replace("{{OPT}}", "")

            if l.startswith("#"):
                continue
            if l.strip() == "":
                continue

            fail_patterns.append(
                {
                    "pattern": l.strip(),
                    "optional": optional,
                }
            )

    _print_header("Known failing tests")
    for fail in fail_patterns:
        _print_format("> {}", fail)
    _print_end_header()
    if len(fail_patterns) == 0:
        _print_error(
            "Fail list is empty, if there are no more failures, please remove the file"
        )
        sys.exit(2)

    final_result = 0

    # First, run all the known good tests
    gtest_filter = "-" + (":".join(map(lambda x: x["pattern"], fail_patterns)))
    if _check_filter(base_invocation, gtest_filter):
        (result, stdout) = _run_cmd(base_invocation, "known good tests", gtest_filter)
        if result != 0:
            _print_error("Tests we expected to pass have failed")
            _print_failure(result, stdout)
            final_result = result
    else:
        _print_format("Note: No tests in this suite are expected to pass")

    # Then run each known failing tests
    for fail in fail_patterns:
        (result, stdout) = _run_cmd(
            base_invocation, "failing test {}".format(fail["pattern"]), fail["pattern"]
        )

        if result == 0 and not fail["optional"]:
            _print_error(
                "Test {} is passing when we expect it to fail!", fail["pattern"]
            )
            _print_failure(result, stdout)
            final_result = 1

    sys.exit(final_result)
