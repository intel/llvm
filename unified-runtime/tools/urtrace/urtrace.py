#!/usr/bin/env python3

# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import subprocess  # nosec B404
import os
import sys


def find_library(paths, name, recursive=False):
    for path in paths:
        for root, _, files in os.walk(path):
            for f in files:
                if f == name:
                    return os.path.abspath(os.path.join(root, name))
            if not recursive:
                break
    return None


def is_filename(string):  # checks whether a string is just a filename or a path
    filename = os.path.basename(string)
    return filename == string


def get_dynamic_library_name(name):
    if sys.platform.startswith("linux"):
        return "lib{}.so".format(name)
    elif sys.platform == "win32":
        return "{}.dll".format(name)
    else:
        sys.exit("Unsupported platform: {}".format(sys.platform))


parser = argparse.ArgumentParser(
    description="""Unified Runtime tracing tool.
    %(prog)s is a program that runs the specified command until its exit,
    intercepting and recording all unified runtime library calls from the executed process.
    It has support for rich printing of function arguments, and can also perform
    rudimentary profiling of unified runtime functions.""",
    epilog="""examples:

    %(prog)s ./myapp --myapp-arg
    %(prog)s --mock --profiling --filter ".*(Device|Platform).*" ./hello_world
    %(prog)s --adapter libur_adapter_cuda.so --begin ./sycl_app""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "command", help="Command to run, including arguments.", nargs=argparse.REMAINDER
)
parser.add_argument(
    "--profiling", help="Measure function execution time.", action="store_true"
)
parser.add_argument(
    "--filter", help="Only trace functions that match the provided regex filter."
)
parser.add_argument(
    "--mock", help="Force the use of the mock adapter.", action="store_true"
)
parser.add_argument(
    "--adapter",
    help="Force the use of the provided adapter.",
    action="append",
    default=[],
)
parser.add_argument(
    "--json", help="Write output in a JSON Trace Event Format.", action="store_true"
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--file", help="Write trace output to a file with the given name instead of stderr."
)
group.add_argument(
    "--stdout",
    help="Write trace output to stdout instead of stderr.",
    action="store_true",
)
parser.add_argument(
    "--no-args",
    help="Don't pretty print traced functions arguments.",
    action="store_true",
)
parser.add_argument(
    "--print-begin", help="Print on function begin.", action="store_true"
)
parser.add_argument(
    "--time-unit",
    choices=["ns", "us", "ms", "s", "auto"],
    default="auto",
    help="Use a specific unit of time for profiling.",
)
parser.add_argument(
    "--libpath",
    default=[".", "../lib/", "/lib/", "/usr/local/lib/", "/usr/lib/"],
    action="append",
    help="Search path for adapters and xpti libraries.",
)
parser.add_argument(
    "--recursive", help="Use recursive library search.", action="store_true"
)
parser.add_argument(
    "--debug", help="Print tool debug information.", action="store_true"
)
parser.add_argument(
    "--flush",
    choices=["debug", "info", "warning", "error"],
    default="error",
    help="Set the flushing level of messages.",
)
args = parser.parse_args()
config = vars(args)
if args.debug:
    print(config)
env = os.environ.copy()

collector_args = ""
if args.print_begin:
    collector_args += "print_begin;"
if args.profiling:
    collector_args += "profiling;"
if args.time_unit:
    collector_args += "time_unit:" + args.time_unit + ";"
if args.filter:
    collector_args += "filter:" + args.filter + ";"
if args.no_args:
    collector_args += "no_args;"
if args.json:
    collector_args += "json;"
env["UR_COLLECTOR_ARGS"] = collector_args

log_collector = ""
if args.debug:
    log_collector += "level:debug;"
else:
    log_collector += "level:info;"
log_collector += f"flush:{args.flush};"
if args.file:
    log_collector += "output:file," + args.file + ";"
elif args.stdout:
    log_collector += "output:stdout"
else:
    log_collector += "output:stderr"
env["UR_LOG_COLLECTOR"] = log_collector

env["XPTI_TRACE_ENABLE"] = "1"

env["UR_ENABLE_LAYERS"] = "UR_LAYER_TRACING"

xptifw_lib = get_dynamic_library_name("xptifw")
xptifw = find_library(args.libpath, xptifw_lib, args.recursive)
if xptifw is None:
    sys.exit("unable to find xptifw library - " + xptifw_lib)
env["XPTI_FRAMEWORK_DISPATCHER"] = xptifw

collector_lib = get_dynamic_library_name("ur_collector")

collector = find_library(args.libpath, collector_lib, args.recursive)
if collector is None:
    sys.exit("unable to find collector library - " + collector_lib)
env["XPTI_SUBSCRIBERS"] = collector

force_load = None

if args.mock:
    mock_lib = get_dynamic_library_name("ur_adapter_mock")
    mock_adapter = find_library(args.libpath, mock_lib, args.recursive)
    if mock_adapter is None:
        sys.exit("unable to find the mock adapter - " + mock_lib)
    force_load = '"' + mock_adapter + '"'

for adapter in args.adapter:
    adapter_path = (
        find_library(args.libpath, adapter, args.recursive)
        if is_filename(adapter)
        else adapter
    )
    if adapter_path is None:
        sys.exit("adapter does not specify a valid file " + args.adapter)
    if force_load is not None:
        force_load += ',"' + adapter_path + '"'
    else:
        force_load = adapter_path

if force_load:
    env["UR_ADAPTERS_FORCE_LOAD"] = force_load

if args.debug:
    print(env)

if config["command"]:
    # The core functionality is to pass the user's command,
    # and it is the user's responsibility to pass secure parameters.
    result = subprocess.run(config["command"], env=env)  # nosec B603
    if args.debug:
        print(result)
    exit(result.returncode)
else:
    parser.print_help()
    sys.exit("\n Error: Missing command to run.")
