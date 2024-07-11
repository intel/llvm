#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from utils.utils import prepare_workdir, load_benchmark_results, save_benchmark_results;
from benches.api_overhead import APIOverheadSYCL
from benches.hashtable import Hashtable
from benches.bitcracker import Bitcracker
from benches.cudaSift import CudaSift
from benches.easywave import Easywave
from benches.quicksilver import QuickSilver
from benches.SobelFilter import SobelFilter
from benches.velocity import VelocityBench
from benches.options import options
from output import generate_markdown
import argparse

# Update this if you are changing the layout of the results files
INTERNAL_WORKDIR_VERSION = '1.0'

def main(directory, additional_env_vars, save_name, compare_names):
    variants = [
        ({'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '0'}, "Imm-CmdLists-OFF"),
        ({'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '1'}, ""),
    ]

    prepare_workdir(directory, INTERNAL_WORKDIR_VERSION)

    vb = VelocityBench(directory)

    benchmarks = [
        APIOverheadSYCL(directory),
        Hashtable(vb),
        Bitcracker(vb),
        #CudaSift(vb), TODO: the benchmark is passing, but is outputting "Failed to allocate device data"
        Easywave(vb),
        QuickSilver(vb),
        SobelFilter(vb)
    ]

    for benchmark in benchmarks:
        benchmark.setup()

    results = []
    for benchmark in benchmarks:
        for env_vars, extra_label in variants:
            merged_env_vars = {**env_vars, **additional_env_vars}
            bench_results = benchmark.run(merged_env_vars)
            for res in bench_results:
                res.unit = benchmark.unit()
                res.name = benchmark.name()
                res.label += f" {extra_label}"
                results.append(res)

    for benchmark in benchmarks:
        benchmark.teardown()

    chart_data = {"This PR" : results}

    for name in compare_names:
        compare_result = load_benchmark_results(directory, name)
        if compare_result:
            chart_data[name] = compare_result

    if save_name:
        save_benchmark_results(directory, save_name, results)

    markdown_content = generate_markdown(chart_data)

    with open('benchmark_results.md', 'w') as file:
        file.write(markdown_content)

    print("Markdown with benchmark results has been written to benchmark_results.md")

def validate_and_parse_env_args(env_args):
    env_vars = {}
    for arg in env_args:
        if '=' not in arg:
            raise ValueError(f"Environment variable argument '{arg}' is not in the form Variable=Value.")
        key, value = arg.split('=', 1)
        env_vars[key] = value
    return env_vars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Runtime Benchmark Runner')
    parser.add_argument('benchmark_directory', type=str, help='Working directory to setup benchmarks.')
    parser.add_argument('sycl', type=str, help='Root directory of the SYCL compiler.')
    parser.add_argument("--no-rebuild", help='Rebuild the benchmarks from scratch.', action="store_true")
    parser.add_argument("--env", type=str, help='Use env variable for a benchmark run.', action="append", default=[])
    parser.add_argument("--save", type=str, help='Save the results for comparison under a specified name.')
    parser.add_argument("--compare", type=str, help='Compare results against previously saved data.', action="append", default=["baseline"])

    args = parser.parse_args()
    additional_env_vars = validate_and_parse_env_args(args.env)

    options.rebuild = not args.no_rebuild
    options.sycl = args.sycl

    main(args.benchmark_directory, additional_env_vars, args.save, args.compare)
