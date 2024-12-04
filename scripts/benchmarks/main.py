#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from benches.compute import *
from benches.velocity import VelocityBench
from benches.syclbench import *
from benches.llamacpp import *
from benches.test import TestSuite
from benches.options import Compare, options
from output_markdown import generate_markdown
from output_html import generate_html
from history import BenchmarkHistory
from utils.utils import prepare_workdir;

import argparse
import re

# Update this if you are changing the layout of the results files
INTERNAL_WORKDIR_VERSION = '2.0'

def main(directory, additional_env_vars, save_name, compare_names, filter):
    prepare_workdir(directory, INTERNAL_WORKDIR_VERSION)

    suites = [
        ComputeBench(directory),
        VelocityBench(directory),
        SyclBench(directory),
        LlamaCppBench(directory),
        #TestSuite()
    ] if not options.dry_run else []

    benchmarks = []

    for s in suites:
        print(f"Setting up {type(s).__name__}")
        s.setup()
        print(f"{type(s).__name__} setup complete.")

    for s in suites:
        benchmarks += s.benchmarks()

    if filter:
        benchmarks = [benchmark for benchmark in benchmarks if filter.search(benchmark.name())]

    for b in benchmarks:
        print(b.name())

    for benchmark in benchmarks:
        try:
            print(f"Setting up {benchmark.name()}... ")
            benchmark.setup()
            print(f"{benchmark.name()} setup complete.")

        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                print(f"failed: {e}")

    results = []
    for benchmark in benchmarks:
        try:
            merged_env_vars = {**additional_env_vars}
            iteration_results = []
            iterations = options.iterations if not benchmark.ignore_iterations() else 1
            for iter in range(iterations):
                print(f"running {benchmark.name()}, iteration {iter}... ", end='', flush=True)
                bench_results = benchmark.run(merged_env_vars)
                if bench_results is not None:
                    for bench_result in bench_results:
                        if bench_result.passed:
                            print(f"complete ({bench_result.label}: {bench_result.value:.3f} {bench_result.unit}).")
                        else:
                            print(f"complete ({bench_result.label}: verification FAILED)")
                        iteration_results.append(bench_result)
                else:
                    print(f"did not finish (OK for sycl-bench).")
                    break

            if len(iteration_results) == 0:
                continue

            for label in set([result.label for result in iteration_results]):
                label_results = [result for result in iteration_results if result.label == label and result.passed == True]
                if len(label_results) > 0:
                    label_results.sort(key=lambda res: res.value)
                    median_index = len(label_results) // 2
                    median_result = label_results[median_index]

                    median_result.name = label
                    median_result.lower_is_better = benchmark.lower_is_better()

                    results.append(median_result)
        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                print(f"failed: {e}")

    for benchmark in benchmarks:
        print(f"tearing down {benchmark.name()}... ", end='', flush=True)
        benchmark.teardown()
        print("complete.")

    this_name = "This PR"

    chart_data = {this_name : results}

    history = BenchmarkHistory(directory)
    # limit how many files we load.
    # should this be configurable?
    history.load(1000)

    for name in compare_names:
        compare_result = history.get_compare(name)
        if compare_result:
            chart_data[name] = compare_result.results

    if options.output_markdown:
        markdown_content = generate_markdown(this_name, chart_data)

        with open('benchmark_results.md', 'w') as file:
            file.write(markdown_content)

        print(f"Markdown with benchmark results has been written to {os.getcwd()}/benchmark_results.md")

    saved_name = save_name if save_name is not None else this_name

    # It's important we don't save the current results into history before
    # we calculate historical averages or get latest results for compare.
    # Otherwise we might be comparing the results to themselves.
    if not options.dry_run:
        history.save(saved_name, results, save_name is not None)
        compare_names.append(saved_name)

    if options.output_html:
        html_content = generate_html(history.runs, 'oneapi-src/unified-runtime', compare_names)

        with open('benchmark_results.html', 'w') as file:
            file.write(html_content)

        print(f"HTML with benchmark results has been written to {os.getcwd()}/benchmark_results.html")

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
    parser.add_argument('--sycl', type=str, help='Root directory of the SYCL compiler.', default=None)
    parser.add_argument('--ur', type=str, help='UR install prefix path', default=None)
    parser.add_argument('--adapter', type=str, help='Options to build the Unified Runtime as part of the benchmark', default="level_zero")
    parser.add_argument("--no-rebuild", help='Rebuild the benchmarks from scratch.', action="store_true")
    parser.add_argument("--env", type=str, help='Use env variable for a benchmark run.', action="append", default=[])
    parser.add_argument("--save", type=str, help='Save the results for comparison under a specified name.')
    parser.add_argument("--compare", type=str, help='Compare results against previously saved data.', action="append", default=["baseline"])
    parser.add_argument("--iterations", type=int, help='Number of times to run each benchmark to select a median value.', default=5)
    parser.add_argument("--timeout", type=int, help='Timeout for individual benchmarks in seconds.', default=600)
    parser.add_argument("--filter", type=str, help='Regex pattern to filter benchmarks by name.', default=None)
    parser.add_argument("--epsilon", type=float, help='Threshold to consider change of performance significant', default=0.005)
    parser.add_argument("--verbose", help='Print output of all the commands.', action="store_true")
    parser.add_argument("--exit-on-failure", help='Exit on first failure.', action="store_true")
    parser.add_argument("--compare-type", type=str, choices=[e.value for e in Compare], help='Compare results against previously saved data.', default=Compare.LATEST.value)
    parser.add_argument("--compare-max", type=int, help='How many results to read for comparisions', default=10)
    parser.add_argument("--output-html", help='Create HTML output', action="store_true", default=False)
    parser.add_argument("--output-markdown", help='Create Markdown output', action="store_true", default=True)
    parser.add_argument("--dry-run", help='Do not run any actual benchmarks', action="store_true", default=False)

    args = parser.parse_args()
    additional_env_vars = validate_and_parse_env_args(args.env)

    options.workdir = args.benchmark_directory
    options.verbose = args.verbose
    options.rebuild = not args.no_rebuild
    options.sycl = args.sycl
    options.iterations = args.iterations
    options.timeout = args.timeout
    options.epsilon = args.epsilon
    options.ur = args.ur
    options.ur_adapter = args.adapter
    options.exit_on_failure = args.exit_on_failure
    options.compare = Compare(args.compare_type)
    options.compare_max = args.compare_max
    options.output_html = args.output_html
    options.output_markdown = args.output_markdown
    options.dry_run = args.dry_run

    benchmark_filter = re.compile(args.filter) if args.filter else None

    main(args.benchmark_directory, additional_env_vars, args.save, args.compare, benchmark_filter)
