#!/usr/bin/env python3

# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from benches.compute import *
from benches.gromacs import GromacsBench
from benches.velocity import VelocityBench
from benches.syclbench import *
from benches.llamacpp import *
from benches.umf import *
from benches.test import TestSuite
from options import Compare, options
from output_markdown import generate_markdown
from output_html import generate_html
from history import BenchmarkHistory
from utils.utils import prepare_workdir
from utils.compute_runtime import *
from utils.validate import Validate
from presets import enabled_suites, presets

import argparse
import re
import statistics

# Update this if you are changing the layout of the results files
INTERNAL_WORKDIR_VERSION = "2.0"


def run_iterations(
    benchmark: Benchmark,
    env_vars,
    iters: int,
    results: dict[str, list[Result]],
    failures: dict[str, str],
):
    for iter in range(iters):
        print(f"running {benchmark.name()}, iteration {iter}... ", flush=True)
        bench_results = benchmark.run(env_vars)
        if bench_results is None:
            failures[benchmark.name()] = "benchmark produced no results!"
            break

        for bench_result in bench_results:
            if not bench_result.passed:
                failures[bench_result.label] = "verification failed"
                print(f"complete ({bench_result.label}: verification failed).")
                continue

            print(
                f"{benchmark.name()} complete ({bench_result.label}: {bench_result.value:.3f} {bench_result.unit})."
            )

            bench_result.name = bench_result.label
            bench_result.lower_is_better = benchmark.lower_is_better()
            bench_result.suite = benchmark.get_suite_name()

            if bench_result.label not in results:
                results[bench_result.label] = []

            results[bench_result.label].append(bench_result)


# https://www.statology.org/modified-z-score/
def modified_z_score(values: list[float]) -> list[float]:
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    if mad == 0:
        return [0] * len(values)
    return [(0.6745 * (v - median)) / mad for v in values]


def remove_outliers(
    results: dict[str, list[Result]], threshold: float = 3.5
) -> dict[str, list[Result]]:
    new_results = {}
    for key, rlist in results.items():
        # don't eliminate outliers on first pass
        if len(rlist) <= options.iterations:
            new_results[key] = rlist
            continue

        values = [r.value for r in rlist]
        z_scores = modified_z_score(values)
        filtered_rlist = [r for r, z in zip(rlist, z_scores) if abs(z) <= threshold]

        if not filtered_rlist:
            new_results[key] = rlist
        else:
            new_results[key] = filtered_rlist

    return new_results


def process_results(
    results: dict[str, list[Result]], stddev_threshold_override
) -> tuple[bool, list[Result]]:
    processed: list[Result] = []
    # technically, we can detect whether result is below or above threshold per
    # individual result. However, we can't repeat benchmark runs with that
    # granularity. So we just reject all results and try again.
    valid_results = True  # above stddev threshold

    for label, rlist in remove_outliers(results).items():
        if len(rlist) == 0:
            continue

        if len(rlist) == 1:
            processed.append(rlist[0])
            continue

        values = [r.value for r in rlist]

        mean_value = statistics.mean(values)
        stddev = statistics.stdev(values)

        threshold = (
            stddev_threshold_override
            if stddev_threshold_override is not None
            else options.stddev_threshold
        ) * mean_value

        if stddev > threshold:
            print(f"stddev {stddev} above the threshold {threshold} for {label}")
            valid_results = False

        rlist.sort(key=lambda res: res.value)
        median_index = len(rlist) // 2
        median_result = rlist[median_index]

        # only override the stddev if not already set
        if median_result.stddev == 0.0:
            median_result.stddev = stddev

        processed.append(median_result)

    return valid_results, processed


def collect_metadata(suites):
    metadata = {}

    for s in suites:
        metadata.update(s.additional_metadata())
        suite_benchmarks = s.benchmarks()
        for benchmark in suite_benchmarks:
            metadata[benchmark.name()] = benchmark.get_metadata()

    return metadata


def main(directory, additional_env_vars, save_name, compare_names, filter):
    prepare_workdir(directory, INTERNAL_WORKDIR_VERSION)

    if options.build_compute_runtime:
        print(f"Setting up Compute Runtime {options.compute_runtime_tag}")
        cr = get_compute_runtime()
        print("Compute Runtime setup complete.")
        options.extra_ld_libraries.extend(cr.ld_libraries())
        options.extra_env_vars.update(cr.env_vars())

    suites = [
        ComputeBench(directory),
        VelocityBench(directory),
        SyclBench(directory),
        LlamaCppBench(directory),
        UMFSuite(directory),
        GromacsBench(directory),
        TestSuite(),
    ]

    # Collect metadata from all benchmarks without setting them up
    metadata = collect_metadata(suites)

    # If dry run, we're done
    if options.dry_run:
        suites = []

    benchmarks = []
    failures = {}

    for s in suites:
        if s.name() not in enabled_suites(options.preset):
            continue

        suite_benchmarks = s.benchmarks()
        if filter:
            suite_benchmarks = [
                benchmark
                for benchmark in suite_benchmarks
                if filter.search(benchmark.name())
            ]

        if suite_benchmarks:
            print(f"Setting up {type(s).__name__}")
            try:
                s.setup()
            except Exception as e:
                failures[s.name()] = f"Suite setup failure: {e}"
                print(f"{type(s).__name__} setup failed. Benchmarks won't be added.")
                print(f"failed: {e}")
            else:
                print(f"{type(s).__name__} setup complete.")
                benchmarks += suite_benchmarks

    for benchmark in benchmarks:
        try:
            if options.verbose:
                print(f"Setting up {benchmark.name()}... ")
            benchmark.setup()
            if options.verbose:
                print(f"{benchmark.name()} setup complete.")

        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                failures[benchmark.name()] = f"Benchmark setup failure: {e}"
                print(f"failed: {e}")

    results = []
    for benchmark in benchmarks:
        try:
            merged_env_vars = {**additional_env_vars}
            intermediate_results: dict[str, list[Result]] = {}
            processed: list[Result] = []
            for _ in range(options.iterations_stddev):
                run_iterations(
                    benchmark,
                    merged_env_vars,
                    options.iterations,
                    intermediate_results,
                    failures,
                )
                valid, processed = process_results(
                    intermediate_results, benchmark.stddev_threshold()
                )
                if valid:
                    break
            results += processed
        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                failures[benchmark.name()] = f"Benchmark run failure: {e}"
                print(f"failed: {e}")

    for benchmark in benchmarks:
        # this never has any useful information anyway, so hide it behind verbose
        if options.verbose:
            print(f"tearing down {benchmark.name()}... ", flush=True)
        benchmark.teardown()
        if options.verbose:
            print(f"{benchmark.name()} teardown complete.")

    this_name = options.current_run_name
    chart_data = {}

    if not options.dry_run:
        chart_data = {this_name: results}

    results_dir = directory
    if options.results_directory_override:
        results_dir = Path(options.results_directory_override)
    history = BenchmarkHistory(results_dir)
    # limit how many files we load.
    # should this be configurable?
    history.load(1000)

    # remove duplicates. this can happen if e.g., --compare baseline is specified manually.
    compare_names = (
        list(dict.fromkeys(compare_names)) if compare_names is not None else []
    )

    for name in compare_names:
        compare_result = history.get_compare(name)
        if compare_result:
            chart_data[name] = compare_result.results

    if options.output_markdown:
        markdown_content = generate_markdown(
            this_name, chart_data, failures, options.output_markdown
        )

        md_path = options.output_directory
        if options.output_directory is None:
            md_path = os.getcwd()

        with open(os.path.join(md_path, "benchmark_results.md"), "w") as file:
            file.write(markdown_content)

        print(
            f"Markdown with benchmark results has been written to {md_path}/benchmark_results.md"
        )

    saved_name = save_name if save_name is not None else this_name

    # It's important we don't save the current results into history before
    # we calculate historical averages or get latest results for compare.
    # Otherwise we might be comparing the results to themselves.
    if not options.dry_run:
        history.save(saved_name, results, save_name is not None)
        if saved_name not in compare_names:
            compare_names.append(saved_name)

    if options.output_html:
        html_path = options.output_directory
        if options.output_directory is None:
            html_path = os.path.join(os.path.dirname(__file__), "html")
        generate_html(history.runs, compare_names, html_path, metadata)


def validate_and_parse_env_args(env_args):
    env_vars = {}
    for arg in env_args:
        if "=" not in arg:
            raise ValueError(
                f"Environment variable argument '{arg}' is not in the form Variable=Value."
            )
        key, value = arg.split("=", 1)
        env_vars[key] = value
    return env_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Runtime Benchmark Runner")
    parser.add_argument(
        "benchmark_directory", type=str, help="Working directory to setup benchmarks."
    )
    parser.add_argument(
        "--sycl", type=str, help="Root directory of the SYCL compiler.", default=None
    )
    parser.add_argument("--ur", type=str, help="UR install prefix path", default=None)
    parser.add_argument("--umf", type=str, help="UMF install prefix path", default=None)
    parser.add_argument(
        "--adapter",
        type=str,
        help="Unified Runtime adapter to use.",
        default="level_zero",
    )
    parser.add_argument(
        "--no-rebuild",
        help="Do not rebuild the benchmarks from scratch.",
        action="store_true",
    )
    parser.add_argument(
        "--redownload",
        help="Always download benchmark data dependencies, even if they already exist.",
        action="store_true",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Use env variable for a benchmark run.",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save the results for comparison under a specified name.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare results against previously saved data.",
        action="append",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of times to run each benchmark to select a median value.",
        default=options.iterations,
    )
    parser.add_argument(
        "--stddev-threshold",
        type=float,
        help="If stddev pct is above this threshold, rerun all iterations",
        default=options.stddev_threshold,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for individual benchmarks in seconds.",
        default=options.timeout,
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Regex pattern to filter benchmarks by name.",
        default=None,
    )
    parser.add_argument(
        "--verbose", help="Print output of all the commands.", action="store_true"
    )
    parser.add_argument(
        "--exit-on-failure", help="Exit on first failure.", action="store_true"
    )
    parser.add_argument(
        "--compare-type",
        type=str,
        choices=[e.value for e in Compare],
        help="Compare results against previously saved data.",
        default=Compare.LATEST.value,
    )
    parser.add_argument(
        "--compare-max",
        type=int,
        help="How many results to read for comparisions",
        default=options.compare_max,
    )
    parser.add_argument(
        "--output-markdown",
        nargs="?",
        const=options.output_markdown,
        help="Specify whether markdown output should fit the content size limit for request validation",
    )
    parser.add_argument(
        "--output-html",
        help="Create HTML output. Local output is for direct local viewing of the html file, remote is for server deployment.",
        nargs="?",
        const=options.output_html,
        choices=["local", "remote"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Location for output files, if --output-html or --output-markdown was specified.",
        default=options.output_directory,
    )
    parser.add_argument(
        "--dry-run",
        help="Do not run any actual benchmarks",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute-runtime",
        nargs="?",
        const=options.compute_runtime_tag,
        help="Fetch and build compute runtime",
    )
    parser.add_argument(
        "--iterations-stddev",
        type=int,
        help="Max number of iterations of the loop calculating stddev after completed benchmark runs",
        default=options.iterations_stddev,
    )
    parser.add_argument(
        "--build-igc",
        help="Build IGC from source instead of using the OS-installed version",
        action="store_true",
        default=options.build_igc,
    )
    parser.add_argument(
        "--relative-perf",
        type=str,
        help="The name of the results which should be used as a baseline for metrics calculation",
        default=options.current_run_name,
    )
    parser.add_argument(
        "--cudnn-directory",
        type=str,
        help="Directory for cudnn library",
        default=None,
    )
    parser.add_argument(
        "--cublas-directory",
        type=str,
        help="Directory for cublas library",
        default=None,
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=[p for p in presets.keys()],
        help="Benchmark preset to run",
        default=options.preset,
    )
    parser.add_argument(
        "--build-jobs",
        type=int,
        help="Number of build jobs to run simultaneously",
        default=options.build_jobs,
    )
    parser.add_argument(
        "--hip-arch",
        type=str,
        help="HIP device architecture",
        default=None,
    )

    # Options intended for CI:
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Specify a custom directory to load/store (historical) results from",
        default=options.results_directory_override,
    )
    parser.add_argument(
        "--timestamp-override",
        type=lambda ts: Validate.timestamp(
            ts,
            throw=argparse.ArgumentTypeError(
                "Specified timestamp not in YYYYMMDD_HHMMSS format."
            ),
        ),
        help="Manually specify timestamp used in metadata",
        default=options.timestamp_override,
    )
    parser.add_argument(
        "--github-repo",
        type=lambda gh_repo: Validate.github_repo(
            gh_repo,
            throw=argparse.ArgumentTypeError(
                "Specified github repo not in <owner>/<repo> format."
            ),
        ),
        help="Manually specify github repo metadata of component tested (e.g. SYCL, UMF)",
        default=options.github_repo_override,
    )
    parser.add_argument(
        "--git-commit",
        type=lambda commit: Validate.commit_hash(
            commit,
            throw=argparse.ArgumentTypeError(
                "Specified commit is not a valid commit hash."
            ),
        ),
        help="Manually specify commit hash metadata of component tested (e.g. SYCL, UMF)",
        default=options.git_commit_override,
    )

    args = parser.parse_args()
    additional_env_vars = validate_and_parse_env_args(args.env)

    options.workdir = args.benchmark_directory
    options.verbose = args.verbose
    options.rebuild = not args.no_rebuild
    options.redownload = args.redownload
    options.sycl = args.sycl
    options.iterations = args.iterations
    options.timeout = args.timeout
    options.ur = args.ur
    options.ur_adapter = args.adapter
    options.exit_on_failure = args.exit_on_failure
    options.compare = Compare(args.compare_type)
    options.compare_max = args.compare_max
    options.output_markdown = args.output_markdown
    options.output_html = args.output_html
    options.dry_run = args.dry_run
    options.umf = args.umf
    options.iterations_stddev = args.iterations_stddev
    options.build_igc = args.build_igc
    options.current_run_name = args.relative_perf
    options.cudnn_directory = args.cudnn_directory
    options.cublas_directory = args.cublas_directory
    options.preset = args.preset
    options.results_directory_override = args.results_dir
    options.build_jobs = args.build_jobs
    options.hip_arch = args.hip_arch

    if args.build_igc and args.compute_runtime is None:
        parser.error("--build-igc requires --compute-runtime to be set")
    if args.compute_runtime is not None:
        options.build_compute_runtime = True
        options.compute_runtime_tag = args.compute_runtime
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            parser.error("Specified --output-dir is not a valid path")
        options.output_directory = os.path.abspath(args.output_dir)

    # Options intended for CI:
    options.timestamp_override = args.timestamp_override
    if args.results_dir is not None:
        if not os.path.isdir(args.results_dir):
            parser.error("Specified --results-dir is not a valid path")
        options.results_directory_override = os.path.abspath(args.results_dir)
    if args.github_repo is not None or args.git_commit is not None:
        if args.github_repo is None or args.git_commit is None:
            parser.error("--github-repo and --git_commit must both be defined together")
        options.github_repo_override = args.github_repo
        options.git_commit_override = args.git_commit

    benchmark_filter = re.compile(args.filter) if args.filter else None

    main(
        args.benchmark_directory,
        additional_env_vars,
        args.save,
        args.compare,
        benchmark_filter,
    )
