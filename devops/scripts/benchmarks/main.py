#!/usr/bin/env python3

# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import re
import statistics
import os

from benches.compute import *
from benches.gromacs import GromacsBench
from benches.velocity import VelocityBench
from benches.syclbench import *
from benches.llamacpp import *
from benches.umf import *
from benches.test import TestSuite
from benches.benchdnn import OneDnnBench
from benches.base import TracingType
from options import Compare, options
from output_markdown import generate_markdown
from output_html import generate_html
from history import BenchmarkHistory
from utils.utils import prepare_workdir
from utils.compute_runtime import *
from utils.validate import Validate
from utils.detect_versions import DetectVersion
from utils.logger import log, initialize_logger
from presets import enabled_suites, presets

# Update this if you are changing the layout of the results files
INTERNAL_WORKDIR_VERSION = "2.0"


def run_iterations(
    benchmark: Benchmark,
    env_vars,
    iters: int,
    results: dict[str, list[Result]],
    failures: dict[str, str],
    run_trace: TracingType = TracingType.NONE,
    force_trace: bool = False,
):
    for iter in range(iters):
        log.info(f"running {benchmark.name()}, iteration {iter}... ")
        try:
            bench_results = benchmark.run(
                env_vars, run_trace=run_trace, force_trace=force_trace
            )
            if bench_results is None:
                if options.exit_on_failure:
                    raise RuntimeError(f"Benchmark produced no results!")
                else:
                    failures[benchmark.name()] = "benchmark produced no results!"
                    break

            for bench_result in bench_results:
                if bench_result.value == 0.0:
                    if options.exit_on_failure:
                        raise RuntimeError("Benchmark result is zero!")
                    else:
                        failure_label = f"{benchmark.name()} iteration {iter}"
                        failures[failure_label] = "benchmark result is zero!"
                        log.error(
                            f"complete ({failure_label}: benchmark result is zero!)."
                        )
                        continue
                log.info(
                    f"{benchmark.name()} complete ({bench_result.label}: {bench_result.value:.3f} {bench_result.unit})."
                )
                bench_result.name = bench_result.label
                bench_result.lower_is_better = benchmark.lower_is_better()
                bench_result.suite = benchmark.get_suite_name()

                if bench_result.label not in results:
                    results[bench_result.label] = []

                results[bench_result.label].append(bench_result)
        except Exception as e:
            failure_label = f"{benchmark.name()} iteration {iter}"
            if options.exit_on_failure:
                raise RuntimeError(
                    f"Benchmark failed: {failure_label} verification failed: {str(e)}"
                )
            else:
                failures[failure_label] = f"verification failed: {str(e)}"
                log.error(f"complete ({failure_label}: verification failed: {str(e)}).")
                continue


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
            log.warning(f"stddev {stddev} above the threshold {threshold} for {label}")
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
            results = benchmark.get_metadata()
            metadata.update(results)

    return metadata


def main(directory, additional_env_vars, compare_names, filter):
    prepare_workdir(directory, INTERNAL_WORKDIR_VERSION)

    if options.dry_run:
        log.info("Dry run mode enabled. No benchmarks will be executed.")

    options.unitrace = args.unitrace is not None

    if options.unitrace and options.save_name is None:
        raise ValueError(
            "Unitrace requires a save name to be specified via --save option."
        )

    if options.flamegraph and options.save_name is None:
        raise ValueError(
            "FlameGraph requires a save name to be specified via --save option."
        )

    if options.build_compute_runtime:
        log.info(f"Setting up Compute Runtime {options.compute_runtime_tag}")
        cr = get_compute_runtime()
        log.info("Compute Runtime setup complete.")
        options.extra_ld_libraries.extend(cr.ld_libraries())
        options.extra_env_vars.update(cr.env_vars())

    suites = [
        ComputeBench(),
        VelocityBench(),
        SyclBench(),
        LlamaCppBench(),
        UMFSuite(),
        GromacsBench(),
        OneDnnBench(),
        TestSuite(),
    ]

    # Collect metadata from all benchmarks without setting them up
    metadata = collect_metadata(suites)

    # If dry run, we're done
    if options.dry_run:
        suites = []

    benchmarks = []
    failures = {}

    # TODO: rename "s", rename setup in suite to suite_setup, rename setup in benchmark to benchmark_setup
    # TODO: do not add benchmarks whose suite setup failed
    # TODO: add a mode where we fail etire script in case of setup (or other) failures and use in CI

    for s in suites:
        if s.name() not in enabled_suites(options.preset):
            continue

        # filter out benchmarks that are disabled
        suite_benchmarks = [
            benchmark for benchmark in s.benchmarks() if benchmark.enabled()
        ]
        if filter:
            suite_benchmarks = [
                benchmark
                for benchmark in suite_benchmarks
                if filter.search(benchmark.name())
            ]

        if suite_benchmarks:
            log.info(f"Setting up {type(s).__name__}")
            try:
                s.setup()
            except Exception as e:
                if options.exit_on_failure:
                    raise e
                failures[s.name()] = f"Suite setup failure: {e}"
                log.error(
                    f"{type(s).__name__} setup failed. Benchmarks won't be added."
                )
                log.error(f"failed: {e}")
            else:
                log.info(f"{type(s).__name__} setup complete.")
                benchmarks += suite_benchmarks

    for benchmark in benchmarks:
        try:
            log.debug(f"Setting up {benchmark.name()}... ")
            benchmark.setup()
            log.debug(f"{benchmark.name()} setup complete.")

        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                failures[benchmark.name()] = f"Benchmark setup failure: {e}"
                log.error(f"failed: {e}")

    results = []
    if benchmarks:
        log.info(f"Running {len(benchmarks)} benchmarks...")
    elif not options.dry_run:
        raise RuntimeError("No benchmarks to run.")
    for benchmark in benchmarks:
        try:
            merged_env_vars = {**additional_env_vars}
            intermediate_results: dict[str, list[Result]] = {}
            processed: list[Result] = []

            # Determine if we should run regular benchmarks
            # Run regular benchmarks if:
            # - No tracing options specified, OR
            # - Any tracing option is set to "inclusive"
            should_run_regular = (
                not options.unitrace
                and not options.flamegraph  # No tracing options
                or args.unitrace == "inclusive"  # Unitrace inclusive
                or args.flamegraph == "inclusive"  # Flamegraph inclusive
            )

            if should_run_regular:
                for _ in range(options.iterations_stddev):
                    run_iterations(
                        benchmark,
                        merged_env_vars,
                        options.iterations,
                        intermediate_results,
                        failures,
                        run_trace=TracingType.NONE,
                    )
                    valid, processed = process_results(
                        intermediate_results, benchmark.stddev_threshold()
                    )
                    if valid:
                        break

            # single unitrace run independent of benchmark iterations (if unitrace enabled)
            if options.unitrace and (
                benchmark.traceable(TracingType.UNITRACE) or args.unitrace == "force"
            ):
                run_iterations(
                    benchmark,
                    merged_env_vars,
                    1,
                    intermediate_results,
                    failures,
                    run_trace=TracingType.UNITRACE,
                    force_trace=(args.unitrace == "force"),
                )
            # single flamegraph run independent of benchmark iterations (if flamegraph enabled)
            if options.flamegraph and (
                benchmark.traceable(TracingType.FLAMEGRAPH)
                or args.flamegraph == "force"
            ):
                run_iterations(
                    benchmark,
                    merged_env_vars,
                    1,
                    intermediate_results,
                    failures,
                    run_trace=TracingType.FLAMEGRAPH,
                    force_trace=(args.flamegraph == "force"),
                )

            results += processed
        except Exception as e:
            if options.exit_on_failure:
                raise e
            else:
                failures[benchmark.name()] = f"Benchmark run failure: {e}"
                log.error(f"failed: {e}")

    for benchmark in benchmarks:
        # this never has any useful information anyway, so hide it behind verbose
        log.debug(f"tearing down {benchmark.name()}... ")
        benchmark.teardown()
        log.debug(f"{benchmark.name()} teardown complete.")

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
    log.info(f"Loading benchmark history from {results_dir}...")
    history.load()
    log.info(f"Loaded {len(history.runs)} benchmark runs.")

    if compare_names:
        log.info(f"Comparing against {len(compare_names)} previous runs...")
        # remove duplicates. this can happen if e.g., --compare baseline is specified manually.
        compare_names = list(dict.fromkeys(compare_names))
        for name in compare_names:
            compare_result = history.get_compare(name)
            if compare_result:
                chart_data[name] = compare_result.results
        log.info(f"Comparison complete.")

    if options.output_markdown:
        log.info("Generating markdown with benchmark results...")
        markdown_content = generate_markdown(
            this_name, chart_data, failures, options.output_markdown, metadata
        )

        md_path = options.output_directory
        if options.output_directory is None:
            md_path = os.getcwd()

        with open(os.path.join(md_path, "benchmark_results.md"), "w") as file:
            file.write(markdown_content)

        log.info(
            f"Markdown with benchmark results has been written to {md_path}/benchmark_results.md"
        )

    saved_name = options.save_name if options.save_name is not None else this_name

    # It's important we don't save the current results into history before
    # we calculate historical averages or get latest results for compare.
    # Otherwise we might be comparing the results to themselves.
    if not options.dry_run:
        log.info(f"Saving benchmark results...")
        history.save(saved_name, results)
        if saved_name not in compare_names:
            compare_names.append(saved_name)
        log.info(f"Benchmark results saved.")

    if options.output_html:
        html_path = options.output_directory
        if options.output_directory is None:
            html_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "html")
            )
        log.info(f"Generating HTML with benchmark results in {html_path}...")

        generate_html(history, compare_names, html_path, metadata)
        log.info(f"HTML with benchmark results has been generated")

    if options.exit_on_failure and failures:
        # just in case code missed to raise earlier
        raise RuntimeError(str(failures))


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
        default=[],
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
        "--verbose",
        help="Set logging level to DEBUG. Overrides --log-level.",
        action="store_true",
    )
    parser.add_argument(
        "--exit-on-failure",
        help="Exit on first benchmark failure.",
        action="store_true",
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
        help="How many results to read for comparisons",
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
    parser.add_argument(
        "--unitrace",
        nargs="?",
        const="exclusive",
        default=None,
        help='Unitrace logs generation. "exclusive" omits regular benchmarks doing only trace generation - this is default and can be omitted. "inclusive" generation is done along regular benchmarks. "force" is same as exclusive but ignores traceable() method.',
        choices=["exclusive", "inclusive", "force"],
    )
    parser.add_argument(
        "--flamegraph",
        nargs="?",
        const="exclusive",
        default=None,
        help='FlameGraphs generation. "exclusive" omits regular benchmarks doing only trace generation - this is default and can be omitted. "inclusive" generation is done along regular benchmarks. "force" is same as exclusive but ignores traceable() method.',
        choices=["exclusive", "inclusive", "force"],
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
                "Specified timestamp not in YYYYMMDD_HHMMSS format"
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
                "Specified github repo not in <owner>/<repo> format"
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
                "Specified commit is not a valid commit hash"
            ),
        ),
        help="Manually specify commit hash metadata of component tested (e.g. SYCL, UMF)",
        default=options.git_commit_override,
    )

    parser.add_argument(
        "--detect-version",
        type=lambda components: Validate.on_re(
            components,
            r"[a-z_,]+",
            throw=argparse.ArgumentTypeError(
                "Specified --detect-version is not a comma-separated list"
            ),
        ),
        help="Detect versions of components used: comma-separated list with choices from sycl,compute_runtime",
        default=None,
    )
    parser.add_argument(
        "--detect-version-cpp-path",
        type=Path,
        help="Location of detect_version.cpp used to query e.g. DPC++, L0",
        default=None,
    )
    parser.add_argument(
        "--archive-baseline-after",
        type=int,
        help="Archive baseline results (runs starting with 'Baseline_') older than this many days. "
        "Archived results are stored separately and can be viewed in the HTML UI by enabling "
        "'Include archived runs'. This helps manage the size of the primary dataset.",
        default=options.archive_baseline_days,
    )
    parser.add_argument(
        "--archive-pr-after",
        type=int,
        help="Archive PR and other non-baseline results older than this many days. "
        "Archived results are stored separately and can be viewed in the HTML UI by enabling "
        "'Include archived runs'. PR runs typically have a shorter retention period than baselines.",
        default=options.archive_pr_days,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level",
        default="info",
    )

    args = parser.parse_args()
    additional_env_vars = validate_and_parse_env_args(args.env)

    options.workdir = args.benchmark_directory
    options.redownload = args.redownload
    options.sycl = args.sycl
    options.iterations = args.iterations
    options.timeout = args.timeout
    options.ur = args.ur
    options.ur_adapter = args.adapter
    options.exit_on_failure = args.exit_on_failure
    options.save_name = args.save
    options.compare = Compare(args.compare_type)
    options.compare_max = args.compare_max
    options.output_markdown = args.output_markdown
    if args.output_html is not None:
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
    options.flamegraph = args.flamegraph is not None
    options.archive_baseline_days = args.archive_baseline_after
    options.archive_pr_days = args.archive_pr_after

    # Initialize logger with command line arguments
    initialize_logger(args.verbose, args.log_level)

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

    # Automatically detect versions:
    if args.detect_version is not None:
        detect_ver_path = args.detect_version_cpp_path
        if detect_ver_path is None:
            detect_ver_path = Path(
                f"{os.path.dirname(__file__)}/utils/detect_versions.cpp"
            )
            if not detect_ver_path.is_file():
                parser.error(
                    f"Unable to find detect_versions.cpp at {detect_ver_path}, please specify --detect-version-cpp-path"
                )
        elif not detect_ver_path.is_file():
            parser.error(f"Specified --detect-version-cpp-path is not a valid file")

        enabled_components = args.detect_version.split(",")
        options.detect_versions.sycl = "sycl" in enabled_components
        options.detect_versions.compute_runtime = (
            "compute_runtime" in enabled_components
        )

        detect_res = DetectVersion.init(detect_ver_path)

    benchmark_filter = re.compile(args.filter) if args.filter else None

    try:
        options.device_architecture = get_device_architecture(additional_env_vars)
    except Exception as e:
        options.device_architecture = ""
        log.warning(f"Failed to fetch device architecture: {e}")
        log.warning("Defaulting to generic benchmark parameters.")

    log.info(f"Selected device architecture: {options.device_architecture}")

    main(
        args.benchmark_directory,
        additional_env_vars,
        args.compare,
        benchmark_filter,
    )
