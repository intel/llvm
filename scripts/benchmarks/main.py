#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from utils.utils import prepare_workdir, load_benchmark_results, save_benchmark_results;
from benches.compute import *
from benches.hashtable import Hashtable
from benches.bitcracker import Bitcracker
from benches.cudaSift import CudaSift
from benches.easywave import Easywave
from benches.quicksilver import QuickSilver
from benches.SobelFilter import SobelFilter
from benches.velocity import VelocityBench
from benches.syclbench import *
from benches.options import options
from output import generate_markdown
import argparse
import re

# Update this if you are changing the layout of the results files
INTERNAL_WORKDIR_VERSION = '1.6'

def main(directory, additional_env_vars, save_name, compare_names, filter):
    prepare_workdir(directory, INTERNAL_WORKDIR_VERSION)

    cb = ComputeBench(directory)
    sb = SyclBench(directory)
    vb = VelocityBench(directory)

    benchmarks = [
        # *** Compute benchmarks
        SubmitKernelSYCL(cb, 0),
        SubmitKernelSYCL(cb, 1),
        SubmitKernelUR(cb, 0),
        SubmitKernelUR(cb, 1),
        QueueInOrderMemcpy(cb, 0, 'Device', 'Device', 1024),
        QueueInOrderMemcpy(cb, 0, 'Host', 'Device', 1024),
        QueueMemcpy(cb, 'Device', 'Device', 1024),
        StreamMemory(cb, 'Triad', 10 * 1024, 'Device'),
        ExecImmediateCopyQueue(cb, 0, 1, 'Device', 'Device', 1024),
        ExecImmediateCopyQueue(cb, 1, 1, 'Device', 'Host', 1024),
        VectorSum(cb),

        # *** Velocity benchmarks
        Hashtable(vb),
        Bitcracker(vb),
        CudaSift(vb),
        Easywave(vb),
        QuickSilver(vb),
        SobelFilter(vb),

        # *** sycl-bench multi benchmarks
        Blocked_transform(sb),
        DagTaskI(sb),
        DagTaskS(sb),
        HostDevBandwidth(sb),
        LocalMem(sb),
        Pattern_L2(sb),
        Reduction(sb),
        ScalarProd(sb),
        SegmentReduction(sb),
        UsmAccLatency(sb),
        UsmAllocLatency(sb),
        UsmInstrMix(sb),
        UsmPinnedOverhead(sb),
        VecAdd(sb),

        # *** sycl-bench single benchmarks
        TwoDConvolution(sb),
        Two_mm(sb),
        Three_mm(sb),
        Arith(sb),
        Atax(sb),
        Atomic_reduction(sb),
        Bicg(sb),
        Correlation(sb),
        Covariance(sb),
        Gemm(sb),
        Gesumv(sb),
        Gramschmidt(sb),
        KMeans(sb),
        LinRegCoeff(sb),
        LinRegError(sb),
        MatmulChain(sb),
        MolDyn(sb),
        Mvt(sb),
        Sf(sb),
        Syr2k(sb),
        Syrk(sb),
    ]

    if filter:
        benchmarks = [benchmark for benchmark in benchmarks if filter.search(benchmark.name())]

    for benchmark in benchmarks:
        try:
            print(f"setting up {benchmark.name()}... ", end='', flush=True)
            benchmark.setup()
            print("complete.")

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
            for iter in range(options.iterations):
                print(f"running {benchmark.name()}, iteration {iter}... ", end='', flush=True)
                bench_results = benchmark.run(merged_env_vars)
                if bench_results is not None:
                    for bench_result in bench_results:
                        print(f"complete ({bench_result.label}: {bench_result.value} {benchmark.unit()}).")
                        iteration_results.append(bench_result)
                else:
                    print(f"did not finish.")

            if len(iteration_results) == 0:
                continue

            for label in set([result.label for result in iteration_results]):
                label_results = [result for result in iteration_results if result.label == label]
                label_results.sort(key=lambda res: res.value)
                median_index = len(label_results) // 2
                median_result = label_results[median_index]

                median_result.unit = benchmark.unit()
                median_result.name = label

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

    chart_data = {"This PR" : results}

    for name in compare_names:
        print(f"compare name: {name}")
        compare_result = load_benchmark_results(directory, name)
        if compare_result:
            chart_data[name] = compare_result

    if save_name:
        save_benchmark_results(directory, save_name, results)

    markdown_content = generate_markdown(chart_data)

    with open('benchmark_results.md', 'w') as file:
        file.write(markdown_content)

    print(f"Markdown with benchmark results has been written to {os.getcwd()}/benchmark_results.md")

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
    parser.add_argument('ur_dir', type=str, help='UR install prefix path')
    parser.add_argument('ur_adapter_name', type=str, help='Options to build the Unified Runtime as part of the benchmark')
    parser.add_argument("--no-rebuild", help='Rebuild the benchmarks from scratch.', action="store_true")
    parser.add_argument("--env", type=str, help='Use env variable for a benchmark run.', action="append", default=[])
    parser.add_argument("--save", type=str, help='Save the results for comparison under a specified name.')
    parser.add_argument("--compare", type=str, help='Compare results against previously saved data.', action="append", default=["baseline"])
    parser.add_argument("--iterations", type=int, help='Number of times to run each benchmark to select a median value.', default=5)
    parser.add_argument("--timeout", type=int, help='Timeout for individual benchmarks in seconds.', default=600)
    parser.add_argument("--filter", type=str, help='Regex pattern to filter benchmarks by name.', default=None)
    parser.add_argument("--epsilon", type=float, help='Threshold to consider change of performance significant', default=0.005)
    parser.add_argument("--verbose", help='Print output of all the commands.', action="store_true")
    parser.add_argument("--exit_on_failure", help='Exit on first failure.', action="store_true")

    args = parser.parse_args()
    additional_env_vars = validate_and_parse_env_args(args.env)

    options.verbose = args.verbose
    options.rebuild = not args.no_rebuild
    options.sycl = args.sycl
    options.iterations = args.iterations
    options.timeout = args.timeout
    options.epsilon = args.epsilon
    options.ur_dir = args.ur_dir
    options.ur_adapter_name = args.ur_adapter_name
    options.exit_on_failure = args.exit_on_failure

    benchmark_filter = re.compile(args.filter) if args.filter else None

    main(args.benchmark_directory, additional_env_vars, args.save, args.compare, benchmark_filter)
