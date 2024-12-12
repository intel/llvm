# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random
from utils.utils import git_clone
from .base import Benchmark, Suite
from .result import Result
from utils.utils import run, create_build_path
from .options import options
from .oneapi import get_oneapi
import os
import csv
import io

def isUMFAvailable():
    return options.umf is not None

class UMFSuite(Suite):
    def __init__(self, directory):
        self.directory = directory
        if not isUMFAvailable():
            print("UMF not provided. Related benchmarks will not run")

    def name(self) -> str:
        return "UMF"

    def setup(self):
        if not isUMFAvailable():
            return []
        self.built = True

    def benchmarks(self) -> list[Benchmark]:
        if not isUMFAvailable():
            return
        
        benches = [
            GBench(self),
        ]

        return benches

class ComputeUMFBenchmark(Benchmark):
    def __init__(self, bench, name):
        self.bench = bench
        self.bench_name = name
        self.oneapi = get_oneapi()

        self.col_name = None
        self.col_iterations = None
        self.col_real_time = None
        self.col_cpu_time = None
        self.col_time_unit = None

        self.col_statistics_time = None

        super().__init__(bench.directory)

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def setup(self):
        if not isUMFAvailable():
            print("UMF prefix path not provided")
            return

        self.benchmark_bin = os.path.join(options.umf, 'benchmark', self.bench_name)

    def run(self, env_vars) -> list[Result]:
        command = [
            f"{self.benchmark_bin}",
        ]

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        result = self.run_bench(command, env_vars, add_sycl=False, ld_library=[self.oneapi.tbb_lib()])
        parsed = self.parse_output(result)
        results = []
        for r in parsed:
            (config, pool, mean) = r
            label = f"{config} {pool}"
            results.append(Result(label=label, value=mean, command=command, env=env_vars, stdout=result, unit="ns", explicit_group=config))
        return results

    # Implementation with self.col_* indices could lead to the division by None
    def get_mean(self, datarow):
        raise NotImplementedError()

    def teardown(self):
        return

class GBench(ComputeUMFBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "umf-benchmark")

        self.col_name = 0
        self.col_iterations = 1
        self.col_real_time = 2
        self.col_cpu_time = 3
        self.col_time_unit = 4

        self.idx_pool = 0
        self.idx_config = 1
        self.name_separator = '/'

        self.col_statistics_time = self.col_real_time

    def name(self):
        return self.bench_name

    # --benchmark_format describes stdout output
    # --benchmark_out=<file> and --benchmark_out_format=<format>
    # describe output to a file 
    def bin_args(self):
        return ["--benchmark_format=csv"]

    # the default unit
    # might be changed globally with --benchmark_time_unit={ns|us|ms|s}
    # the change affects only benchmark where time unit has not been set
    # explicitly
    def unit(self):
        return "ns"

    # these benchmarks are not stable, so set this at a large value
    def stddev_threshold(self) -> float:
        return 0.2 # 20%

    def get_pool_and_config(self, full_name):
        list_split = full_name.split(self.name_separator, 1)
        if len(list_split) != 2:
            raise ValueError("Incorrect benchmark name format: ", full_name)
        
        return list_split[self.idx_pool], list_split[self.idx_config]

    def get_mean(self, datarow):
        return float(datarow[self.col_statistics_time])

    def parse_output(self, output):
        csv_file = io.StringIO(output)
        reader = csv.reader(csv_file)

        data_row = next(reader, None)
        if data_row is None:
            raise ValueError("Benchmark output does not contain data.")

        results = []
        for row in reader:
            try:
                full_name = row[self.col_name]
                pool, config = self.get_pool_and_config(full_name)
                mean = self.get_mean(row)
                results.append((config, pool, mean))
            except KeyError as e:
                raise ValueError(f"Error parsing output: {e}")

        return results
