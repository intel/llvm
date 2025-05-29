# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random
from utils.utils import git_clone
from .base import Benchmark, Suite
from utils.result import Result
from utils.utils import run, create_build_path
from options import options
from utils.oneapi import get_oneapi
import os
import csv
import io
import re


def isUMFAvailable():
    return options.umf is not None


class UMFSuite(Suite):
    def __init__(self, directory):
        self.directory = directory

    def name(self) -> str:
        return "UMF"

    def setup(self):
        if not isUMFAvailable():
            return []
        self.built = True

    def benchmarks(self) -> list[Benchmark]:
        if not isUMFAvailable():
            return []

        benches = [
            GBench(self),
            GBenchUmfProxy(self),
            GBenchJemalloc(self),
            GBenchTbbProxy(self),
        ]

        return benches


class GBench(Benchmark):
    def __init__(self, bench):
        super().__init__(bench.directory, bench)

        self.bench = bench
        self.bench_name = "umf-benchmark"
        self.oneapi = get_oneapi()
        self.umf_lib = options.umf + "lib"

        self.fragmentation_prefix = "FRAGMENTATION_"

        self.num_cols_with_memory = 13

        self.col_name = "name"
        self.col_iterations = "iterations"
        self.col_real_time = "real_time"
        self.col_cpu_time = "cpu_time"
        self.col_time_unit = "time_unit"
        self.col_memory_overhead = "memory_overhead"

        self.idx_pool = 0
        self.idx_config = 1
        self.name_separator = "/"

        self.col_statistics_time = self.col_real_time
        self.col_statistics_memory = self.col_memory_overhead

        self.is_preloaded = False

        self.lib_to_be_replaced = None

    def name(self):
        return self.bench_name

    # --benchmark_format describes stdout output
    # --benchmark_out=<file> and --benchmark_out_format=<format>
    # describe output to a file
    def bin_args(self):
        return ["--benchmark_format=csv"]

    # these benchmarks are not stable, so set this at a large value
    def stddev_threshold(self) -> float:
        return 0.2  # 20%

    def extra_env_vars(self) -> dict:
        return {}

    def setup(self):
        if not isUMFAvailable():
            print("UMF prefix path not provided")
            return

        self.benchmark_bin = os.path.join(options.umf, "benchmark", self.bench_name)

    def is_memory_statistics_included(self, data_row):
        return len(data_row) == self.num_cols_with_memory

    def get_pool_and_config(self, full_name):
        list_split = full_name.split(self.name_separator, 1)
        if len(list_split) != 2:
            raise ValueError("Incorrect benchmark name format: ", full_name)

        return list_split[self.idx_pool], list_split[self.idx_config]

    def get_mean_time(self, datarow):
        return float(datarow[self.col_statistics_time])

    def get_memory_overhead(self, datarow):
        return float(datarow[self.col_statistics_memory])

    def get_unit_time_or_overhead(self, config):
        if re.search(f"^{self.fragmentation_prefix}", config):
            return "%"

        # the default time unit
        # might be changed globally with --benchmark_time_unit={ns|us|ms|s}
        # the change affects only benchmark where time unit has not been set
        # explicitly
        return "ns"

    def get_names_of_benchmarks_to_be_run(self, command, env_vars):
        list_all_command = command + ["--benchmark_list_tests"]

        if self.is_preloaded:
            list_all_command += ["--benchmark_filter=" + self.lib_to_be_replaced]

        all_names = self.run_bench(
            list_all_command, env_vars, add_sycl=False, ld_library=[self.umf_lib]
        ).splitlines()

        return all_names

    def run(self, env_vars) -> list[Result]:
        command = [f"{self.benchmark_bin}"]

        all_names = self.get_names_of_benchmarks_to_be_run(command, env_vars)

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        results = []

        for name in all_names:
            specific_benchmark = command + ["--benchmark_filter=^" + name + "$"]

            result = self.run_bench(
                specific_benchmark, env_vars, add_sycl=False, ld_library=[self.umf_lib]
            )

            parsed = self.parse_output(result)
            for r in parsed:
                (explicit_group, pool, value) = r
                label = f"{explicit_group} {pool}"
                results.append(
                    Result(
                        label=label,
                        value=value,
                        command=command,
                        env=env_vars,
                        stdout=result,
                        unit=self.get_unit_time_or_overhead(explicit_group),
                    )
                )

        return results

    def parse_output(self, output):
        csv_file = io.StringIO(output)
        reader = csv.DictReader(csv_file)

        results = []

        for row in reader:
            try:
                full_name = row[self.col_name]
                pool, config = self.get_pool_and_config(full_name)
                statistics_time = self.get_mean_time(row)

                if self.is_preloaded:
                    pool = self.get_preloaded_pool_name(pool)

                results.append((config, pool, statistics_time))

                if self.is_memory_statistics_included(row):
                    statistics_overhead = self.get_memory_overhead(row)
                    config = self.fragmentation_prefix + config

                    results.append((config, pool, statistics_overhead))

            except KeyError as e:
                raise ValueError(f"Error parsing output: {e}")

        return results

    def teardown(self):
        return


class GBenchPreloaded(GBench):
    def __init__(self, bench, lib_to_be_replaced, replacing_lib):
        super().__init__(bench)

        self.is_preloaded = True

        self.lib_to_be_replaced = lib_to_be_replaced
        self.replacing_lib = replacing_lib

    def get_preloaded_pool_name(self, pool_name) -> str:
        new_pool_name = pool_name.replace(self.lib_to_be_replaced, self.replacing_lib)

        return new_pool_name


class GBenchGlibc(GBenchPreloaded):
    def __init__(self, bench, replacing_lib):
        super().__init__(bench, lib_to_be_replaced="glibc", replacing_lib=replacing_lib)


class GBenchUmfProxy(GBenchGlibc):
    def __init__(self, bench):
        super().__init__(bench, replacing_lib="umfProxy")

    def extra_env_vars(self) -> dict:
        umf_proxy_path = os.path.join(options.umf, "lib", "libumf_proxy.so")
        return {"LD_PRELOAD": umf_proxy_path}


class GBenchJemalloc(GBenchGlibc):
    def __init__(self, bench):
        super().__init__(bench, replacing_lib="jemalloc")

    def extra_env_vars(self) -> dict:
        return {"LD_PRELOAD": "libjemalloc.so"}


class GBenchTbbProxy(GBenchGlibc):
    def __init__(self, bench):
        super().__init__(bench, replacing_lib="tbbProxy")

    def extra_env_vars(self) -> dict:
        return {"LD_PRELOAD": "libtbbmalloc_proxy.so"}
