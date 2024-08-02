# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
import io
from utils.utils import run, git_clone
from .base import Benchmark
from .result import Result
from .options import options

class APIOverheadSYCL(Benchmark):
    def __init__(self, directory):
        super().__init__(directory)

    def name(self):
        return "api_overhead_benchmark_sycl, mean execution time per 10 kernels"

    def unit(self):
        return "μs"

    def setup(self):
        repo_path = git_clone(self.directory, "compute-benchmarks-repo", "https://github.com/intel/compute-benchmarks.git", "0f758021dce9ba32341a503739b69db057433c59")
        build_path = self.create_build_path('compute-benchmarks-build')

        configure_command = [
            "cmake",
            f"-B {build_path}",
            f"-S {repo_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_SYCL=ON",
            f"-DSYCL_COMPILER_ROOT={options.sycl}",
            f"-DALLOW_WARNINGS=ON"
        ]
        run(configure_command, add_sycl=True)

        run(f"cmake --build {build_path} -j", add_sycl=True)
        self.benchmark_bin = f"{build_path}/bin/api_overhead_benchmark_sycl"

    def run_internal(self, ioq, env_vars):
        command = [
            f"{self.benchmark_bin}",
            "--test=SubmitKernel",
            f"--Ioq={ioq}",
            "--DiscardEvents=0",
            "--MeasureCompletion=0",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1",
            "--csv",
            "--noHeaders"
        ]
        result = self.run_bench(command, env_vars)
        (label, mean) = self.parse_output(result)
        return Result(label=label, value=mean, command=command, env=env_vars, stdout=result)

    def run(self, env_vars) -> list[Result]:
        results = []
        for ioq in [0, 1]:
            results.append(self.run_internal(ioq, env_vars))

        return results

    def parse_output(self, output):
        csv_file = io.StringIO(output)
        reader = csv.reader(csv_file)
        next(reader, None)
        data_row = next(reader, None)
        if data_row is None:
            raise ValueError("Benchmark output does not contain data.")
        try:
            label = data_row[0]
            mean = float(data_row[1])
            return (label, mean)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing output: {e}")

    def teardown(self):
        return
