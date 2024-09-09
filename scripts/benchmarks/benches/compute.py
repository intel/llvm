# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
import io
from utils.utils import run, git_clone, create_build_path
from .base import Benchmark
from .result import Result
from .options import options

class ComputeBench:
    def __init__(self, directory):
        self.directory = directory
        self.built = False
        return

    def setup(self):
        if self.built:
            return

        repo_path = git_clone(self.directory, "compute-benchmarks-repo", "https://github.com/intel/compute-benchmarks.git", "08c41bb8bc1762ad53c6194df6d36bfcceff4aa2")
        build_path = create_build_path(self.directory, 'compute-benchmarks-build')

        configure_command = [
            "cmake",
            f"-B {build_path}",
            f"-S {repo_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_SYCL=ON",
            f"-DSYCL_COMPILER_ROOT={options.sycl}",
            f"-DALLOW_WARNINGS=ON",
            f"-DBUILD_UR=ON",
            f"-DUR_BUILD_TESTS=OFF",
            f"-DUR_BUILD_TESTS=OFF",
            f"-DUMF_DISABLE_HWLOC=ON",
            f"-DBENCHMARK_UR_SOURCE_DIR={options.ur_dir}",
        ]
        run(configure_command, add_sycl=True)

        run(f"cmake --build {build_path} -j", add_sycl=True)

        self.built = True
        self.bins = os.path.join(build_path, 'bin')

class ComputeBenchmark(Benchmark):
    def __init__(self, bench, name, test):
        self.bench = bench
        self.bench_name = name
        self.test = test
        super().__init__(bench.directory)

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def unit(self):
        return "μs"

    def setup(self):
        self.bench.setup()
        self.benchmark_bin = os.path.join(self.bench.bins, self.bench_name)

    def run(self, env_vars) -> Result:
        command = [
            f"{self.benchmark_bin}",
            f"--test={self.test}",
            "--csv",
            "--noHeaders"
        ]

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        result = self.run_bench(command, env_vars)
        (label, mean) = self.parse_output(result)
        return Result(label=label, value=mean, command=command, env=env_vars, stdout=result, lower_is_better=self.lower_is_better())

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

class SubmitKernelSYCL(ComputeBenchmark):
    def __init__(self, bench, ioq):
        self.ioq = ioq
        super().__init__(bench, "api_overhead_benchmark_sycl", "SubmitKernel")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_sycl SubmitKernel {order}"

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            "--DiscardEvents=0",
            "--MeasureCompletion=0",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1"
        ]

class SubmitKernelUR(ComputeBenchmark):
    def __init__(self, bench, ioq):
        self.ioq = ioq
        super().__init__(bench, "api_overhead_benchmark_ur", "SubmitKernel")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_ur SubmitKernel {order}"

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            "--DiscardEvents=0",
            "--MeasureCompletion=0",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1"
        ]

class ExecImmediateCopyQueue(ComputeBenchmark):
    def __init__(self, bench, ioq, isCopyOnly, source, destination, size):
        self.ioq = ioq
        self.isCopyOnly = isCopyOnly
        self.source = source
        self.destination = destination
        self.size = size
        super().__init__(bench, "api_overhead_benchmark_sycl", "ExecImmediateCopyQueue")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_sycl ExecImmediateCopyQueue {order} from {self.source} to {self.destination}, size {self.size}"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=100000",
            f"--ioq={self.ioq}",
            f"--IsCopyOnly={self.isCopyOnly}",
            "--MeasureCompletionTime=0",
            f"--src={self.destination}",
            f"--dst={self.destination}",
            f"--size={self.size}"
        ]

class QueueInOrderMemcpy(ComputeBenchmark):
    def __init__(self, bench, isCopyOnly, source, destination, size):
        self.isCopyOnly = isCopyOnly
        self.source = source
        self.destination = destination
        self.size = size
        super().__init__(bench, "memory_benchmark_sycl", "QueueInOrderMemcpy")

    def name(self):
        return f"memory_benchmark_sycl QueueInOrderMemcpy from {self.source} to {self.destination}, size {self.size}"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--IsCopyOnly={self.isCopyOnly}",
            f"--sourcePlacement={self.source}",
            f"--destinationPlacement={self.destination}",
            f"--size={self.size}",
            "--count=100"
        ]

class QueueMemcpy(ComputeBenchmark):
    def __init__(self, bench, source, destination, size):
        self.source = source
        self.destination = destination
        self.size = size
        super().__init__(bench, "memory_benchmark_sycl", "QueueMemcpy")

    def name(self):
        return f"memory_benchmark_sycl QueueMemcpy from {self.source} to {self.destination}, size {self.size}"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--sourcePlacement={self.source}",
            f"--destinationPlacement={self.destination}",
            f"--size={self.size}",
        ]

class StreamMemory(ComputeBenchmark):
    def __init__(self, bench, type, size, placement):
        self.type = type
        self.size = size
        self.placement = placement
        super().__init__(bench, "memory_benchmark_sycl", "StreamMemory")

    def name(self):
        return f"memory_benchmark_sycl StreamMemory, placement {self.placement}, type {self.type}, size {self.size}"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--type={self.type}",
            f"--size={self.size}",
            f"--memoryPlacement={self.placement}",
            "--useEvents=0",
            "--contents=Zeros",
        ]

class VectorSum(ComputeBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "miscellaneous_benchmark_sycl", "VectorSum")

    def name(self):
        return f"miscellaneous_benchmark_sycl VectorSum"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=1000",
            "--numberOfElementsX=512",
            "--numberOfElementsY=256",
            "--numberOfElementsZ=256",
        ]
