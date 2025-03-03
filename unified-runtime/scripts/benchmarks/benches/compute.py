# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
import io
from utils.utils import run, git_clone, create_build_path
from .base import Benchmark, Suite
from .result import Result
from options import options
from enum import Enum

class ComputeBench(Suite):
    def __init__(self, directory):
        self.directory = directory

    def name(self) -> str:
        return "Compute Benchmarks"

    def setup(self):
        if options.sycl is None:
            return

        repo_path = git_clone(
            self.directory,
            "compute-benchmarks-repo",
            "https://github.com/intel/compute-benchmarks.git",
            "9369275026229b182bc4a555b73c2ec995a9e2b7",
        )
        build_path = create_build_path(self.directory, "compute-benchmarks-build")

        configure_command = [
            "cmake",
            f"-B {build_path}",
            f"-S {repo_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_SYCL=ON",
            f"-DSYCL_COMPILER_ROOT={options.sycl}",
            f"-DALLOW_WARNINGS=ON",
        ]

        if options.ur is not None:
            configure_command += [
                f"-DBUILD_UR=ON",
                f"-Dunified-runtime_DIR={options.ur}/lib/cmake/unified-runtime",
            ]

        print(f"{self.__class__.__name__}: Run {configure_command}")
        run(configure_command, add_sycl=True)
        print(f"{self.__class__.__name__}: Run cmake --build {build_path} -j")
        run(f"cmake --build {build_path} -j", add_sycl=True)

        self.built = True

    def benchmarks(self) -> list[Benchmark]:
        if options.sycl is None:
            return []

        if options.ur_adapter == "cuda":
            return []

        benches = [
            SubmitKernelL0(self, 0),
            SubmitKernelL0(self, 1),
            SubmitKernelSYCL(self, 0),
            SubmitKernelSYCL(self, 1),
            QueueInOrderMemcpy(self, 0, "Device", "Device", 1024),
            QueueInOrderMemcpy(self, 0, "Host", "Device", 1024),
            QueueMemcpy(self, "Device", "Device", 1024),
            StreamMemory(self, "Triad", 10 * 1024, "Device"),
            ExecImmediateCopyQueue(self, 0, 1, "Device", "Device", 1024),
            ExecImmediateCopyQueue(self, 1, 1, "Device", "Host", 1024),
            VectorSum(self),
            MemcpyExecute(self, 400, 1, 102400, 10, 1, 1, 1),
            MemcpyExecute(self, 100, 8, 102400, 10, 1, 1, 1),
            MemcpyExecute(self, 400, 8, 1024, 1000, 1, 1, 1),
            MemcpyExecute(self, 10, 16, 1024, 10000, 1, 1, 1),
            MemcpyExecute(self, 400, 1, 102400, 10, 0, 1, 1),
            MemcpyExecute(self, 100, 8, 102400, 10, 0, 1, 1),
            MemcpyExecute(self, 400, 8, 1024, 1000, 0, 1, 1),
            MemcpyExecute(self, 10, 16, 1024, 10000, 0, 1, 1),
            MemcpyExecute(self, 4096, 1, 1024, 10, 0, 1, 0),
            MemcpyExecute(self, 4096, 4, 1024, 10, 0, 1, 0),
            GraphApiSinKernelGraph(self, RUNTIMES.SYCL, 0, 5),
            GraphApiSinKernelGraph(self, RUNTIMES.SYCL, 1, 5),
            GraphApiSinKernelGraph(self, RUNTIMES.SYCL, 0, 100),
            GraphApiSinKernelGraph(self, RUNTIMES.SYCL, 1, 100),
            GraphApiSinKernelGraph(self, RUNTIMES.LEVEL_ZERO, 0, 5),
            GraphApiSinKernelGraph(self, RUNTIMES.LEVEL_ZERO, 1, 5),
            GraphApiSinKernelGraph(self, RUNTIMES.LEVEL_ZERO, 0, 100),
            GraphApiSinKernelGraph(self, RUNTIMES.LEVEL_ZERO, 1, 100),
        ]

        if options.ur is not None:
            benches += [
                SubmitKernelUR(self, 0, 0),
                SubmitKernelUR(self, 1, 0),
                SubmitKernelUR(self, 1, 1),
                GraphApiSinKernelGraph(self, RUNTIMES.UR, 0, 5),
                GraphApiSinKernelGraph(self, RUNTIMES.UR, 1, 5),
                GraphApiSinKernelGraph(self, RUNTIMES.UR, 0, 100),
                GraphApiSinKernelGraph(self, RUNTIMES.UR, 1, 100),
            ]

        return benches


def parse_unit_type(compute_unit):
    if "[count]" in compute_unit:
        return "instr"
    elif "[us]" in compute_unit:
        return "μs"
    return compute_unit.replace("[", "").replace("]", "")


class ComputeBenchmark(Benchmark):
    def __init__(self, bench, name, test):
        super().__init__(bench.directory, bench)
        self.bench = bench
        self.bench_name = name
        self.test = test

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def setup(self):
        self.benchmark_bin = os.path.join(
            self.bench.directory, "compute-benchmarks-build", "bin", self.bench_name
        )

    def explicit_group(self):
        return ""

    def run(self, env_vars) -> list[Result]:
        command = [
            f"{self.benchmark_bin}",
            f"--test={self.test}",
            "--csv",
            "--noHeaders",
        ]

        command += self.bin_args()
        env_vars.update(self.extra_env_vars())

        result = self.run_bench(command, env_vars)
        parsed_results = self.parse_output(result)
        ret = []
        for label, median, stddev, unit in parsed_results:
            extra_label = " CPU count" if parse_unit_type(unit) == "instr" else ""
            explicit_group = (
                self.explicit_group() + extra_label
                if self.explicit_group() != ""
                else ""
            )
            ret.append(
                Result(
                    label=self.name() + extra_label,
                    explicit_group=explicit_group,
                    value=median,
                    stddev=stddev,
                    command=command,
                    env=env_vars,
                    stdout=result,
                    unit=parse_unit_type(unit),
                )
            )
        return ret

    def parse_output(self, output):
        csv_file = io.StringIO(output)
        reader = csv.reader(csv_file)
        next(reader, None)
        results = []
        while True:
            data_row = next(reader, None)
            if data_row is None:
                break
            try:
                label = data_row[0]
                mean = float(data_row[1])
                median = float(data_row[2])
                # compute benchmarks report stddev as %
                stddev = mean * (float(data_row[3].strip("%")) / 100.0)
                unit = data_row[7]
                results.append((label, median, stddev, unit))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing output: {e}")
        if len(results) == 0:
            raise ValueError("Benchmark output does not contain data.")
        return results

    def teardown(self):
        return


class SubmitKernelSYCL(ComputeBenchmark):
    def __init__(self, bench, ioq):
        self.ioq = ioq
        super().__init__(bench, "api_overhead_benchmark_sycl", "SubmitKernel")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_sycl SubmitKernel {order}"

    def explicit_group(self):
        return "SubmitKernel"

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            "--DiscardEvents=0",
            "--MeasureCompletion=0",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1",
        ]


class SubmitKernelUR(ComputeBenchmark):
    def __init__(self, bench, ioq, measureCompletion):
        self.ioq = ioq
        self.measureCompletion = measureCompletion
        super().__init__(bench, "api_overhead_benchmark_ur", "SubmitKernel")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_ur SubmitKernel {order}" + (
            " with measure completion" if self.measureCompletion else ""
        )

    def explicit_group(self):
        return "SubmitKernel"

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            "--DiscardEvents=0",
            f"--MeasureCompletion={self.measureCompletion}",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1",
        ]


class SubmitKernelL0(ComputeBenchmark):
    def __init__(self, bench, ioq):
        self.ioq = ioq
        super().__init__(bench, "api_overhead_benchmark_l0", "SubmitKernel")

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_l0 SubmitKernel {order}"

    def explicit_group(self):
        return "SubmitKernel"

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            "--DiscardEvents=0",
            "--MeasureCompletion=0",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1",
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
            f"--size={self.size}",
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
            "--count=100",
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

    # measurement is in GB/s
    def lower_is_better(self):
        return False

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--type={self.type}",
            f"--size={self.size}",
            f"--memoryPlacement={self.placement}",
            "--useEvents=0",
            "--contents=Zeros",
            "--multiplier=1",
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


class MemcpyExecute(ComputeBenchmark):
    def __init__(
        self,
        bench,
        numOpsPerThread,
        numThreads,
        allocSize,
        iterations,
        srcUSM,
        dstUSM,
        useEvent,
    ):
        self.numOpsPerThread = numOpsPerThread
        self.numThreads = numThreads
        self.allocSize = allocSize
        self.iterations = iterations
        self.srcUSM = srcUSM
        self.dstUSM = dstUSM
        self.useEvents = useEvent
        super().__init__(bench, "multithread_benchmark_ur", "MemcpyExecute")

    def name(self):
        return (
            f"multithread_benchmark_ur MemcpyExecute opsPerThread:{self.numOpsPerThread}, numThreads:{self.numThreads}, allocSize:{self.allocSize} srcUSM:{self.srcUSM} dstUSM:{self.dstUSM}"
            + (" without events" if not self.useEvents else "")
        )

    def bin_args(self) -> list[str]:
        return [
            "--Ioq=1",
            f"--UseEvents={self.useEvents}",
            "--MeasureCompletion=1",
            "--UseQueuePerThread=1",
            f"--AllocSize={self.allocSize}",
            f"--NumThreads={self.numThreads}",
            f"--NumOpsPerThread={self.numOpsPerThread}",
            f"--iterations={self.iterations}",
            f"--SrcUSM={self.srcUSM}",
            f"--DstUSM={self.dstUSM}",
        ]


class RUNTIMES(Enum):
    SYCL = "sycl"
    LEVEL_ZERO = "l0"
    UR = "ur"


class GraphApiSinKernelGraph(ComputeBenchmark):
    def __init__(self, bench, runtime: RUNTIMES, withGraphs, numKernels):
        self.withGraphs = withGraphs
        self.numKernels = numKernels
        self.runtime = runtime
        super().__init__(
            bench, f"graph_api_benchmark_{runtime.value}", "SinKernelGraph"
        )

    def explicit_group(self):
        return f"SinKernelGraph {self.numKernels}"

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} SinKernelGraph graphs:{self.withGraphs}, numKernels:{self.numKernels}"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--numKernels={self.numKernels}",
            f"--withGraphs={self.withGraphs}",
            "--withCopyOffload=1",
            "--immediateAppendCmdList=0",
        ]


class GraphApiSubmitExecGraph(ComputeBenchmark):
    def __init__(self, bench, ioq, submit, numKernels):
        self.ioq = ioq
        self.submit = submit
        self.numKernels = numKernels
        super().__init__(bench, "graph_api_benchmark_sycl", "SubmitExecGraph")

    def name(self):
        return f"graph_api_benchmark_sycl SubmitExecGraph ioq:{self.ioq}, submit:{self.submit}, numKernels:{self.numKernels}"

    def explicit_group(self):
        if self.submit:
            return "SubmitGraph"
        else:
            return "ExecGraph"

    def bin_args(self) -> list[str]:
        return [
            "--iterations=100",
            f"--measureSubmit={self.submit}",
            f"--ioq={self.ioq}",
            f"--numKernels={self.numKernels}",
        ]
