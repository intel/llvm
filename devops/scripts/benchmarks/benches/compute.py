# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import csv
import io
from utils.utils import run, git_clone, create_build_path
from .base import Benchmark, Suite
from utils.result import BenchmarkMetadata, Result
from options import options
from enum import Enum


class RUNTIMES(Enum):
    SYCL_PREVIEW = "syclpreview"
    SYCL = "sycl"
    LEVEL_ZERO = "l0"
    UR = "ur"


def runtime_to_name(runtime: RUNTIMES) -> str:
    return {
        RUNTIMES.SYCL_PREVIEW: "SYCL Preview",
        RUNTIMES.SYCL: "SYCL",
        RUNTIMES.LEVEL_ZERO: "Level Zero",
        RUNTIMES.UR: "Unified Runtime",
    }[runtime]


def runtime_to_tag_name(runtime: RUNTIMES) -> str:
    return {
        RUNTIMES.SYCL_PREVIEW: "SYCL",
        RUNTIMES.SYCL: "SYCL",
        RUNTIMES.LEVEL_ZERO: "L0",
        RUNTIMES.UR: "UR",
    }[runtime]


class ComputeBench(Suite):
    def __init__(self, directory):
        self.directory = directory

    def name(self) -> str:
        return "Compute Benchmarks"

    def git_url(self) -> str:
        return "https://github.com/intel/compute-benchmarks.git"

    def git_hash(self) -> str:
        return "015019f815f73e308171145a48b63d57c1263e45"

    def setup(self):
        if options.sycl is None:
            return

        repo_path = git_clone(
            self.directory,
            "compute-benchmarks-repo",
            self.git_url(),
            self.git_hash(),
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

        if options.ur_adapter == "cuda":
            configure_command += ["-DBUILD_SYCL_WITH_CUDA=ON"]

        if options.ur is not None:
            configure_command += [
                f"-DBUILD_UR=ON",
                f"-Dunified-runtime_DIR={options.ur}/lib/cmake/unified-runtime",
            ]

        run(configure_command, add_sycl=True)

        run(f"cmake --build {build_path} -j {options.build_jobs}", add_sycl=True)

        self.built = True

    def additionalMetadata(self) -> dict[str, BenchmarkMetadata]:
        return {
            "SubmitKernel": BenchmarkMetadata(
                type="group",
                description="Measures CPU time overhead of submitting kernels through different APIs.",
                notes="Each layer builds on top of the previous layer, adding functionality and overhead.\n"
                "The first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\n"
                "The UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\n"
                "Work is ongoing to reduce the overhead of the SYCL API\n",
                tags=["submit", "micro", "SYCL", "UR", "L0"],
                range_min=0.0,
            ),
            "SinKernelGraph": BenchmarkMetadata(
                type="group",
                unstable="This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
                tags=["submit", "memory", "proxy", "SYCL", "UR", "L0", "graph"],
            ),
            "SubmitGraph": BenchmarkMetadata(
                type="group", tags=["submit", "micro", "SYCL", "UR", "L0", "graph"]
            ),
        }

    def enabled_runtimes(self, supported_runtimes=None, extra_runtimes=None):
        # all runtimes in the RUNTIMES enum
        runtimes = supported_runtimes or list(RUNTIMES)

        # filter out SYCL_PREVIEW which is not supported by default in all benchmarks
        runtimes = [r for r in runtimes if r != RUNTIMES.SYCL_PREVIEW]

        if extra_runtimes is not None:
            runtimes.extend(extra_runtimes)

        # Filter out UR if not available
        if options.ur is None:
            runtimes = [r for r in runtimes if r != RUNTIMES.UR]

        # Filter out L0 if cuda backend
        if options.ur_adapter == "cuda":
            runtimes = [r for r in runtimes if r != RUNTIMES.LEVEL_ZERO]

        return runtimes

    def benchmarks(self) -> list[Benchmark]:
        if options.sycl is None:
            return []

        if options.ur_adapter == "hip":
            return []

        benches = []

        # Add SubmitKernel benchmarks using loops
        for runtime in self.enabled_runtimes(extra_runtimes=[RUNTIMES.SYCL_PREVIEW]):
            for in_order_queue in [0, 1]:
                for measure_completion in [0, 1]:
                    for use_events in [0, 1]:
                        benches.append(
                            SubmitKernel(
                                self,
                                runtime,
                                in_order_queue,
                                measure_completion,
                                use_events,
                            )
                        )

        # Add SinKernelGraph benchmarks
        for runtime in self.enabled_runtimes():
            for with_graphs in [0, 1]:
                for num_kernels in [5, 100]:
                    benches.append(
                        GraphApiSinKernelGraph(self, runtime, with_graphs, num_kernels)
                    )

        # Add ULLS benchmarks
        for runtime in self.enabled_runtimes([RUNTIMES.SYCL, RUNTIMES.LEVEL_ZERO]):
            benches.append(UllsEmptyKernel(self, runtime, 1000, 256))
            benches.append(UllsKernelSwitch(self, runtime, 8, 200, 0, 0, 1, 1))

        # Add GraphApiSubmitGraph benchmarks
        for runtime in self.enabled_runtimes():
            for in_order_queue in [0, 1]:
                for num_kernels in [4, 10, 32]:
                    for measure_completion_time in [0, 1]:
                        benches.append(
                            GraphApiSubmitGraph(
                                self,
                                runtime,
                                in_order_queue,
                                num_kernels,
                                measure_completion_time,
                            )
                        )

        # Add other benchmarks
        benches += [
            QueueInOrderMemcpy(self, 0, "Device", "Device", 1024),
            QueueInOrderMemcpy(self, 0, "Host", "Device", 1024),
            QueueMemcpy(self, "Device", "Device", 1024),
            StreamMemory(self, "Triad", 10 * 1024, "Device"),
            ExecImmediateCopyQueue(self, 0, 1, "Device", "Device", 1024),
            ExecImmediateCopyQueue(self, 1, 1, "Device", "Host", 1024),
            VectorSum(self),
        ]

        # Add UR-specific benchmarks
        if options.ur is not None:
            benches += [
                MemcpyExecute(self, 400, 1, 102400, 10, 1, 1, 1, 1),
                MemcpyExecute(self, 400, 1, 102400, 10, 0, 1, 1, 1),
                MemcpyExecute(self, 100, 4, 102400, 10, 1, 1, 0, 1),
                MemcpyExecute(self, 100, 4, 102400, 10, 1, 1, 0, 0),
                MemcpyExecute(self, 4096, 4, 1024, 10, 0, 1, 0, 1),
                UsmMemoryAllocation(self, RUNTIMES.UR, "Device", 256, "Both"),
                UsmMemoryAllocation(self, RUNTIMES.UR, "Device", 256 * 1024, "Both"),
                UsmBatchMemoryAllocation(self, RUNTIMES.UR, "Device", 128, 256, "Both"),
                UsmBatchMemoryAllocation(
                    self, RUNTIMES.UR, "Device", 128, 16 * 1024, "Both"
                ),
                UsmBatchMemoryAllocation(
                    self, RUNTIMES.UR, "Device", 128, 128 * 1024, "Both"
                ),
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

    def description(self) -> str:
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
                    git_url=self.bench.git_url(),
                    git_hash=self.bench.git_hash(),
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


class SubmitKernel(ComputeBenchmark):
    def __init__(self, bench, runtime: RUNTIMES, ioq, MeasureCompletion=0, UseEvents=0):
        self.ioq = ioq
        self.runtime = runtime
        self.MeasureCompletion = MeasureCompletion
        self.UseEvents = UseEvents
        super().__init__(
            bench, f"api_overhead_benchmark_{runtime.value}", "SubmitKernel"
        )

    def get_tags(self):
        return ["submit", "latency", runtime_to_tag_name(self.runtime), "micro"]

    def name(self):
        order = "in order" if self.ioq else "out of order"
        completion_str = " with measure completion" if self.MeasureCompletion else ""

        # this needs to be inversed (i.e., using events is empty string)
        # to match the existing already stored results
        events_str = " not using events" if not self.UseEvents else ""

        return f"api_overhead_benchmark_{self.runtime.value} SubmitKernel {order}{completion_str}{events_str}"

    def explicit_group(self):
        order = "In Order" if self.ioq else "Out Of Order"
        completion_str = " With Completion" if self.MeasureCompletion else ""

        # this needs to be inversed (i.e., using events is empty string)
        # to match the existing already stored results
        events_str = " not using events" if not self.UseEvents else ""

        return f"SubmitKernel {order}{completion_str}{events_str}"

    def description(self) -> str:
        order = "in-order" if self.ioq else "out-of-order"
        runtime_name = runtime_to_name(self.runtime)

        completion_desc = completion_desc = (
            f", {'including' if self.MeasureCompletion else 'excluding'} kernel completion time"
        )

        return (
            f"Measures CPU time overhead of submitting {order} kernels through {runtime_name} API{completion_desc}. "
            f"Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time."
        )

    def range(self) -> tuple[float, float]:
        return (0.0, None)

    def bin_args(self) -> list[str]:
        return [
            f"--Ioq={self.ioq}",
            f"--MeasureCompletion={self.MeasureCompletion}",
            "--iterations=100000",
            "--Profiling=0",
            "--NumKernels=10",
            "--KernelExecTime=1",
            f"--UseEvents={self.UseEvents}",
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

    def description(self) -> str:
        order = "in-order" if self.ioq else "out-of-order"
        operation = "copy-only" if self.isCopyOnly else "copy and command submission"
        return (
            f"Measures SYCL {order} queue overhead for {operation} from {self.source} to "
            f"{self.destination} memory with {self.size} bytes. Tests immediate execution overheads."
        )

    def get_tags(self):
        return ["memory", "submit", "latency", "SYCL", "micro"]

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

    def description(self) -> str:
        operation = "copy-only" if self.isCopyOnly else "copy and command submission"
        return (
            f"Measures SYCL in-order queue memory copy performance for {operation} from "
            f"{self.source} to {self.destination} with {self.size} bytes, executed 100 times per iteration."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

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

    def description(self) -> str:
        return (
            f"Measures general SYCL queue memory copy performance from {self.source} to "
            f"{self.destination} with {self.size} bytes per operation."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

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

    def description(self) -> str:
        return (
            f"Measures {self.placement} memory bandwidth using {self.type} pattern with "
            f"{self.size} bytes. Higher values (GB/s) indicate better performance."
        )

    # measurement is in GB/s
    def lower_is_better(self):
        return False

    def get_tags(self):
        return ["memory", "throughput", "SYCL", "micro"]

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--type={self.type}",
            f"--size={self.size}",
            f"--memoryPlacement={self.placement}",
            "--useEvents=0",
            "--contents=Zeros",
            "--multiplier=1",
            "--vectorSize=1",
        ]


class VectorSum(ComputeBenchmark):
    def __init__(self, bench):
        super().__init__(bench, "miscellaneous_benchmark_sycl", "VectorSum")

    def name(self):
        return f"miscellaneous_benchmark_sycl VectorSum"

    def description(self) -> str:
        return (
            "Measures performance of vector addition across 3D grid (512x256x256 elements) "
            "using SYCL."
        )

    def get_tags(self):
        return ["math", "throughput", "SYCL", "micro"]

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
        useCopyOffload,
    ):
        self.numOpsPerThread = numOpsPerThread
        self.numThreads = numThreads
        self.allocSize = allocSize
        self.iterations = iterations
        self.srcUSM = srcUSM
        self.dstUSM = dstUSM
        self.useEvents = useEvent
        self.useCopyOffload = useCopyOffload
        super().__init__(bench, "multithread_benchmark_ur", "MemcpyExecute")

    def extra_env_vars(self) -> dict:
        if not self.useCopyOffload:
            return {"UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD": "1"}
        else:
            return {}

    def name(self):
        return (
            f"multithread_benchmark_ur MemcpyExecute opsPerThread:{self.numOpsPerThread}, numThreads:{self.numThreads}, allocSize:{self.allocSize} srcUSM:{self.srcUSM} dstUSM:{self.dstUSM}"
            + (" without events" if not self.useEvents else "")
            + (" without copy offload" if not self.useCopyOffload else "")
        )

    def explicit_group(self):
        return (
            "MemcpyExecute opsPerThread: "
            + str(self.numOpsPerThread)
            + " numThreads: "
            + str(self.numThreads)
        )

    def description(self) -> str:
        src_type = "device" if self.srcUSM == 1 else "host"
        dst_type = "device" if self.dstUSM == 1 else "host"
        events = "with" if self.useEvents else "without"
        copy_offload = "with" if self.useCopyOffload else "without"
        return (
            f"Measures multithreaded memory copy performance with {self.numThreads} threads "
            f"each performing {self.numOpsPerThread} operations on {self.allocSize} bytes "
            f"from {src_type} to {dst_type} memory {events} events {copy_offload} driver copy offload."
        )

    def get_tags(self):
        return ["memory", "latency", "UR", "micro"]

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

    def description(self) -> str:
        execution = "using graphs" if self.withGraphs else "without graphs"
        return (
            f"Measures {self.runtime.value.upper()} performance when executing {self.numKernels} "
            f"sin kernels {execution}. Tests overhead and benefits of graph-based execution."
        )

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} SinKernelGraph graphs:{self.withGraphs}, numKernels:{self.numKernels}"

    def unstable(self) -> str:
        return "This benchmark combines both eager and graph execution, and may not be representative of real use cases."

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self.runtime),
            "proxy",
            "submit",
            "memory",
            "latency",
        ]

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--numKernels={self.numKernels}",
            f"--withGraphs={self.withGraphs}",
            "--withCopyOffload=1",
            "--immediateAppendCmdList=0",
        ]


# TODO: once L0 SubmitGraph exists, this needs to be cleaned up split benchmarks into more groups,
# set all the parameters (UseEvents 0/1) and
# unify the benchmark naming scheme with SubmitKernel.
class GraphApiSubmitGraph(ComputeBenchmark):
    def __init__(
        self, bench, runtime: RUNTIMES, inOrderQueue, numKernels, measureCompletionTime
    ):
        self.inOrderQueue = inOrderQueue
        self.numKernels = numKernels
        self.runtime = runtime
        self.measureCompletionTime = measureCompletionTime
        super().__init__(bench, f"graph_api_benchmark_{runtime.value}", "SubmitGraph")

    def explicit_group(self):
        return f"SubmitGraph {self.numKernels}"

    def description(self) -> str:
        return (
            f"Measures {self.runtime.value.upper()} performance when executing {self.numKernels} "
            f"trivial kernels using graphs. Tests overhead and benefits of graph-based execution."
        )

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} SubmitGraph numKernels:{self.numKernels} ioq {self.inOrderQueue} measureCompletion {self.measureCompletionTime}"

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self.runtime),
            "micro",
            "submit",
            "latency",
        ]

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--NumKernels={self.numKernels}",
            f"--MeasureCompletionTime={self.measureCompletionTime}",
            f"--InOrderQueue={self.inOrderQueue}",
            "--Profiling=0",
            "--KernelExecutionTime=1",
            "--UseEvents=0",  # not all implementations support UseEvents=1
            "--UseExplicit=0",
        ]


class UllsEmptyKernel(ComputeBenchmark):
    def __init__(self, bench, runtime: RUNTIMES, wgc, wgs):
        self.wgc = wgc
        self.wgs = wgs
        self.runtime = runtime
        super().__init__(bench, f"ulls_benchmark_{runtime.value}", "EmptyKernel")

    def explicit_group(self):
        return f"EmptyKernel {self.wgc} {self.wgs}"

    def description(self) -> str:
        return ""

    def name(self):
        return f"ulls_benchmark_{self.runtime.value} EmptyKernel wgc:{self.wgc}, wgs:{self.wgs}"

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "submit"]

    def bin_args(self) -> list[str]:
        return [
            "--iterations=10000",
            f"--wgs={self.wgs}",
            f"--wgc={self.wgs}",
        ]


class UllsKernelSwitch(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        count,
        kernelTime,
        barrier,
        hostVisible,
        ioq,
        ctrBasedEvents,
    ):
        self.count = count
        self.kernelTime = kernelTime
        self.barrier = barrier
        self.hostVisible = hostVisible
        self.ctrBasedEvents = ctrBasedEvents
        self.runtime = runtime
        self.ioq = ioq
        super().__init__(bench, f"ulls_benchmark_{runtime.value}", "KernelSwitch")

    def explicit_group(self):
        return f"KernelSwitch {self.count} {self.kernelTime}"

    def description(self) -> str:
        return ""

    def name(self):
        return f"ulls_benchmark_{self.runtime.value} KernelSwitch count {self.count} kernelTime {self.kernelTime}"

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "submit"]

    def bin_args(self) -> list[str]:
        return [
            "--iterations=1000",
            f"--count={self.count}",
            f"--kernelTime={self.kernelTime}",
            f"--barrier={self.barrier}",
            f"--hostVisible={self.hostVisible}",
            f"--ioq={self.ioq}",
            f"--ctrBasedEvents={self.ctrBasedEvents}",
        ]


class UsmMemoryAllocation(ComputeBenchmark):
    def __init__(
        self, bench, runtime: RUNTIMES, usm_memory_placement, size, measure_mode
    ):
        self.runtime = runtime
        self.usm_memory_placement = usm_memory_placement
        self.size = size
        self.measure_mode = measure_mode
        super().__init__(
            bench, f"api_overhead_benchmark_{runtime.value}", "UsmMemoryAllocation"
        )

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "memory"]

    def name(self):
        return (
            f"api_overhead_benchmark_{self.runtime.value} UsmMemoryAllocation "
            f"usmMemoryPlacement:{self.usm_memory_placement} size:{self.size} measureMode:{self.measure_mode}"
        )

    def explicit_group(self):
        return f"UsmMemoryAllocation"

    def description(self) -> str:
        what_is_measured = "Both memory allocation and memory free are timed"
        if self.measure_mode == "Allocate":
            what_is_measured = "Only memory allocation is timed"
        elif self.measure_mode == "Free":
            what_is_measured = "Only memory free is timed"
        return (
            f"Measures memory allocation overhead by allocating {self.size} bytes of "
            f"usm {self.usm_memory_placement} memory and free'ing it immediately. "
            f"{what_is_measured}. "
        )

    def bin_args(self) -> list[str]:
        return [
            f"--type={self.usm_memory_placement}",
            f"--size={self.size}",
            f"--measureMode={self.measure_mode}",
            "--iterations=10000",
        ]


class UsmBatchMemoryAllocation(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        usm_memory_placement,
        allocation_count,
        size,
        measure_mode,
    ):
        self.runtime = runtime
        self.usm_memory_placement = usm_memory_placement
        self.allocation_count = allocation_count
        self.size = size
        self.measure_mode = measure_mode
        super().__init__(
            bench, f"api_overhead_benchmark_{runtime.value}", "UsmBatchMemoryAllocation"
        )

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "memory"]

    def name(self):
        return (
            f"api_overhead_benchmark_{self.runtime.value} UsmBatchMemoryAllocation "
            f"usmMemoryPlacement:{self.usm_memory_placement} allocationCount:{self.allocation_count} size:{self.size} measureMode:{self.measure_mode}"
        )

    def explicit_group(self):
        return f"UsmBatchMemoryAllocation"

    def description(self) -> str:
        what_is_measured = "Both memory allocation and memory free are timed"
        if self.measure_mode == "Allocate":
            what_is_measured = "Only memory allocation is timed"
        elif self.measure_mode == "Free":
            what_is_measured = "Only memory free is timed"
        return (
            f"Measures memory allocation overhead by allocating {self.size} bytes of "
            f"usm {self.usm_memory_placement} memory {self.allocation_count} times, then free'ing it all at once. "
            f"{what_is_measured}. "
        )

    def bin_args(self) -> list[str]:
        return [
            f"--type={self.usm_memory_placement}",
            f"--allocationCount={self.allocation_count}",
            f"--size={self.size}",
            f"--measureMode={self.measure_mode}",
            "--iterations=1000",
        ]
