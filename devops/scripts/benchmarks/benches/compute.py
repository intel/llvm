# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import csv
import io
import math
from enum import Enum
from itertools import product
from pathlib import Path

from git_project import GitProject
from options import options
from utils.result import BenchmarkMetadata, Result
from utils.logger import log

from .base import Benchmark, Suite, TracingType
from .compute_metadata import ComputeMetadataGenerator


class RUNTIMES(Enum):
    SYCL_PREVIEW = "syclpreview"
    SYCL = "sycl"
    LEVEL_ZERO = "l0"
    UR = "ur"


class PROFILERS(Enum):
    TIMER = "timer"
    CPU_COUNTER = "cpuCounter"


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
    def __init__(self):
        self.submit_graph_num_kernels = [4, 10, 32]
        self.project = None

    def name(self) -> str:
        return "Compute Benchmarks"

    def git_url(self) -> str:
        return "https://github.com/intel/compute-benchmarks.git"

    def git_hash(self) -> str:
        # Oct 31, 2025
        return "1d4f68f82a5fe8c404aa1126615da4a1b789e254"

    def setup(self) -> None:
        if options.sycl is None:
            return

        if self.project is None:
            self.project = GitProject(
                self.git_url(),
                self.git_hash(),
                Path(options.workdir),
                "compute-benchmarks",
                use_installdir=False,
            )

        if not self.project.needs_rebuild():
            log.info(f"Rebuilding {self.project.name} skipped")
            return

        extra_args = [
            f"-DBUILD_SYCL=ON",
            f"-DSYCL_COMPILER_ROOT={options.sycl}",
            f"-DALLOW_WARNINGS=ON",
            f"-DUSE_SYSTEM_LEVEL_ZERO=OFF",
            f"-DCMAKE_CXX_COMPILER=clang++",
            f"-DCMAKE_C_COMPILER=clang",
        ]
        if options.ur_adapter == "cuda":
            extra_args += [
                "-DBUILD_SYCL_WITH_CUDA=ON",
                "-DBUILD_L0=OFF",
                "-DBUILD_OCL=OFF",
            ]
        if options.ur is not None:
            extra_args += [
                f"-DBUILD_UR=ON",
                f"-Dunified-runtime_DIR={options.ur}/lib/cmake/unified-runtime",
            ]

        self.project.configure(extra_args, add_sycl=True)
        self.project.build(add_sycl=True)

    def additional_metadata(self) -> dict[str, BenchmarkMetadata]:
        """
        Returns:
            Dictionary mapping group names to their metadata
        """
        # Generate metadata based on actual benchmark instances
        generator = ComputeMetadataGenerator()
        benchmarks = self.benchmarks()
        return generator.generate_metadata_from_benchmarks(benchmarks)

    def benchmarks(self) -> list[Benchmark]:
        """
        Returns:
            List of all possible benchmark instances
        """
        benches = []

        # hand-picked value so that total execution time of the benchmark is
        # similar on all architectures
        long_kernel_exec_time_ioq = [20]
        # For BMG server, a new value 200 is used, but we have to create metadata
        # for both values to keep the dashboard consistent.
        # See SubmitKernel.enabled()
        long_kernel_exec_time_ooo = [20, 200]

        submit_kernel_params = product(
            list(RUNTIMES),
            [0, 1],  # in_order_queue
            [0, 1],  # measure_completion
            [0, 1],  # use_events
        )
        for (
            runtime,
            in_order_queue,
            measure_completion,
            use_events,
        ) in submit_kernel_params:
            long_kernel_exec_time = (
                long_kernel_exec_time_ioq
                if in_order_queue
                else long_kernel_exec_time_ooo
            )
            for kernel_exec_time in [1, *long_kernel_exec_time]:
                benches.append(
                    SubmitKernel(
                        self,
                        runtime,
                        in_order_queue,
                        measure_completion,
                        use_events,
                        kernel_exec_time,
                    )
                )
                if runtime in (RUNTIMES.SYCL, RUNTIMES.SYCL_PREVIEW, RUNTIMES.UR):
                    # Create CPU count variant
                    benches.append(
                        SubmitKernel(
                            self,
                            runtime,
                            in_order_queue,
                            measure_completion,
                            use_events,
                            kernel_exec_time,
                            profiler_type=PROFILERS.CPU_COUNTER,
                        )
                    )

        # Add SinKernelGraph benchmarks
        sin_kernel_graph_params = product(
            list(RUNTIMES),
            [0, 1],  # with_graphs
            [5, 100],  # num_kernels
        )
        for runtime, with_graphs, num_kernels in sin_kernel_graph_params:
            benches.append(
                GraphApiSinKernelGraph(self, runtime, with_graphs, num_kernels)
            )

            # Add ULLS benchmarks
        for runtime in list(RUNTIMES):
            if runtime == RUNTIMES.SYCL:
                benches.append(
                    UllsEmptyKernel(
                        self, runtime, 1000, 256, profiler_type=PROFILERS.CPU_COUNTER
                    )
                )
            benches.append(UllsEmptyKernel(self, runtime, 1000, 256))
            benches.append(UllsKernelSwitch(self, runtime, 8, 200, 0, 0, 1, 1))

        # Add GraphApiSubmitGraph benchmarks
        submit_graph_params = product(
            list(RUNTIMES),
            [0, 1],  # in_order_queue
            self.submit_graph_num_kernels,
            [0, 1],  # measure_completion_time
            [0, 1],  # use_events
        )
        for (
            runtime,
            in_order_queue,
            num_kernels,
            measure_completion_time,
            use_events,
        ) in submit_graph_params:
            # Non-sycl runtimes have to be run with emulated graphs,
            # see: https://github.com/intel/compute-benchmarks/commit/d81d5d602739482b9070c872a28c0b5ebb41de70
            emulate_graphs = (
                0 if runtime in (RUNTIMES.SYCL, RUNTIMES.SYCL_PREVIEW) else 1
            )
            benches.append(
                GraphApiSubmitGraph(
                    self,
                    runtime,
                    in_order_queue,
                    num_kernels,
                    measure_completion_time,
                    use_events,
                    emulate_graphs,
                    useHostTasks=0,
                )
            )
            if runtime == RUNTIMES.SYCL:
                # Create CPU count variant
                benches.append(
                    GraphApiSubmitGraph(
                        self,
                        runtime,
                        in_order_queue,
                        num_kernels,
                        measure_completion_time,
                        use_events,
                        emulate_graphs,
                        useHostTasks=0,
                        profiler_type=PROFILERS.CPU_COUNTER,
                    )
                )

        # Add other benchmarks
        benches += [
            StreamMemory(self, "Triad", 10 * 1024, "Device"),
            VectorSum(self),
            GraphApiFinalizeGraph(self, RUNTIMES.SYCL, 0, "Gromacs"),
            GraphApiFinalizeGraph(self, RUNTIMES.SYCL, 1, "Gromacs"),
            GraphApiFinalizeGraph(self, RUNTIMES.SYCL, 0, "Llama"),
            GraphApiFinalizeGraph(self, RUNTIMES.SYCL, 1, "Llama"),
        ]
        for profiler_type in list(PROFILERS):
            benches.append(
                QueueInOrderMemcpy(self, 0, "Device", "Device", 1024, profiler_type)
            )
            benches.append(
                QueueInOrderMemcpy(self, 0, "Host", "Device", 1024, profiler_type)
            )
            benches.append(QueueMemcpy(self, "Device", "Device", 1024, profiler_type))
            benches.append(
                ExecImmediateCopyQueue(
                    self, 0, 1, "Device", "Device", 1024, profiler_type
                )
            )
            benches.append(
                ExecImmediateCopyQueue(
                    self, 1, 1, "Device", "Host", 1024, profiler_type
                )
            )

        # Add UR-specific benchmarks
        benches += [
            # TODO: multithread_benchmark_ur fails with segfault
            # MemcpyExecute(self, RUNTIMES.UR, 400, 1, 102400, 10, 1, 1, 1, 1, 0),
            # MemcpyExecute(self, RUNTIMES.UR, 400, 1, 102400, 10, 0, 1, 1, 1, 0),
            # MemcpyExecute(self, RUNTIMES.UR, 100, 4, 102400, 10, 1, 1, 0, 1, 0),
            # MemcpyExecute(self, RUNTIMES.UR, 100, 4, 102400, 10, 1, 1, 0, 0, 0),
            # MemcpyExecute(self, RUNTIMES.UR, 4096, 4, 1024, 10, 0, 1, 0, 1, 0),
            # MemcpyExecute(self, RUNTIMES.UR, 4096, 4, 1024, 10, 0, 1, 0, 1, 1),
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

        benches += [
            MemcpyExecute(
                self, RUNTIMES.SYCL_PREVIEW, 4096, 1, 1024, 40, 1, 1, 0, 1, 0
            ),
            MemcpyExecute(
                self, RUNTIMES.SYCL_PREVIEW, 4096, 1, 1024, 40, 1, 1, 0, 1, 1
            ),
            MemcpyExecute(
                self, RUNTIMES.SYCL_PREVIEW, 4096, 4, 1024, 10, 1, 1, 0, 1, 0
            ),
            MemcpyExecute(
                self, RUNTIMES.SYCL_PREVIEW, 4096, 4, 1024, 10, 1, 1, 0, 1, 1
            ),
        ]

        return benches


class ComputeBenchmark(Benchmark):
    def __init__(
        self,
        suite: ComputeBench,
        name: str,
        test: str,
        runtime: RUNTIMES = None,
        profiler_type: PROFILERS = PROFILERS.TIMER,
    ):
        super().__init__(suite)
        self.suite = suite
        self.bench_name = name
        self.test = test
        self.runtime = runtime
        self.profiler_type = profiler_type
        # Mandatory per-benchmark iteration counts.
        # Subclasses MUST set both `self.iterations_regular` and
        # `self.iterations_trace` (positive ints) in their __init__ before
        # calling super().__init__(). The base class enforces this.

        self._validate_attr("iterations_regular")
        self._validate_attr("iterations_trace")

    @property
    def benchmark_bin(self) -> Path:
        """Returns the path to the benchmark binary"""
        return self.suite.project.build_dir / "bin" / self.bench_name

    def cpu_count_str(self, separator: str = "") -> str:
        # Note: SYCL CI currently relies on this "CPU count" value.
        # Please update /devops/scripts/benchmarks/compare.py if this value
        # is changed. See compare.py usage (w.r.t. --regression-filter) in
        # /devops/actions/run-tests/benchmarks/action.yml.
        return (
            f"{separator} CPU count"
            if self.profiler_type == PROFILERS.CPU_COUNTER
            else ""
        )

    def get_iters(self, run_trace: TracingType):
        """Returns the number of iterations to run for the given tracing type."""
        if options.exit_on_failure:
            # we are just testing that the benchmark runs successfully
            return 3
        if run_trace == TracingType.NONE:
            return self.iterations_regular
        return self.iterations_trace

    def supported_runtimes(self) -> list[RUNTIMES]:
        """Base runtimes supported by this benchmark, can be overridden."""
        # By default, support all runtimes except SYCL_PREVIEW
        return [r for r in RUNTIMES if r != RUNTIMES.SYCL_PREVIEW]

    def enabled_runtimes(self) -> list[RUNTIMES]:
        """Runtimes available given the current configuration."""
        # Start with all supported runtimes and apply configuration filters
        runtimes = self.supported_runtimes()

        # Remove UR if not available
        if options.ur is None:
            runtimes = [r for r in runtimes if r != RUNTIMES.UR]

        # Remove Level Zero if using CUDA backend
        if options.ur_adapter == "cuda":
            runtimes = [r for r in runtimes if r != RUNTIMES.LEVEL_ZERO]

        return runtimes

    def enabled(self) -> bool:
        # SYCL is required for all benchmarks
        if options.sycl is None:
            return False

        # HIP adapter is not supported
        if options.ur_adapter == "hip":
            return False

        # Check if the specific runtime is enabled (or no specific runtime required)
        return self.runtime is None or self.runtime in self.enabled_runtimes()

    def name(self):
        """Returns the name of the benchmark, can be overridden."""
        return self.bench_name

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        # Subclasses must implement this and include all flags except --iterations;
        # the base `run()` will prepend the proper --iterations value based on
        # `run_trace` and the subclass's `iterations_regular`/`iterations_trace`.
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def explicit_group(self):
        return ""

    def description(self) -> str:
        return ""

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        command = [
            str(self.benchmark_bin),
            f"--test={self.test}",
            "--csv",
            "--noHeaders",
        ]
        # Let subclass provide remaining args; bin_args(run_trace) must
        # include the proper --iterations token computed from this class's
        # iteration fields.
        command += self.bin_args(run_trace)
        env_vars.update(self.extra_env_vars())

        result = self.run_bench(
            command, env_vars, run_trace=run_trace, force_trace=force_trace
        )
        parsed_results = self.parse_output(result)
        ret = []
        for median, stddev in parsed_results:
            unit = "instr" if self.profiler_type == PROFILERS.CPU_COUNTER else "μs"
            ret.append(
                Result(
                    label=self.name(),
                    value=median,
                    stddev=stddev,
                    command=command,
                    env=env_vars,
                    unit=unit,
                    git_url=self.suite.git_url(),
                    git_hash=self.suite.git_hash(),
                )
            )
        return ret

    def parse_output(self, output: str) -> list[tuple[float, float]]:
        csv_file = io.StringIO(output)
        reader = csv.reader(csv_file)
        next(reader, None)
        results = []
        while True:
            data_row = next(reader, None)
            if data_row is None:
                break
            try:
                mean = float(data_row[1])
                median = float(data_row[2])
                # compute benchmarks report stddev as %
                stddev = mean * (float(data_row[3].strip("%")) / 100.0)
                if not math.isfinite(stddev):
                    stddev = 0.0  # Default to 0.0 if stddev is invalid

                results.append((median, stddev))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing output: {e}")
        if len(results) == 0:
            raise ValueError("Benchmark output does not contain data.")
        return results

    def teardown(self):
        return

    def _validate_attr(self, attr_name: str):
        if (
            not hasattr(self, attr_name)
            or not isinstance(getattr(self, attr_name, None), int)
            or getattr(self, attr_name, 0) <= 0
        ):
            raise ValueError(
                f"{self.bench_name}: subclasses must set `{attr_name}` (positive int) before calling super().__init__"
            )


class SubmitKernel(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        ioq,
        MeasureCompletion=0,
        UseEvents=0,
        KernelExecTime=1,
        profiler_type=PROFILERS.TIMER,
    ):
        self.ioq = ioq
        self.MeasureCompletion = MeasureCompletion
        self.UseEvents = UseEvents
        self.KernelExecTime = KernelExecTime
        self.NumKernels = 10
        # iterations set per existing bin_args: --iterations=100000
        self.iterations_regular = 100000
        self.iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "SubmitKernel",
            runtime,
            profiler_type,
        )

    def supported_runtimes(self) -> list[RUNTIMES]:
        return super().supported_runtimes() + [RUNTIMES.SYCL_PREVIEW]

    def enabled(self) -> bool:
        # This is a workaround for the BMG server where we have old results for self.KernelExecTime=20
        # The benchmark instance gets created just to make metadata for these old results
        if not super().enabled():
            return False

        device_arch = getattr(options, "device_architecture", "")
        if "bmg" in device_arch and self.KernelExecTime == 20:
            # Disable this benchmark for BMG server, just create metadata
            return False
        if "bmg" not in device_arch and self.KernelExecTime == 200:
            # Disable KernelExecTime=200 for non-BMG systems, just create metadata
            return False
        return True

    def get_tags(self):
        return ["submit", "latency", runtime_to_tag_name(self.runtime), "micro"]

    def name(self):
        order = "in order" if self.ioq else "out of order"
        completion_str = " with measure completion" if self.MeasureCompletion else ""

        # this needs to be inversed (i.e., using events is empty string)
        # to match the existing already stored results
        events_str = " not using events" if not self.UseEvents else ""

        kernel_exec_time_str = (
            f" KernelExecTime={self.KernelExecTime}" if self.KernelExecTime != 1 else ""
        )

        return f"api_overhead_benchmark_{self.runtime.value} SubmitKernel {order}{completion_str}{events_str}{kernel_exec_time_str}{self.cpu_count_str()}"

    def display_name(self) -> str:
        order = "in order" if self.ioq else "out of order"
        info = []
        if self.MeasureCompletion:
            info.append("with measure completion")
        if self.UseEvents:
            info.append("using events")
        if self.KernelExecTime != 1:
            info.append(f"KernelExecTime={self.KernelExecTime}")
        additional_info = f" {' '.join(info)}" if info else ""
        return f"{self.runtime.value.upper()} SubmitKernel {order}{additional_info}, NumKernels {self.NumKernels}{self.cpu_count_str(separator=',')}"

    def explicit_group(self):
        order = "in order" if self.ioq else "out of order"
        completion_str = " with completion" if self.MeasureCompletion else ""
        events_str = " using events" if self.UseEvents else ""

        kernel_exec_time_str = f" long kernel" if self.KernelExecTime != 1 else ""

        return f"SubmitKernel {order}{completion_str}{events_str}{kernel_exec_time_str}{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        order = "in-order" if self.ioq else "out-of-order"
        runtime_name = runtime_to_name(self.runtime)
        completion_desc = f", {'including' if self.MeasureCompletion else 'excluding'} kernel completion time"

        return (
            f"Measures CPU time overhead of submitting {order} kernels through {runtime_name} API{completion_desc}. "
            f"Runs {self.NumKernels} simple kernels with minimal execution time to isolate API overhead from kernel execution time."
            f"Each kernel executes for approximately {self.KernelExecTime} micro seconds."
        )

    def range(self) -> tuple[float, float]:
        return (0.0, None)

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--Ioq={self.ioq}",
            f"--MeasureCompletion={self.MeasureCompletion}",
            "--Profiling=0",
            f"--NumKernels={self.NumKernels}",
            f"--KernelExecTime={self.KernelExecTime}",
            f"--UseEvents={self.UseEvents}",
            f"--profilerType={self.profiler_type.value}",
        ]


class ExecImmediateCopyQueue(ComputeBenchmark):
    def __init__(
        self, bench, ioq, isCopyOnly, source, destination, size, profiler_type
    ):
        self.ioq = ioq
        self.isCopyOnly = isCopyOnly
        self.source = source
        self.destination = destination
        self.size = size
        # iterations per bin_args: --iterations=100000
        self.iterations_regular = 100000
        self.iterations_trace = 10
        super().__init__(
            bench,
            "api_overhead_benchmark_sycl",
            "ExecImmediateCopyQueue",
            profiler_type=profiler_type,
        )

    def name(self):
        order = "in order" if self.ioq else "out of order"
        return f"api_overhead_benchmark_sycl ExecImmediateCopyQueue {order} from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str()}"

    def display_name(self) -> str:
        order = "in order" if self.ioq else "out of order"
        return f"SYCL ExecImmediateCopyQueue {order} from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        order = "in-order" if self.ioq else "out-of-order"
        operation = "copy-only" if self.isCopyOnly else "copy and command submission"
        return (
            f"Measures SYCL {order} queue overhead for {operation} from {self.source} to "
            f"{self.destination} memory with {self.size} bytes. Tests immediate execution overheads."
        )

    def get_tags(self):
        return ["memory", "submit", "latency", "SYCL", "micro"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--ioq={self.ioq}",
            f"--IsCopyOnly={self.isCopyOnly}",
            "--MeasureCompletionTime=0",
            f"--src={self.destination}",
            f"--dst={self.destination}",
            f"--size={self.size}",
            "--withCopyOffload=0",
            f"--profilerType={self.profiler_type.value}",
        ]


class QueueInOrderMemcpy(ComputeBenchmark):
    def __init__(self, bench, isCopyOnly, source, destination, size, profiler_type):
        self.isCopyOnly = isCopyOnly
        self.source = source
        self.destination = destination
        self.size = size
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench,
            "memory_benchmark_sycl",
            "QueueInOrderMemcpy",
            profiler_type=profiler_type,
        )

    def name(self):
        return f"memory_benchmark_sycl QueueInOrderMemcpy from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str()}"

    def display_name(self) -> str:
        return f"SYCL QueueInOrderMemcpy from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        operation = "copy-only" if self.isCopyOnly else "copy and command submission"
        return (
            f"Measures SYCL in-order queue memory copy performance for {operation} from "
            f"{self.source} to {self.destination} with {self.size} bytes, executed 100 times per iteration."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--IsCopyOnly={self.isCopyOnly}",
            f"--sourcePlacement={self.source}",
            f"--destinationPlacement={self.destination}",
            f"--size={self.size}",
            "--count=100",
            "--withCopyOffload=0",
            f"--profilerType={self.profiler_type.value}",
        ]


class QueueMemcpy(ComputeBenchmark):
    def __init__(self, bench, source, destination, size, profiler_type):
        self.source = source
        self.destination = destination
        self.size = size
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench, "memory_benchmark_sycl", "QueueMemcpy", profiler_type=profiler_type
        )

    def name(self):
        return f"memory_benchmark_sycl QueueMemcpy from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str()}"

    def display_name(self) -> str:
        return f"SYCL QueueMemcpy from {self.source} to {self.destination}, size {self.size}{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        return (
            f"Measures general SYCL queue memory copy performance from {self.source} to "
            f"{self.destination} with {self.size} bytes per operation."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--sourcePlacement={self.source}",
            f"--destinationPlacement={self.destination}",
            f"--size={self.size}",
            f"--profilerType={self.profiler_type.value}",
        ]


class StreamMemory(ComputeBenchmark):
    def __init__(self, bench, type, size, placement):
        self.type = type
        self.size = size
        self.placement = placement
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(bench, "memory_benchmark_sycl", "StreamMemory")

    def name(self):
        return f"memory_benchmark_sycl StreamMemory, placement {self.placement}, type {self.type}, size {self.size}"

    def display_name(self) -> str:
        return f"SYCL StreamMemory, placement {self.placement}, type {self.type}, size {self.size}"

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

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self.type}",
            f"--size={self.size}",
            f"--memoryPlacement={self.placement}",
            "--useEvents=0",
            "--contents=Zeros",
            "--multiplier=1",
            "--vectorSize=1",
            "--lws=256",
        ]


class VectorSum(ComputeBenchmark):
    def __init__(self, bench):
        # iterations per bin_args: --iterations=1000
        self.iterations_regular = 1000
        self.iterations_trace = 10
        super().__init__(bench, "miscellaneous_benchmark_sycl", "VectorSum")

    def name(self):
        return f"miscellaneous_benchmark_sycl VectorSum"

    def display_name(self) -> str:
        return f"SYCL VectorSum"

    def description(self) -> str:
        return (
            "Measures performance of vector addition across 3D grid (512x256x256 elements) "
            "using SYCL."
        )

    def get_tags(self):
        return ["math", "throughput", "SYCL", "micro"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            "--numberOfElementsX=512",
            "--numberOfElementsY=256",
            "--numberOfElementsZ=256",
        ]


class MemcpyExecute(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        numOpsPerThread,
        numThreads,
        allocSize,
        iterations,
        srcUSM,
        dstUSM,
        useEvent,
        useCopyOffload,
        useBarrier,
    ):
        self.numOpsPerThread = numOpsPerThread
        self.numThreads = numThreads
        self.allocSize = allocSize
        # preserve provided iterations value
        # self.iterations = iterations
        self.iterations_regular = iterations
        self.iterations_trace = min(iterations, 10)
        self.srcUSM = srcUSM
        self.dstUSM = dstUSM
        self.useEvents = useEvent
        self.useCopyOffload = useCopyOffload
        self.useBarrier = useBarrier
        super().__init__(
            bench, f"multithread_benchmark_{runtime.value}", "MemcpyExecute", runtime
        )

    def extra_env_vars(self) -> dict:
        if not self.useCopyOffload:
            return {"UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD": "1"}
        else:
            return {}

    def name(self):
        return (
            f"multithread_benchmark_{self.runtime.value} MemcpyExecute opsPerThread:{self.numOpsPerThread}, numThreads:{self.numThreads}, allocSize:{self.allocSize} srcUSM:{self.srcUSM} dstUSM:{self.dstUSM}"
            + (" without events" if not self.useEvents else "")
            + (" without copy offload" if not self.useCopyOffload else "")
            + (" with barrier" if self.useBarrier else "")
        )

    def display_name(self) -> str:
        info = []
        if not self.useEvents:
            info.append("without events")
        if not self.useCopyOffload:
            info.append("without copy offload")
        additional_info = f", {' '.join(info)}" if info else ""
        return (
            f"UR MemcpyExecute, opsPerThread {self.numOpsPerThread}, "
            f"numThreads {self.numThreads}, allocSize {self.allocSize}, srcUSM {self.srcUSM}, "
            f"dstUSM {self.dstUSM}{additional_info}"
        )

    def explicit_group(self):
        return (
            "MemcpyExecute, opsPerThread: "
            + str(self.numOpsPerThread)
            + ", numThreads: "
            + str(self.numThreads)
            + ", allocSize: "
            + str(self.allocSize)
        )

    def description(self) -> str:
        src_type = "device" if self.srcUSM == 1 else "host"
        dst_type = "device" if self.dstUSM == 1 else "host"
        events = "with" if self.useEvents else "without"
        copy_offload = "with" if self.useCopyOffload else "without"
        with_barrier = "with" if self.useBarrier else "without"
        return (
            f"Measures multithreaded memory copy performance with {self.numThreads} threads "
            f"each performing {self.numOpsPerThread} operations on {self.allocSize} bytes "
            f"from {src_type} to {dst_type} memory {events} events {copy_offload} driver copy offload "
            f"{with_barrier} barrier. "
        )

    def get_tags(self):
        return ["memory", "latency", runtime_to_tag_name(self.runtime), "micro"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            "--Ioq=1",
            f"--UseEvents={self.useEvents}",
            "--MeasureCompletion=1",
            "--UseQueuePerThread=1",
            f"--AllocSize={self.allocSize}",
            f"--NumThreads={self.numThreads}",
            f"--NumOpsPerThread={self.numOpsPerThread}",
            f"--SrcUSM={self.srcUSM}",
            f"--DstUSM={self.dstUSM}",
            f"--UseBarrier={self.useBarrier}",
        ]


class GraphApiSinKernelGraph(ComputeBenchmark):
    def __init__(self, bench, runtime: RUNTIMES, withGraphs, numKernels):
        self.withGraphs = withGraphs
        self.numKernels = numKernels
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench, f"graph_api_benchmark_{runtime.value}", "SinKernelGraph", runtime
        )

    def explicit_group(self):
        return f"SinKernelGraph, numKernels: {self.numKernels}"

    def description(self) -> str:
        execution = "using graphs" if self.withGraphs else "without graphs"
        return (
            f"Measures {self.runtime.value.upper()} performance when executing {self.numKernels} "
            f"sin kernels {execution}. Tests overhead and benefits of graph-based execution."
        )

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} SinKernelGraph graphs:{self.withGraphs}, numKernels:{self.numKernels}"

    def display_name(self) -> str:
        return f"{self.runtime.value.upper()} SinKernelGraph, graphs {self.withGraphs}, numKernels {self.numKernels}"

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

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--numKernels={self.numKernels}",
            f"--withGraphs={self.withGraphs}",
            "--withCopyOffload=1",
            "--immediateAppendCmdList=0",
        ]


class GraphApiSubmitGraph(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        inOrderQueue,
        numKernels,
        measureCompletionTime,
        useEvents,
        emulate_graphs,
        useHostTasks,
        profiler_type=PROFILERS.TIMER,
    ):
        self.inOrderQueue = inOrderQueue
        self.numKernels = numKernels
        self.measureCompletionTime = measureCompletionTime
        self.useEvents = useEvents
        self.useHostTasks = useHostTasks
        self.emulateGraphs = emulate_graphs
        self.ioq_str = "in order" if self.inOrderQueue else "out of order"
        self.measure_str = (
            " with measure completion" if self.measureCompletionTime else ""
        )
        self.use_events_str = f" with events" if self.useEvents else ""
        self.host_tasks_str = f" use host tasks" if self.useHostTasks else ""
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench,
            f"graph_api_benchmark_{runtime.value}",
            "SubmitGraph",
            runtime,
            profiler_type,
        )

    def supported_runtimes(self) -> list[RUNTIMES]:
        return super().supported_runtimes() + [RUNTIMES.SYCL_PREVIEW]

    def explicit_group(self):
        return f"SubmitGraph {self.ioq_str}{self.measure_str}{self.use_events_str}{self.host_tasks_str}, {self.numKernels} kernels{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        return (
            f"Measures {self.runtime.value.upper()} performance when executing {self.numKernels} "
            f"trivial kernels using graphs. Tests overhead and benefits of graph-based execution."
        )

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} SubmitGraph{self.use_events_str}{self.host_tasks_str} numKernels:{self.numKernels} ioq {self.inOrderQueue} measureCompletion {self.measureCompletionTime}{self.cpu_count_str()}"

    def display_name(self) -> str:
        return f"{self.runtime.value.upper()} SubmitGraph {self.ioq_str}{self.measure_str}{self.use_events_str}{self.host_tasks_str}, {self.numKernels} kernels{self.cpu_count_str(separator=',')}"

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self.runtime),
            "micro",
            "submit",
            "latency",
        ]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--NumKernels={self.numKernels}",
            f"--MeasureCompletionTime={self.measureCompletionTime}",
            f"--InOrderQueue={self.inOrderQueue}",
            "--Profiling=0",
            "--KernelExecutionTime=1",
            f"--UseEvents={self.useEvents}",
            "--UseExplicit=0",
            f"--UseHostTasks={self.useHostTasks}",
            f"--profilerType={self.profiler_type.value}",
            f"--EmulateGraphs={self.emulateGraphs}",
        ]


class UllsEmptyKernel(ComputeBenchmark):
    def __init__(
        self, bench, runtime: RUNTIMES, wgc, wgs, profiler_type=PROFILERS.TIMER
    ):
        self.wgc = wgc
        self.wgs = wgs
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench,
            f"ulls_benchmark_{runtime.value}",
            "EmptyKernel",
            runtime,
            profiler_type,
        )

    def supported_runtimes(self) -> list[RUNTIMES]:
        return [RUNTIMES.SYCL, RUNTIMES.LEVEL_ZERO]

    def explicit_group(self):
        return f"EmptyKernel, wgc: {self.wgc}, wgs: {self.wgs}{self.cpu_count_str(separator=',')}"

    def description(self) -> str:
        return ""

    def name(self):
        return f"ulls_benchmark_{self.runtime.value} EmptyKernel wgc:{self.wgc}, wgs:{self.wgs}{self.cpu_count_str()}"

    def display_name(self) -> str:
        return f"{self.runtime.value.upper()} EmptyKernel, wgc {self.wgc}, wgs {self.wgs}{self.cpu_count_str(separator=',')}"

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "submit"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--wgs={self.wgs}",
            f"--wgc={self.wgc}",
            f"--profilerType={self.profiler_type.value}",
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
        self.ioq = ioq
        # iterations per bin_args: --iterations=1000
        self.iterations_regular = 1000
        self.iterations_trace = 10
        super().__init__(
            bench, f"ulls_benchmark_{runtime.value}", "KernelSwitch", runtime
        )

    def supported_runtimes(self):
        return [RUNTIMES.SYCL, RUNTIMES.LEVEL_ZERO]

    def explicit_group(self):
        return f"KernelSwitch, count: {self.count}, kernelTime: {self.kernelTime}"

    def description(self) -> str:
        return ""

    def name(self):
        return f"ulls_benchmark_{self.runtime.value} KernelSwitch count {self.count} kernelTime {self.kernelTime}"

    def display_name(self) -> str:
        return f"{self.runtime.value.upper()} KernelSwitch, count {self.count}, kernelTime {self.kernelTime}"

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "submit"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
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
        self.usm_memory_placement = usm_memory_placement
        self.size = size
        self.measure_mode = measure_mode
        # iterations per bin_args: --iterations=10000
        self.iterations_regular = 10000
        self.iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "UsmMemoryAllocation",
            runtime,
        )

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "memory"]

    def name(self):
        return (
            f"api_overhead_benchmark_{self.runtime.value} UsmMemoryAllocation "
            f"usmMemoryPlacement:{self.usm_memory_placement} size:{self.size} measureMode:{self.measure_mode}"
        )

    def display_name(self) -> str:
        return (
            f"{self.runtime.value.upper()} UsmMemoryAllocation, "
            f"usmMemoryPlacement {self.usm_memory_placement}, size {self.size}, measureMode {self.measure_mode}"
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

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self.usm_memory_placement}",
            f"--size={self.size}",
            f"--measureMode={self.measure_mode}",
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
        self.usm_memory_placement = usm_memory_placement
        self.allocation_count = allocation_count
        self.size = size
        self.measure_mode = measure_mode
        # iterations per bin_args: --iterations=1000
        self.iterations_regular = 1000
        self.iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "UsmBatchMemoryAllocation",
            runtime,
        )

    def get_tags(self):
        return [runtime_to_tag_name(self.runtime), "micro", "latency", "memory"]

    def name(self):
        return (
            f"api_overhead_benchmark_{self.runtime.value} UsmBatchMemoryAllocation "
            f"usmMemoryPlacement:{self.usm_memory_placement} allocationCount:{self.allocation_count} size:{self.size} measureMode:{self.measure_mode}"
        )

    def display_name(self) -> str:
        return (
            f"{self.runtime.value.upper()} UsmBatchMemoryAllocation, "
            f"usmMemoryPlacement {self.usm_memory_placement}, allocationCount {self.allocation_count}, size {self.size}, measureMode {self.measure_mode}"
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

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self.usm_memory_placement}",
            f"--allocationCount={self.allocation_count}",
            f"--size={self.size}",
            f"--measureMode={self.measure_mode}",
        ]


class GraphApiFinalizeGraph(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        rebuild_graph_every_iteration,
        graph_structure,
    ):
        self.rebuild_graph_every_iteration = rebuild_graph_every_iteration
        self.graph_structure = graph_structure
        # base iterations value mirrors previous behaviour
        base_iters = 10000
        if graph_structure == "Llama":
            base_iters = base_iters // 10
        self.iterations_regular = base_iters
        self.iterations_trace = min(base_iters, 10)

        super().__init__(
            bench,
            f"graph_api_benchmark_{runtime.value}",
            "FinalizeGraph",
            runtime,
        )

    def explicit_group(self):
        return f"FinalizeGraph, GraphStructure: {self.graph_structure}"

    def description(self) -> str:
        what_is_measured = ""

        if self.rebuild_graph_every_iteration == 0:
            what_is_measured = (
                "It measures finalizing the same modifiable graph repeatedly "
                "over multiple iterations."
            )
        else:
            what_is_measured = (
                "It measures finalizing a unique modifiable graph per iteration."
            )

        return (
            "Measures the time taken to finalize a SYCL graph, using a graph "
            f"structure based on the usage of graphs in {self.graph_structure}. "
            f"{what_is_measured}"
        )

    def name(self):
        return f"graph_api_benchmark_{self.runtime.value} FinalizeGraph rebuildGraphEveryIter:{self.rebuild_graph_every_iteration} graphStructure:{self.graph_structure}"

    def display_name(self) -> str:
        return f"{self.runtime.value.upper()} FinalizeGraph, rebuildGraphEveryIter {self.rebuild_graph_every_iteration}, graphStructure {self.graph_structure}"

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self.runtime),
            "micro",
            "finalize",
            "latency",
        ]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self.get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--rebuildGraphEveryIter={self.rebuild_graph_every_iteration}",
            f"--graphStructure={self.graph_structure}",
        ]
