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
        self._submit_graph_num_kernels = [4, 10, 32]
        self._project = None

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

        if self._project is None:
            self._project = GitProject(
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

        self._project.configure(extra_args, add_sycl=True)
        self._project.build(add_sycl=True)

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
            self._submit_graph_num_kernels,
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

        record_and_replay_params = product([0, 1], [0, 1])
        for emulate, instantiate in record_and_replay_params:

            def createRrBench(variant_name: str, **kwargs):
                return RecordAndReplay(
                    self,
                    RUNTIMES.LEVEL_ZERO,
                    variant_name,
                    PROFILERS.TIMER,
                    mRec=1,
                    mInst=instantiate,
                    mDest=0,
                    emulate=emulate,
                    **kwargs,
                )

            benches += [
                createRrBench(
                    "large",
                    nForksInLvl=2,
                    nLvls=4,
                    nCmdSetsInLvl=10,
                    nInstantiations=10,
                    nAppendKern=10,
                    nAppendCopy=1,
                ),
                createRrBench(
                    "medium",
                    nForksInLvl=1,
                    nLvls=1,
                    nCmdSetsInLvl=10,
                    nInstantiations=10,
                    nAppendKern=10,
                    nAppendCopy=10,
                ),
                createRrBench(
                    "short",
                    nForksInLvl=1,
                    nLvls=4,
                    nCmdSetsInLvl=1,
                    nInstantiations=0,
                    nAppendKern=1,
                    nAppendCopy=0,
                ),
            ]

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
        self._suite = suite
        self._bench_name = name
        self._test = test
        self._runtime = runtime
        self._profiler_type = profiler_type
        # Mandatory per-benchmark iteration counts.
        # Subclasses MUST set both `self._iterations_regular` and
        # `self._iterations_trace` (positive ints) in their __init__ before
        # calling super().__init__(). The base class enforces this.

        self.__validate_attr("_iterations_regular")
        self.__validate_attr("_iterations_trace")

    def name(self):
        """Returns the name of the benchmark, can be overridden."""
        return self._bench_name

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        command = [
            str(self.__benchmark_bin),
            f"--test={self._test}",
            "--csv",
            "--noHeaders",
        ]
        # Let subclass provide remaining args; bin_args(run_trace) must
        # include the proper --iterations token computed from this class's
        # iteration fields.
        command += self._bin_args(run_trace)
        env_vars.update(self._extra_env_vars())

        result = self.run_bench(
            command, env_vars, run_trace=run_trace, force_trace=force_trace
        )
        parsed_results = self.__parse_output(result)
        ret = []
        for median, stddev in parsed_results:
            unit = "instr" if self._profiler_type == PROFILERS.CPU_COUNTER else "μs"
            ret.append(
                Result(
                    label=self.name(),
                    value=median,
                    stddev=stddev,
                    command=command,
                    env=env_vars,
                    unit=unit,
                    git_url=self._suite.git_url(),
                    git_hash=self._suite.git_hash(),
                )
            )
        return ret

    def explicit_group(self):
        return ""

    def enabled(self) -> bool:
        # SYCL is required for all benchmarks
        if options.sycl is None:
            return False

        # HIP adapter is not supported
        if options.ur_adapter == "hip":
            return False

        # Check if the specific runtime is enabled (or no specific runtime required)
        return self._runtime is None or self._runtime in self.__enabled_runtimes()

    def description(self) -> str:
        return ""

    def _cpu_count_str(self, separator: str = "") -> str:
        # Note: SYCL CI currently relies on this "CPU count" value.
        # Please update /devops/scripts/benchmarks/compare.py if this value
        # is changed. See compare.py usage (w.r.t. --regression-filter) in
        # /devops/actions/run-tests/benchmarks/action.yml.
        return (
            f"{separator} CPU count"
            if self._profiler_type == PROFILERS.CPU_COUNTER
            else ""
        )

    def _get_iters(self, run_trace: TracingType):
        """Returns the number of iterations to run for the given tracing type."""
        if options.exit_on_failure:
            # we are just testing that the benchmark runs successfully
            return 3
        if run_trace == TracingType.NONE:
            return self._iterations_regular
        return self._iterations_trace

    def _supported_runtimes(self) -> list[RUNTIMES]:
        """Base runtimes supported by this benchmark, can be overridden."""
        # By default, support all runtimes except SYCL_PREVIEW
        return [r for r in RUNTIMES if r != RUNTIMES.SYCL_PREVIEW]

    def _extra_env_vars(self) -> dict:
        return {}

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        # Subclasses must implement this and include all flags except --iterations;
        # the base `run()` will prepend the proper --iterations value based on
        # `run_trace` and the subclass's `iterations_regular`/`iterations_trace`.
        return []

    @property
    def __benchmark_bin(self) -> Path:
        """Returns the path to the benchmark binary"""
        return self._suite._project.build_dir / "bin" / self._bench_name

    def __enabled_runtimes(self) -> list[RUNTIMES]:
        """Runtimes available given the current configuration."""
        # Start with all supported runtimes and apply configuration filters
        runtimes = self._supported_runtimes()

        # Remove UR if not available
        if options.ur is None:
            runtimes = [r for r in runtimes if r != RUNTIMES.UR]

        # Remove Level Zero if using CUDA backend
        if options.ur_adapter == "cuda":
            runtimes = [r for r in runtimes if r != RUNTIMES.LEVEL_ZERO]

        return runtimes

    def __parse_output(self, output: str) -> list[tuple[float, float]]:
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

    def __validate_attr(self, attr_name: str):
        if (
            not hasattr(self, attr_name)
            or not isinstance(getattr(self, attr_name, None), int)
            or getattr(self, attr_name, 0) <= 0
        ):
            raise ValueError(
                f"{self._bench_name}: subclasses must set `{attr_name}` (positive int) before calling super().__init__"
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
        self._ioq = ioq
        self._measure_completion = MeasureCompletion
        self._use_events = UseEvents
        self._kernel_exec_time = KernelExecTime
        self._num_kernels = 10
        # iterations set per existing bin_args: --iterations=100000
        self._iterations_regular = 100000
        self._iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "SubmitKernel",
            runtime,
            profiler_type,
        )

    def name(self):
        order = "in order" if self._ioq else "out of order"
        completion_str = " with measure completion" if self._measure_completion else ""

        # this needs to be inversed (i.e., using events is empty string)
        # to match the existing already stored results
        events_str = " not using events" if not self._use_events else ""

        kernel_exec_time_str = (
            f" KernelExecTime={self._kernel_exec_time}"
            if self._kernel_exec_time != 1
            else ""
        )

        return f"api_overhead_benchmark_{self._runtime.value} SubmitKernel {order}{completion_str}{events_str}{kernel_exec_time_str}{self._cpu_count_str()}"

    def display_name(self) -> str:
        order = "in order" if self._ioq else "out of order"
        info = []
        if self._measure_completion:
            info.append("with measure completion")
        if self._use_events:
            info.append("using events")
        if self._kernel_exec_time != 1:
            info.append(f"KernelExecTime={self._kernel_exec_time}")
        additional_info = f" {' '.join(info)}" if info else ""
        return f"{self._runtime.value.upper()} SubmitKernel {order}{additional_info}, NumKernels {self._num_kernels}{self._cpu_count_str(separator=',')}"

    def explicit_group(self):
        order = "in order" if self._ioq else "out of order"
        completion_str = " with completion" if self._measure_completion else ""
        events_str = " using events" if self._use_events else ""

        kernel_exec_time_str = f" long kernel" if self._kernel_exec_time != 1 else ""

        return f"SubmitKernel {order}{completion_str}{events_str}{kernel_exec_time_str}{self._cpu_count_str(separator=',')}"

    def enabled(self) -> bool:
        # This is a workaround for the BMG server where we have old results for self._kernel_exec_time=20
        # The benchmark instance gets created just to make metadata for these old results
        if not super().enabled():
            return False

        device_arch = getattr(options, "device_architecture", "")
        if "bmg" in device_arch and self._kernel_exec_time == 20:
            # Disable this benchmark for BMG server, just create metadata
            return False
        if "bmg" not in device_arch and self._kernel_exec_time == 200:
            # Disable KernelExecTime=200 for non-BMG systems, just create metadata
            return False
        return True

    def description(self) -> str:
        order = "in-order" if self._ioq else "out-of-order"
        runtime_name = runtime_to_name(self._runtime)
        completion_desc = f", {'including' if self._measure_completion else 'excluding'} kernel completion time"

        return (
            f"Measures CPU time overhead of submitting {order} kernels through {runtime_name} API{completion_desc}. "
            f"Runs {self._num_kernels} simple kernels with minimal execution time to isolate API overhead from kernel execution time."
            f"Each kernel executes for approximately {self._kernel_exec_time} micro seconds."
        )

    def get_tags(self):
        return ["submit", "latency", runtime_to_tag_name(self._runtime), "micro"]

    def range(self) -> tuple[float, float]:
        return (0.0, None)

    def _supported_runtimes(self) -> list[RUNTIMES]:
        return super()._supported_runtimes() + [RUNTIMES.SYCL_PREVIEW]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--Ioq={self._ioq}",
            f"--MeasureCompletion={self._measure_completion}",
            "--Profiling=0",
            f"--NumKernels={self._num_kernels}",
            f"--KernelExecTime={self._kernel_exec_time}",
            f"--UseEvents={self._use_events}",
            f"--profilerType={self._profiler_type.value}",
        ]


class ExecImmediateCopyQueue(ComputeBenchmark):
    def __init__(
        self, bench, ioq, isCopyOnly, source, destination, size, profiler_type
    ):
        self._ioq = ioq
        self._is_copy_only = isCopyOnly
        self._source = source
        self._destination = destination
        self._size = size
        # iterations per bin_args: --iterations=100000
        self._iterations_regular = 100000
        self._iterations_trace = 10
        super().__init__(
            bench,
            "api_overhead_benchmark_sycl",
            "ExecImmediateCopyQueue",
            profiler_type=profiler_type,
        )

    def name(self):
        order = "in order" if self._ioq else "out of order"
        return f"api_overhead_benchmark_sycl ExecImmediateCopyQueue {order} from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str()}"

    def display_name(self) -> str:
        order = "in order" if self._ioq else "out of order"
        return f"SYCL ExecImmediateCopyQueue {order} from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str(separator=',')}"

    def description(self) -> str:
        order = "in-order" if self._ioq else "out-of-order"
        operation = "copy-only" if self._is_copy_only else "copy and command submission"
        return (
            f"Measures SYCL {order} queue overhead for {operation} from {self._source} to "
            f"{self._destination} memory with {self._size} bytes. Tests immediate execution overheads."
        )

    def get_tags(self):
        return ["memory", "submit", "latency", "SYCL", "micro"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--ioq={self._ioq}",
            f"--IsCopyOnly={self._is_copy_only}",
            "--MeasureCompletionTime=0",
            f"--src={self._destination}",
            f"--dst={self._destination}",
            f"--size={self._size}",
            "--withCopyOffload=0",
            f"--profilerType={self._profiler_type.value}",
        ]


class RecordAndReplay(ComputeBenchmark):
    def __init__(
        self, suite, runtime: RUNTIMES, variant_name: str, profiler_type, **kwargs
    ):
        self.variant_name = variant_name
        self.rr_params = kwargs
        self.iterations_regular = 1000
        self.iterations_trace = 10
        super().__init__(
            suite,
            f"record_and_replay_benchmark_{runtime.value}",
            "RecordGraph",
            runtime,
            profiler_type,
        )

    def explicit_group(self):
        return f"{self.test} {self.variant_name}"

    def display_name(self) -> str:
        return f"{self.explicit_group()}_{self.runtime.value}"

    def name(self):
        ret = []
        for k, v in self.rr_params.items():
            if k[0] == "n":  # numeric parameter
                ret.append(f"{k[1:]} {v}")
            elif k[0] == "m":
                if v != 0:  # measure parameter
                    ret.append(f"{k[1:]}")
            else:  # boolean parameter
                if v != 0:
                    ret.append(k)
        ret.sort()
        return self.bench_name + " " + ", ".join(ret)

    def get_tags(self):
        return ["L0"]

    def bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        return [f"--{k}={v}" for k, v in self.rr_params.items()]


class QueueInOrderMemcpy(ComputeBenchmark):
    def __init__(self, bench, isCopyOnly, source, destination, size, profiler_type):
        self._is_copy_only = isCopyOnly
        self._source = source
        self._destination = destination
        self._size = size
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench,
            "memory_benchmark_sycl",
            "QueueInOrderMemcpy",
            profiler_type=profiler_type,
        )

    def name(self):
        return f"memory_benchmark_sycl QueueInOrderMemcpy from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str()}"

    def display_name(self) -> str:
        return f"SYCL QueueInOrderMemcpy from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str(separator=',')}"

    def description(self) -> str:
        operation = "copy-only" if self._is_copy_only else "copy and command submission"
        return (
            f"Measures SYCL in-order queue memory copy performance for {operation} from "
            f"{self._source} to {self._destination} with {self._size} bytes, executed 100 times per iteration."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--IsCopyOnly={self._is_copy_only}",
            f"--sourcePlacement={self._source}",
            f"--destinationPlacement={self._destination}",
            f"--size={self._size}",
            "--count=100",
            "--withCopyOffload=0",
            f"--profilerType={self._profiler_type.value}",
        ]


class QueueMemcpy(ComputeBenchmark):
    def __init__(self, bench, source, destination, size, profiler_type):
        self._source = source
        self._destination = destination
        self._size = size
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench, "memory_benchmark_sycl", "QueueMemcpy", profiler_type=profiler_type
        )

    def name(self):
        return f"memory_benchmark_sycl QueueMemcpy from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str()}"

    def display_name(self) -> str:
        return f"SYCL QueueMemcpy from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str(separator=',')}"

    def description(self) -> str:
        return (
            f"Measures general SYCL queue memory copy performance from {self._source} to "
            f"{self._destination} with {self._size} bytes per operation."
        )

    def get_tags(self):
        return ["memory", "latency", "SYCL", "micro"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--sourcePlacement={self._source}",
            f"--destinationPlacement={self._destination}",
            f"--size={self._size}",
            f"--profilerType={self._profiler_type.value}",
        ]


class StreamMemory(ComputeBenchmark):
    def __init__(self, bench, type, size, placement):
        self._type = type
        self._size = size
        self._placement = placement
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(bench, "memory_benchmark_sycl", "StreamMemory")

    def name(self):
        return f"memory_benchmark_sycl StreamMemory, placement {self._placement}, type {self._type}, size {self._size}"

    def display_name(self) -> str:
        return f"SYCL StreamMemory, placement {self._placement}, type {self._type}, size {self._size}"

    # measurement is in GB/s
    def lower_is_better(self):
        return False

    def description(self) -> str:
        return (
            f"Measures {self._placement} memory bandwidth using {self._type} pattern with "
            f"{self._size} bytes. Higher values (GB/s) indicate better performance."
        )

    def get_tags(self):
        return ["memory", "throughput", "SYCL", "micro"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self._type}",
            f"--size={self._size}",
            f"--memoryPlacement={self._placement}",
            "--useEvents=0",
            "--contents=Zeros",
            "--multiplier=1",
            "--vectorSize=1",
            "--lws=256",
        ]


class VectorSum(ComputeBenchmark):
    def __init__(self, bench):
        # iterations per bin_args: --iterations=1000
        self._iterations_regular = 1000
        self._iterations_trace = 10
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

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
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
        self._num_ops_per_thread = numOpsPerThread
        self._num_threads = numThreads
        self._alloc_size = allocSize
        # preserve provided iterations value
        # self._iterations = iterations
        self._iterations_regular = iterations
        self._iterations_trace = min(iterations, 10)
        self._src_usm = srcUSM
        self._dst_usm = dstUSM
        self._use_events = useEvent
        self._use_copy_offload = useCopyOffload
        self._use_barrier = useBarrier
        super().__init__(
            bench, f"multithread_benchmark_{runtime.value}", "MemcpyExecute", runtime
        )

    def name(self):
        return (
            f"multithread_benchmark_{self._runtime.value} MemcpyExecute opsPerThread:{self._num_ops_per_thread}, numThreads:{self._num_threads}, allocSize:{self._alloc_size} srcUSM:{self._src_usm} dstUSM:{self._dst_usm}"
            + (" without events" if not self._use_events else "")
            + (" without copy offload" if not self._use_copy_offload else "")
            + (" with barrier" if self._use_barrier else "")
        )

    def display_name(self) -> str:
        info = []
        if not self._use_events:
            info.append("without events")
        if not self._use_copy_offload:
            info.append("without copy offload")
        additional_info = f", {' '.join(info)}" if info else ""
        return (
            f"UR MemcpyExecute, opsPerThread {self._num_ops_per_thread}, "
            f"numThreads {self._num_threads}, allocSize {self._alloc_size}, srcUSM {self._src_usm}, "
            f"dstUSM {self._dst_usm}{additional_info}"
        )

    def explicit_group(self):
        return (
            "MemcpyExecute, opsPerThread: "
            + str(self._num_ops_per_thread)
            + ", numThreads: "
            + str(self._num_threads)
            + ", allocSize: "
            + str(self._alloc_size)
        )

    def description(self) -> str:
        src_type = "device" if self._src_usm == 1 else "host"
        dst_type = "device" if self._dst_usm == 1 else "host"
        events = "with" if self._use_events else "without"
        copy_offload = "with" if self._use_copy_offload else "without"
        with_barrier = "with" if self._use_barrier else "without"
        return (
            f"Measures multithreaded memory copy performance with {self._num_threads} threads "
            f"each performing {self._num_ops_per_thread} operations on {self._alloc_size} bytes "
            f"from {src_type} to {dst_type} memory {events} events {copy_offload} driver copy offload "
            f"{with_barrier} barrier. "
        )

    def get_tags(self):
        return ["memory", "latency", runtime_to_tag_name(self._runtime), "micro"]

    def _extra_env_vars(self) -> dict:
        if not self._use_copy_offload:
            return {"UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD": "1"}
        else:
            return {}

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            "--Ioq=1",
            f"--UseEvents={self._use_events}",
            "--MeasureCompletion=1",
            "--UseQueuePerThread=1",
            f"--AllocSize={self._alloc_size}",
            f"--NumThreads={self._num_threads}",
            f"--NumOpsPerThread={self._num_ops_per_thread}",
            f"--SrcUSM={self._src_usm}",
            f"--DstUSM={self._dst_usm}",
            f"--UseBarrier={self._use_barrier}",
        ]


class GraphApiSinKernelGraph(ComputeBenchmark):
    def __init__(self, bench, runtime: RUNTIMES, withGraphs, numKernels):
        self._with_graphs = withGraphs
        self._num_kernels = numKernels
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench, f"graph_api_benchmark_{runtime.value}", "SinKernelGraph", runtime
        )

    def name(self):
        return f"graph_api_benchmark_{self._runtime.value} SinKernelGraph graphs:{self._with_graphs}, numKernels:{self._num_kernels}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} SinKernelGraph, graphs {self._with_graphs}, numKernels {self._num_kernels}"

    def explicit_group(self):
        return f"SinKernelGraph, numKernels: {self._num_kernels}"

    def description(self) -> str:
        execution = "using graphs" if self._with_graphs else "without graphs"
        return (
            f"Measures {self._runtime.value.upper()} performance when executing {self._num_kernels} "
            f"sin kernels {execution}. Tests overhead and benefits of graph-based execution."
        )

    def unstable(self) -> str:
        return "This benchmark combines both eager and graph execution, and may not be representative of real use cases."

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self._runtime),
            "proxy",
            "submit",
            "memory",
            "latency",
        ]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--numKernels={self._num_kernels}",
            f"--withGraphs={self._with_graphs}",
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
        self._in_order_queue = inOrderQueue
        self._num_kernels = numKernels
        self._measure_completion_time = measureCompletionTime
        self._use_events = useEvents
        self._use_host_tasks = useHostTasks
        self._emulate_graphs = emulate_graphs
        self._ioq_str = "in order" if self._in_order_queue else "out of order"
        self._measure_str = (
            " with measure completion" if self._measure_completion_time else ""
        )
        self._use_events_str = f" with events" if self._use_events else ""
        self._host_tasks_str = f" use host tasks" if self._use_host_tasks else ""
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench,
            f"graph_api_benchmark_{runtime.value}",
            "SubmitGraph",
            runtime,
            profiler_type,
        )

    def name(self):
        return f"graph_api_benchmark_{self._runtime.value} SubmitGraph{self._use_events_str}{self._host_tasks_str} numKernels:{self._num_kernels} ioq {self._in_order_queue} measureCompletion {self._measure_completion_time}{self._cpu_count_str()}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} SubmitGraph {self._ioq_str}{self._measure_str}{self._use_events_str}{self._host_tasks_str}, {self._num_kernels} kernels{self._cpu_count_str(separator=',')}"

    def explicit_group(self):
        return f"SubmitGraph {self._ioq_str}{self._measure_str}{self._use_events_str}{self._host_tasks_str}, {self._num_kernels} kernels{self._cpu_count_str(separator=',')}"

    def description(self) -> str:
        return (
            f"Measures {self._runtime.value.upper()} performance when executing {self._num_kernels} "
            f"trivial kernels using graphs. Tests overhead and benefits of graph-based execution."
        )

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self._runtime),
            "micro",
            "submit",
            "latency",
        ]

    def _supported_runtimes(self) -> list[RUNTIMES]:
        return super()._supported_runtimes() + [RUNTIMES.SYCL_PREVIEW]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--NumKernels={self._num_kernels}",
            f"--MeasureCompletionTime={self._measure_completion_time}",
            f"--InOrderQueue={self._in_order_queue}",
            "--Profiling=0",
            "--KernelExecutionTime=1",
            f"--UseEvents={self._use_events}",
            "--UseExplicit=0",
            f"--UseHostTasks={self._use_host_tasks}",
            f"--profilerType={self._profiler_type.value}",
            f"--EmulateGraphs={self._emulate_graphs}",
        ]


class UllsEmptyKernel(ComputeBenchmark):
    def __init__(
        self, bench, runtime: RUNTIMES, wgc, wgs, profiler_type=PROFILERS.TIMER
    ):
        self._wgc = wgc
        self._wgs = wgs
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench,
            f"ulls_benchmark_{runtime.value}",
            "EmptyKernel",
            runtime,
            profiler_type,
        )

    def name(self):
        return f"ulls_benchmark_{self._runtime.value} EmptyKernel wgc:{self._wgc}, wgs:{self._wgs}{self._cpu_count_str()}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} EmptyKernel, wgc {self._wgc}, wgs {self._wgs}{self._cpu_count_str(separator=',')}"

    def explicit_group(self):
        return f"EmptyKernel, wgc: {self._wgc}, wgs: {self._wgs}{self._cpu_count_str(separator=',')}"

    def description(self) -> str:
        return ""

    def get_tags(self):
        return [runtime_to_tag_name(self._runtime), "micro", "latency", "submit"]

    def _supported_runtimes(self) -> list[RUNTIMES]:
        return [RUNTIMES.SYCL, RUNTIMES.LEVEL_ZERO]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--wgs={self._wgs}",
            f"--wgc={self._wgc}",
            f"--profilerType={self._profiler_type.value}",
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
        self._count = count
        self._kernel_time = kernelTime
        self._barrier = barrier
        self._host_visible = hostVisible
        self._ctr_based_events = ctrBasedEvents
        self._ioq = ioq
        # iterations per bin_args: --iterations=1000
        self._iterations_regular = 1000
        self._iterations_trace = 10
        super().__init__(
            bench, f"ulls_benchmark_{runtime.value}", "KernelSwitch", runtime
        )

    def name(self):
        return f"ulls_benchmark_{self._runtime.value} KernelSwitch count {self._count} kernelTime {self._kernel_time}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} KernelSwitch, count {self._count}, kernelTime {self._kernel_time}"

    def explicit_group(self):
        return f"KernelSwitch, count: {self._count}, kernelTime: {self._kernel_time}"

    def description(self) -> str:
        return ""

    def get_tags(self):
        return [runtime_to_tag_name(self._runtime), "micro", "latency", "submit"]

    def _supported_runtimes(self):
        return [RUNTIMES.SYCL, RUNTIMES.LEVEL_ZERO]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--count={self._count}",
            f"--kernelTime={self._kernel_time}",
            f"--barrier={self._barrier}",
            f"--hostVisible={self._host_visible}",
            f"--ioq={self._ioq}",
            f"--ctrBasedEvents={self._ctr_based_events}",
        ]


class UsmMemoryAllocation(ComputeBenchmark):
    def __init__(
        self, bench, runtime: RUNTIMES, usm_memory_placement, size, measure_mode
    ):
        self._usm_memory_placement = usm_memory_placement
        self._size = size
        self._measure_mode = measure_mode
        # iterations per bin_args: --iterations=10000
        self._iterations_regular = 10000
        self._iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "UsmMemoryAllocation",
            runtime,
        )

    def name(self):
        return (
            f"api_overhead_benchmark_{self._runtime.value} UsmMemoryAllocation "
            f"usmMemoryPlacement:{self._usm_memory_placement} size:{self._size} measureMode:{self._measure_mode}"
        )

    def display_name(self) -> str:
        return (
            f"{self._runtime.value.upper()} UsmMemoryAllocation, "
            f"usmMemoryPlacement {self._usm_memory_placement}, size {self._size}, measureMode {self._measure_mode}"
        )

    def explicit_group(self):
        return f"UsmMemoryAllocation"

    def description(self) -> str:
        what_is_measured = "Both memory allocation and memory free are timed"
        if self._measure_mode == "Allocate":
            what_is_measured = "Only memory allocation is timed"
        elif self._measure_mode == "Free":
            what_is_measured = "Only memory free is timed"
        return (
            f"Measures memory allocation overhead by allocating {self._size} bytes of "
            f"usm {self._usm_memory_placement} memory and free'ing it immediately. "
            f"{what_is_measured}. "
        )

    def get_tags(self):
        return [runtime_to_tag_name(self._runtime), "micro", "latency", "memory"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self._usm_memory_placement}",
            f"--size={self._size}",
            f"--measureMode={self._measure_mode}",
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
        self._usm_memory_placement = usm_memory_placement
        self._allocation_count = allocation_count
        self._size = size
        self._measure_mode = measure_mode
        # iterations per bin_args: --iterations=1000
        self._iterations_regular = 1000
        self._iterations_trace = 10
        super().__init__(
            bench,
            f"api_overhead_benchmark_{runtime.value}",
            "UsmBatchMemoryAllocation",
            runtime,
        )

    def name(self):
        return (
            f"api_overhead_benchmark_{self._runtime.value} UsmBatchMemoryAllocation "
            f"usmMemoryPlacement:{self._usm_memory_placement} allocationCount:{self._allocation_count} size:{self._size} measureMode:{self._measure_mode}"
        )

    def display_name(self) -> str:
        return (
            f"{self._runtime.value.upper()} UsmBatchMemoryAllocation, "
            f"usmMemoryPlacement {self._usm_memory_placement}, allocationCount {self._allocation_count}, size {self._size}, measureMode {self._measure_mode}"
        )

    def explicit_group(self):
        return f"UsmBatchMemoryAllocation"

    def description(self) -> str:
        what_is_measured = "Both memory allocation and memory free are timed"
        if self._measure_mode == "Allocate":
            what_is_measured = "Only memory allocation is timed"
        elif self._measure_mode == "Free":
            what_is_measured = "Only memory free is timed"
        return (
            f"Measures memory allocation overhead by allocating {self._size} bytes of "
            f"usm {self._usm_memory_placement} memory {self._allocation_count} times, then free'ing it all at once. "
            f"{what_is_measured}. "
        )

    def get_tags(self):
        return [runtime_to_tag_name(self._runtime), "micro", "latency", "memory"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--type={self._usm_memory_placement}",
            f"--allocationCount={self._allocation_count}",
            f"--size={self._size}",
            f"--measureMode={self._measure_mode}",
        ]


class GraphApiFinalizeGraph(ComputeBenchmark):
    def __init__(
        self,
        bench,
        runtime: RUNTIMES,
        rebuild_graph_every_iteration,
        graph_structure,
    ):
        self._rebuild_graph_every_iteration = rebuild_graph_every_iteration
        self._graph_structure = graph_structure
        # base iterations value mirrors previous behaviour
        base_iters = 10000
        if graph_structure == "Llama":
            base_iters = base_iters // 10
        self._iterations_regular = base_iters
        self._iterations_trace = min(base_iters, 10)

        super().__init__(
            bench,
            f"graph_api_benchmark_{runtime.value}",
            "FinalizeGraph",
            runtime,
        )

    def name(self):
        return f"graph_api_benchmark_{self._runtime.value} FinalizeGraph rebuildGraphEveryIter:{self._rebuild_graph_every_iteration} graphStructure:{self._graph_structure}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} FinalizeGraph, rebuildGraphEveryIter {self._rebuild_graph_every_iteration}, graphStructure {self._graph_structure}"

    def explicit_group(self):
        return f"FinalizeGraph, GraphStructure: {self._graph_structure}"

    def description(self) -> str:
        what_is_measured = ""

        if self._rebuild_graph_every_iteration == 0:
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
            f"structure based on the usage of graphs in {self._graph_structure}. "
            f"{what_is_measured}"
        )

    def get_tags(self):
        return [
            "graph",
            runtime_to_tag_name(self._runtime),
            "micro",
            "finalize",
            "latency",
        ]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        return [
            f"--iterations={iters}",
            f"--rebuildGraphEveryIter={self._rebuild_graph_every_iteration}",
            f"--graphStructure={self._graph_structure}",
        ]
