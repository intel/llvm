# Copyright (C) 2024-2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from itertools import product
from pathlib import Path

from git_project import GitProject
from options import options
from utils.result import BenchmarkMetadata
from utils.logger import log

from ..base import Benchmark, Suite, TracingType
from .compute_benchmark import ComputeBenchmark
from .compute_enums import RUNTIMES, PROFILERS, KERNEL_NAME, runtime_to_tag_name
from .compute_metadata import ComputeMetadataGenerator
from .compute_torch import *


def runtime_to_name(runtime: RUNTIMES) -> str:
    return {
        RUNTIMES.SYCL_PREVIEW: "SYCL Preview",
        RUNTIMES.SYCL: "SYCL",
        RUNTIMES.LEVEL_ZERO: "Level Zero",
        RUNTIMES.UR: "Unified Runtime",
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
        # Mar 23, 2026
        return "86d86fd37d703db4f0f75779ccdfd50193e0ab3d"

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

        if not self._project.needs_rebuild():
            log.info(f"Rebuilding {self._project.name} skipped")
            return

        extra_args = [
            f"-DBUILD_SYCL=ON",
            f"-DSYCL_COMPILER_ROOT={options.sycl}",
            f"-DALLOW_WARNINGS=ON",
            f"-DUSE_SYSTEM_LEVEL_ZERO=OFF",
            f"-DCMAKE_CXX_COMPILER=clang++",
            f"-DCMAKE_C_COMPILER=clang",
            f"-DBUILD_UR=ON",
            f"-DCMAKE_PREFIX_PATH={options.sycl}",
        ]

        is_gdb_mode = os.environ.get("LLVM_BENCHMARKS_USE_GDB", "") == "1"
        if is_gdb_mode:
            extra_args += [
                f"-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG -fdebug-info-for-profiling",
            ]

        if options.ur_adapter == "cuda":
            extra_args += [
                "-DBUILD_SYCL_WITH_CUDA=ON",
                "-DBUILD_L0=OFF",
                "-DBUILD_OCL=OFF",
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
            # SYCL only supports graph mode, UR & L0 support both emulated
            # and non-emulated graph APIs.
            if runtime == RUNTIMES.SYCL or runtime == RUNTIMES.SYCL_PREVIEW:
                emulate_graphs = [0]
            else:  # level-zero and unified-runtime
                # SubmitGraph with L0 / UR graph segfaults on PVC
                device_arch = getattr(options, "device_architecture", "")
                # UR currently only supports EmulateGraphs=0 with in-order queue and Level-Zero V2 Adapter
                skip_ur_native_graph = runtime == RUNTIMES.UR and (
                    in_order_queue == 0 or options.ur_adapter != "level_zero_v2"
                )
                emulate_graphs = (
                    [1] if "pvc" in device_arch or skip_ur_native_graph else [0, 1]
                )
            for emulate_graph in emulate_graphs:
                benches.append(
                    GraphApiSubmitGraph(
                        self,
                        runtime,
                        in_order_queue,
                        num_kernels,
                        measure_completion_time,
                        use_events,
                        emulate_graph,
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
                            emulate_graph,
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
                ExecImmCopy(self, 0, 1, "Device", "Device", 1024, profiler_type)
            )
            benches.append(
                ExecImmCopy(self, 1, 1, "Device", "Host", 1024, profiler_type)
            )

        # Add RecordAndReplay benchmarks
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

        # Add TorchSingleQueue benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchSingleQueueBench(variant_name: str, **kwargs):
                    return TorchSingleQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "KernelName": "Add",
                            "KernelParamsNum": 5,
                            "KernelSubmitPattern": "Single",
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchSingleQueueBench(
                        "Int32Large",
                        KernelDataType="Int32",
                        KernelWGCount=4096,
                        KernelWGSize=512,
                    ),
                    createTorchSingleQueueBench(
                        "Int32Medium",
                        KernelDataType="Int32",
                        KernelWGCount=512,
                        KernelWGSize=256,
                    ),
                    createTorchSingleQueueBench(
                        "Int32Small",
                        KernelDataType="Int32",
                        KernelWGCount=256,
                        KernelWGSize=128,
                    ),
                    createTorchSingleQueueBench(
                        "MixedLarge",
                        KernelDataType="Mixed",
                        KernelWGCount=4096,
                        KernelWGSize=512,
                    ),
                    createTorchSingleQueueBench(
                        "MixedMedium",
                        KernelDataType="Mixed",
                        KernelWGCount=512,
                        KernelWGSize=256,
                    ),
                    createTorchSingleQueueBench(
                        "MixedSmall",
                        KernelDataType="Mixed",
                        KernelWGCount=256,
                        KernelWGSize=128,
                    ),
                ]

        # Add TorchMultiQueue benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type, measure_completion in product(list(PROFILERS), [0, 1]):

                def createTorchMultiQueueBench(variant_name: str, **kwargs):
                    return TorchMultiQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        measure_completion,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchMultiQueueBench(
                        "large",
                        KernelsPerQueue=20,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchMultiQueueBench(
                        "medium",
                        KernelsPerQueue=10,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchMultiQueueBench(
                        "small",
                        KernelsPerQueue=4,
                        MeasureCompletionTime=measure_completion,
                    ),
                ]

        # Add TorchSlmSize benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type, measure_completion in product(list(PROFILERS), [0, 1]):

                def createTorchSlmSizeBench(variant_name: str, **kwargs):
                    return TorchSlmSize(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        measure_completion,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "Profiling": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchSlmSizeBench(
                        "small",
                        SlmNum=1,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchSlmSizeBench(
                        "medium",
                        SlmNum=1024,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchSlmSizeBench(
                        "large",
                        SlmNum=16384,
                        MeasureCompletionTime=measure_completion,
                    ),
                ]

        # Add TorchMemoryReuse benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchMemoryReuseBench(variant_name: str, **kwargs):
                    return TorchMemoryReuse(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        **kwargs,
                    )

                benches += [
                    createTorchMemoryReuseBench(
                        "Int32Large",
                        KernelBatchSize=4096,
                        KernelDataType="Int32",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "Int32Medium",
                        KernelBatchSize=512,
                        KernelDataType="Int32",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "FloatLarge",
                        KernelBatchSize=4096,
                        KernelDataType="Float",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "FloatMedium",
                        KernelBatchSize=512,
                        KernelDataType="Float",
                        UseEvents=0,
                        Profiling=0,
                    ),
                ]

        # Add TorchLinearKernelSize benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchLinearKernelSizeBench(variant_name: str, **kwargs):
                    return TorchLinearKernelSize(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "Profiling": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchLinearKernelSizeBench(
                        "array32",
                        KernelSize=32,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array128",
                        KernelSize=128,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array512",
                        KernelSize=512,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array1024",
                        KernelSize=1024,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array5120",
                        KernelSize=5120,
                    ),
                ]

        # Add TorchEventRecordWait benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            for profiler_type in list(PROFILERS):
                benches.append(
                    TorchEventRecordWait(
                        self,
                        runtime,
                        "medium",
                        profiler_type,
                        Profiling=0,
                        KernelWGCount=256,
                        KernelWGSize=512,
                    )
                )

        #
        # Note: Graph benchmarks segfault on pvc on L0
        #
        device_arch = getattr(options, "device_architecture", "")

        # Add TorchGraphSingleQueue benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            if "pvc" in device_arch and runtime == RUNTIMES.LEVEL_ZERO:
                continue

            for profiler_type, kernel_name in product(
                list(PROFILERS), list(KERNEL_NAME)
            ):

                def createTorchGraphSingleQueueBench(variant_name: str, **kwargs):
                    return TorchGraphSingleQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphSingleQueueBench(
                        "small",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=10,
                        KernelBatchSize=10,
                    ),
                    createTorchGraphSingleQueueBench(
                        "medium",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=32,
                        KernelBatchSize=32,
                    ),
                    createTorchGraphSingleQueueBench(
                        "large",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=64,
                        KernelBatchSize=64,
                    ),
                ]

        # Add TorchGraphMultiQueue benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            if "pvc" in device_arch and runtime == RUNTIMES.LEVEL_ZERO:
                continue

            for profiler_type in list(PROFILERS):

                def createTorchGraphMultiQueueBench(variant_name: str, **kwargs):
                    return TorchGraphMultiQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphMultiQueueBench(
                        "small",
                        KernelsPerQueue=10,
                    ),
                    createTorchGraphMultiQueueBench(
                        "medium",
                        KernelsPerQueue=32,
                    ),
                    createTorchGraphMultiQueueBench(
                        "large",
                        KernelsPerQueue=64,
                    ),
                ]

        # Add TorchGraphVllmMock benchmarks
        for runtime in filter(lambda x: x != RUNTIMES.UR, RUNTIMES):
            if "pvc" in device_arch and runtime == RUNTIMES.LEVEL_ZERO:
                continue

            for profiler_type in list(PROFILERS):

                def createTorchGraphVllmMockBench(variant_name: str, **kwargs):
                    return TorchGraphVllmMock(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphVllmMockBench(
                        "small", AllocCount=32, GraphScenario=0
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=0
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=1
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=2
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=3
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


class ComputeBenchCore(ComputeBench):
    """
    A suite for core compute benchmarks scenarios for quick runs.
    """

    def name(self) -> str:
        return "Compute Benchmarks Core"

    def benchmarks(self) -> list[Benchmark]:
        core_benches = []
        submit_kernel_params = product(
            list(RUNTIMES),
            [0, 1],  # in_order_queue
            list(PROFILERS),
        )
        for (
            runtime,
            in_order_queue,
            profiler_type,
        ) in submit_kernel_params:
            core_benches.append(
                SubmitKernel(
                    self,
                    runtime,
                    in_order_queue,
                    KernelExecTime=1,
                    profiler_type=profiler_type,
                )
            )
        return core_benches


class ComputeBenchNative(ComputeBench):
    """
    A suite for torch compute benchmarks scenarios using native graphs from
    custom llvm branch. It's required to pass that custom llvm build path
    via '--sycl' option to get proper results off of this benchmarks.
    """

    def name(self) -> str:
        return "Compute Benchmarks Native"

    def benchmarks(self) -> list[Benchmark]:
        benches = []

        # Add TorchSingleQueue benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchSingleQueueBench(variant_name: str, **kwargs):
                    return TorchSingleQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "KernelName": "Add",
                            "KernelParamsNum": 5,
                            "KernelSubmitPattern": "Single",
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchSingleQueueBench(
                        "Int32Large",
                        KernelDataType="Int32",
                        KernelWGCount=4096,
                        KernelWGSize=512,
                    ),
                    createTorchSingleQueueBench(
                        "Int32Medium",
                        KernelDataType="Int32",
                        KernelWGCount=512,
                        KernelWGSize=256,
                    ),
                    createTorchSingleQueueBench(
                        "Int32Small",
                        KernelDataType="Int32",
                        KernelWGCount=256,
                        KernelWGSize=128,
                    ),
                    createTorchSingleQueueBench(
                        "MixedLarge",
                        KernelDataType="Mixed",
                        KernelWGCount=4096,
                        KernelWGSize=512,
                    ),
                    createTorchSingleQueueBench(
                        "MixedMedium",
                        KernelDataType="Mixed",
                        KernelWGCount=512,
                        KernelWGSize=256,
                    ),
                    createTorchSingleQueueBench(
                        "MixedSmall",
                        KernelDataType="Mixed",
                        KernelWGCount=256,
                        KernelWGSize=128,
                    ),
                ]

        # Add TorchMultiQueue benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type, measure_completion in product(list(PROFILERS), [0, 1]):

                def createTorchMultiQueueBench(variant_name: str, **kwargs):
                    return TorchMultiQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        measure_completion,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchMultiQueueBench(
                        "large",
                        KernelsPerQueue=20,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchMultiQueueBench(
                        "medium",
                        KernelsPerQueue=10,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchMultiQueueBench(
                        "small",
                        KernelsPerQueue=4,
                        MeasureCompletionTime=measure_completion,
                    ),
                ]

        # Add TorchSlmSize benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type, measure_completion in product(list(PROFILERS), [0, 1]):

                def createTorchSlmSizeBench(variant_name: str, **kwargs):
                    return TorchSlmSize(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        measure_completion,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "Profiling": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchSlmSizeBench(
                        "small",
                        SlmNum=1,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchSlmSizeBench(
                        "medium",
                        SlmNum=1024,
                        MeasureCompletionTime=measure_completion,
                    ),
                    createTorchSlmSizeBench(
                        "large",
                        SlmNum=16384,
                        MeasureCompletionTime=measure_completion,
                    ),
                ]

        # Add TorchMemoryReuse benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchMemoryReuseBench(variant_name: str, **kwargs):
                    return TorchMemoryReuse(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        **kwargs,
                    )

                benches += [
                    createTorchMemoryReuseBench(
                        "Int32Large",
                        KernelBatchSize=4096,
                        KernelDataType="Int32",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "Int32Medium",
                        KernelBatchSize=512,
                        KernelDataType="Int32",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "FloatLarge",
                        KernelBatchSize=4096,
                        KernelDataType="Float",
                        UseEvents=0,
                        Profiling=0,
                    ),
                    createTorchMemoryReuseBench(
                        "FloatMedium",
                        KernelBatchSize=512,
                        KernelDataType="Float",
                        UseEvents=0,
                        Profiling=0,
                    ),
                ]

        # Add TorchLinearKernelSize benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchLinearKernelSizeBench(variant_name: str, **kwargs):
                    return TorchLinearKernelSize(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelBatchSize": 512,
                            "Profiling": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchLinearKernelSizeBench(
                        "array32",
                        KernelSize=32,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array128",
                        KernelSize=128,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array512",
                        KernelSize=512,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array1024",
                        KernelSize=1024,
                    ),
                    createTorchLinearKernelSizeBench(
                        "array5120",
                        KernelSize=5120,
                    ),
                ]

        #
        # Note: Graph benchmarks segfault on pvc on L0
        #
        device_arch = getattr(options, "device_architecture", "")

        # Add TorchGraphSingleQueue benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type, kernel_name in product(
                list(PROFILERS), list(KERNEL_NAME)
            ):

                def createTorchGraphSingleQueueBench(variant_name: str, **kwargs):
                    return TorchGraphSingleQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphSingleQueueBench(
                        "small",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=10,
                        KernelBatchSize=10,
                    ),
                    createTorchGraphSingleQueueBench(
                        "medium",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=32,
                        KernelBatchSize=32,
                    ),
                    createTorchGraphSingleQueueBench(
                        "large",
                        KernelName=kernel_name.value,
                        KernelsPerQueue=64,
                        KernelBatchSize=64,
                    ),
                ]

        # Add TorchGraphMultiQueue benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchGraphMultiQueueBench(variant_name: str, **kwargs):
                    return TorchGraphMultiQueue(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphMultiQueueBench(
                        "small",
                        KernelsPerQueue=10,
                    ),
                    createTorchGraphMultiQueueBench(
                        "medium",
                        KernelsPerQueue=32,
                    ),
                    createTorchGraphMultiQueueBench(
                        "large",
                        KernelsPerQueue=64,
                    ),
                ]

        # Add TorchGraphVllmMock benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):

                def createTorchGraphVllmMockBench(variant_name: str, **kwargs):
                    return TorchGraphVllmMock(
                        self,
                        runtime,
                        variant_name,
                        profiler_type,
                        fixed_args={
                            "KernelWGCount": 512,
                            "KernelWGSize": 256,
                            "Profiling": 0,
                            "UseEvents": 0,
                        },
                        **kwargs,
                    )

                benches += [
                    createTorchGraphVllmMockBench(
                        "small", AllocCount=32, GraphScenario=0
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=0
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=1
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=2
                    ),
                    createTorchGraphVllmMockBench(
                        "large", AllocCount=128, GraphScenario=3
                    ),
                ]

        # Add TorchSubmitEventRecordWait benchmarks
        for runtime in filter(lambda x: (x != RUNTIMES.UR and x != RUNTIMES.LEVEL_ZERO), RUNTIMES):
            for profiler_type in list(PROFILERS):
                benches.append(
                    TorchSubmitEventRecordWait(
                        self,
                        runtime,
                        "medium",
                        profiler_type,
                        Profiling=0,
                        KernelWGCount=256,
                        KernelWGSize=512,
                    )
                )

        return benches


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


class ExecImmCopy(ComputeBenchmark):
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
            "ExecImmCopy",
            profiler_type=profiler_type,
        )

    def name(self):
        order = "in order" if self._ioq else "out of order"
        return f"api_overhead_benchmark_sycl ExecImmCopy {order} from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str()}"

    def display_name(self) -> str:
        order = "in order" if self._ioq else "out of order"
        return f"SYCL ExecImmCopy {order} from {self._source} to {self._destination}, size {self._size}{self._cpu_count_str(separator=',')}"

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
            f"--src={self._source}",
            f"--dst={self._destination}",
            f"--size={self._size}",
            "--CopyOffload=0",
            f"--profilerType={self._profiler_type.value}",
        ]


class RecordAndReplay(ComputeBenchmark):
    def __init__(
        self, suite, runtime: RUNTIMES, variant_name: str, profiler_type, **kwargs
    ):
        self._variant_name = variant_name
        self._rr_params = kwargs
        self._iterations_regular = 1000
        self._iterations_trace = 10
        super().__init__(
            suite,
            f"record_and_replay_benchmark_{runtime.value}",
            "RecordGraph",
            runtime,
            profiler_type,
        )

    def name(self):
        ret = []
        for k, v in self._rr_params.items():
            if k[0] == "n":  # numeric parameter
                ret.append(f"{k[1:]} {v}")
            elif k[0] == "m":
                if v != 0:  # measure parameter
                    ret.append(f"{k[1:]}")
            else:  # boolean parameter
                if v != 0:
                    ret.append(k)
        ret.sort()
        return self._bench_name + " " + ", ".join(ret)

    def display_name(self) -> str:
        return f"{self.explicit_group()}_{self._runtime.value}"

    def explicit_group(self):
        return f"{self._test} {self._variant_name}"

    def get_tags(self):
        return ["L0"]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        return [f"--{k}={v}" for k, v in self._rr_params.items()]


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
            f"--memory={self._placement}",
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
        self._native_str = (
            " native recording"
            if self._emulate_graphs == 0
            and (runtime == RUNTIMES.UR or runtime == RUNTIMES.LEVEL_ZERO)
            else ""
        )
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
        return f"graph_api_benchmark_{self._runtime.value} SubmitGraph{self._native_str}{self._use_events_str}{self._host_tasks_str} numKernels:{self._num_kernels} ioq {self._in_order_queue} MeasureCompletionTime {self._measure_completion_time}{self._cpu_count_str()}"

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} SubmitGraph{self._native_str} {self._ioq_str}{self._measure_str}{self._use_events_str}{self._host_tasks_str}, {self._num_kernels} kernels{self._cpu_count_str(separator=',')}"

    def explicit_group(self):
        return f"SubmitGraph{self._native_str} {self._ioq_str}{self._measure_str}{self._use_events_str}{self._host_tasks_str}, {self._num_kernels} kernels{self._cpu_count_str(separator=',')}"

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
