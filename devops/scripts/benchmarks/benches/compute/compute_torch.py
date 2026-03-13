# Copyright (C) 2025-2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..base import TracingType
from .compute_benchmark import ComputeBenchmark
from .compute_enums import RUNTIMES, PROFILERS, runtime_to_tag_name


class TorchBenchmark(ComputeBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        bench_name: str,
        variant_name: str,
        profiler_type: PROFILERS,
        measure_completion: int = 0,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        self._variant_name = variant_name
        self._measure_completion_str = (
            " with measure completion" if measure_completion else ""
        )
        self._torch_params: dict = kwargs
        self._fixed_args: dict | None = fixed_args  # args used for charts legends
        self._iterations_regular = 1000
        self._iterations_trace = 10
        super().__init__(
            suite,
            f"torch_benchmark_{runtime.value}",
            bench_name,
            runtime,
            profiler_type,
        )

    def name(self):
        ret = []
        for k, v in self._torch_params.items():
            ret.append(f"{k} {v}")
        ret.sort()
        return (
            self._bench_name
            + " "
            + self._test
            + " "
            + ", ".join(ret)
            + self._cpu_count_str()
        )

    def display_name(self) -> str:
        return f"{self._runtime.value.upper()} {self.explicit_group()}"

    def explicit_group(self):
        return f"{self._test} {self._variant_name}{self._measure_completion_str}{self._cpu_count_str(separator=',')}"

    def get_tags(self):
        return ["pytorch", runtime_to_tag_name(self._runtime)]

    def _supported_runtimes(self) -> list[RUNTIMES]:
        return super()._supported_runtimes() + [RUNTIMES.SYCL_PREVIEW]

    def _bin_args(self, run_trace: TracingType = TracingType.NONE) -> list[str]:
        iters = self._get_iters(run_trace)
        if self._fixed_args is not None:
            params = dict(self._fixed_args | self._torch_params)
        else:
            params = self._torch_params
        return (
            [f"--iterations={iters}"]
            + [f"--profilerType={self._profiler_type.value}"]
            + [f"--{k}={v}" for k, v in params.items()]
        )


class TorchSingleQueue(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitSingleQueue",
            variant_name,
            profiler_type,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchMultiQueue(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        measure_completion: int = 0,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitMultiQueue",
            variant_name,
            profiler_type,
            measure_completion=measure_completion,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchSlmSize(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        measure_completion: int = 0,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitSlmSize",
            variant_name,
            profiler_type,
            measure_completion=measure_completion,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchLinearKernelSize(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitLinearKernelSize",
            variant_name,
            profiler_type,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchMemoryReuse(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitMemoryReuse",
            variant_name,
            profiler_type,
            **kwargs,
        )


class TorchGraphSingleQueue(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitGraphSingleQueue",
            variant_name,
            profiler_type,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchGraphMultiQueue(TorchBenchmark):

    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        fixed_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitGraphMultiQueue",
            variant_name,
            profiler_type,
            fixed_args=fixed_args,
            **kwargs,
        )


class TorchSubmitEventRecordWait(TorchBenchmark):
    def __init__(
        self,
        suite,
        runtime: RUNTIMES,
        variant_name: str,
        profiler_type: PROFILERS,
        **kwargs,
    ):
        super().__init__(
            suite,
            runtime,
            "KernelSubmitEventRecordWait",
            variant_name,
            profiler_type,
            **kwargs,
        )
