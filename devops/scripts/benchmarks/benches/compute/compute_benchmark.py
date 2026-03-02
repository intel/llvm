# Copyright (C) 2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv
import io
import math
import os

from pathlib import Path

from options import options
from utils.logger import log
from utils.result import Result

from ..base import Benchmark, Suite, TracingType
from .compute_enums import RUNTIMES, PROFILERS


class ComputeBenchmark(Benchmark):
    def __init__(
        self,
        suite: Suite,
        name: str,
        test: str,
        runtime: RUNTIMES | None = None,
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
        env_vars = dict(env_vars) if env_vars else {}
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

    def explicit_group(self) -> str:
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
        is_gdb_mode = os.environ.get("LLVM_BENCHMARKS_USE_GDB", "") == "1"

        if is_gdb_mode:
            log.info(output)
            return [(0.0, 0.0)]

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
