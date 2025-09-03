# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from pathlib import Path
from enum import Enum
from utils.result import BenchmarkMetadata, BenchmarkTag, Result
from options import options
from utils.utils import download, run
from abc import ABC, abstractmethod
from utils.unitrace import get_unitrace
from utils.flamegraph import get_flamegraph
from utils.logger import log


class TracingType(Enum):
    """Enumeration of available tracing types."""

    NONE = ""
    UNITRACE = "unitrace"
    FLAMEGRAPH = "flamegraph"


benchmark_tags = [
    BenchmarkTag("SYCL", "Benchmark uses SYCL runtime"),
    BenchmarkTag("UR", "Benchmark uses Unified Runtime API"),
    BenchmarkTag("L0", "Benchmark uses Level Zero API directly"),
    BenchmarkTag("UMF", "Benchmark uses Unified Memory Framework directly"),
    BenchmarkTag("micro", "Microbenchmark focusing on a specific functionality"),
    BenchmarkTag("application", "Real application-based performance test"),
    BenchmarkTag("proxy", "Benchmark that simulates real application use-cases"),
    BenchmarkTag("submit", "Tests kernel submission performance"),
    BenchmarkTag("math", "Tests math computation performance"),
    BenchmarkTag("memory", "Tests memory transfer or bandwidth performance"),
    BenchmarkTag("allocation", "Tests memory allocation performance"),
    BenchmarkTag("graph", "Tests graph-based execution performance"),
    BenchmarkTag("latency", "Measures operation latency"),
    BenchmarkTag("throughput", "Measures operation throughput"),
    BenchmarkTag("inference", "Tests ML/AI inference performance"),
    BenchmarkTag("image", "Image processing benchmark"),
    BenchmarkTag("simulation", "Physics or scientific simulation benchmark"),
]

benchmark_tags_dict = {tag.name: tag for tag in benchmark_tags}


class Benchmark(ABC):
    def __init__(self, directory, suite):
        self.directory = directory
        self.suite = suite

    @abstractmethod
    def name(self) -> str:
        pass

    def display_name(self) -> str:
        """Returns a user-friendly name for display in charts.
        By default returns the same as name(), but can be overridden.
        """
        return self.name()

    def explicit_group(self) -> str:
        """Returns the explicit group name for this benchmark, if any.
        Can be modified."""
        return ""

    def enabled(self) -> bool:
        """Returns whether this benchmark is enabled.
        By default, it returns True, but can be overridden to disable a benchmark."""
        return True

    def traceable(self, tracing_type: TracingType) -> bool:
        """Returns whether this benchmark should be traced by the specified tracing method.
        By default, it returns True for all tracing types, but can be overridden
        to disable specific tracing methods for a benchmark.
        """
        return True

    def tracing_enabled(self, run_trace, force_trace, tr_type: TracingType):
        """Returns whether tracing is enabled for the given type."""
        return (self.traceable(tr_type) or force_trace) and run_trace == tr_type

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass

    @abstractmethod
    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        """Execute the benchmark with the given environment variables.

        Args:
            env_vars: Environment variables to use when running the benchmark.
            run_trace: The type of tracing to run (NONE, UNITRACE, or FLAMEGRAPH).
            force_trace: If True, ignore the traceable() method and force tracing.

        Returns:
            A list of Result objects with the benchmark results.

        Raises:
            Exception: If the benchmark fails for any reason.
        """
        pass

    @staticmethod
    def get_adapter_full_path():
        for libs_dir_name in ["lib", "lib64"]:
            adapter_path = os.path.join(
                options.ur, libs_dir_name, f"libur_adapter_{options.ur_adapter}.so"
            )
            if os.path.isfile(adapter_path):
                return adapter_path
        assert (
            False
        ), f"could not find adapter file {adapter_path} (and in similar lib paths)"

    def run_bench(
        self,
        command,
        env_vars,
        ld_library=[],
        add_sycl=True,
        use_stdout=True,
        run_trace: TracingType = TracingType.NONE,
        extra_trace_opt=None,
        force_trace: bool = False,
    ):
        env_vars = env_vars.copy()
        if options.ur is not None:
            env_vars.update(
                {"UR_ADAPTERS_FORCE_LOAD": Benchmark.get_adapter_full_path()}
            )

        env_vars.update(options.extra_env_vars)

        ld_libraries = options.extra_ld_libraries.copy()
        ld_libraries.extend(ld_library)

        unitrace_output = None
        if self.tracing_enabled(run_trace, force_trace, TracingType.UNITRACE):
            if extra_trace_opt is None:
                extra_trace_opt = []
            unitrace_output, command = get_unitrace().setup(
                self.name(), command, extra_trace_opt
            )
            log.debug(f"Unitrace output: {unitrace_output}")
            log.debug(f"Unitrace command: {' '.join(command)}")

        # flamegraph run

        perf_data_file = None
        if self.tracing_enabled(run_trace, force_trace, TracingType.FLAMEGRAPH):
            perf_data_file, command = get_flamegraph().setup(
                self.name(), self.get_suite_name(), command
            )
            log.debug(f"FlameGraph perf data: {perf_data_file}")
            log.debug(f"FlameGraph command: {' '.join(command)}")

        try:
            result = run(
                command=command,
                env_vars=env_vars,
                add_sycl=add_sycl,
                cwd=options.benchmark_cwd,
                ld_library=ld_libraries,
            )
        except subprocess.CalledProcessError:
            if run_trace == TracingType.UNITRACE and unitrace_output:
                get_unitrace().cleanup(options.benchmark_cwd, unitrace_output)
            if run_trace == TracingType.FLAMEGRAPH and perf_data_file:
                get_flamegraph().cleanup(perf_data_file)
            raise

        if (
            self.tracing_enabled(run_trace, force_trace, TracingType.UNITRACE)
            and unitrace_output
        ):
            get_unitrace().handle_output(unitrace_output)

        if (
            self.tracing_enabled(run_trace, force_trace, TracingType.FLAMEGRAPH)
            and perf_data_file
        ):
            svg_file = get_flamegraph().handle_output(
                self.name(), perf_data_file, self.get_suite_name()
            )
            log.info(f"FlameGraph generated: {svg_file}")

        if use_stdout:
            return result.stdout.decode()
        else:
            return result.stderr.decode()

    def create_data_path(self, name, skip_data_dir=False):
        if skip_data_dir:
            data_path = os.path.join(self.directory, name)
        else:
            data_path = os.path.join(self.directory, "data", name)
            if options.redownload and Path(data_path).exists():
                shutil.rmtree(data_path)

        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def download(
        self,
        name,
        url,
        file,
        untar=False,
        unzip=False,
        skip_data_dir=False,
        checksum="",
    ):
        self.data_path = self.create_data_path(name, skip_data_dir)
        return download(self.data_path, url, file, untar, unzip, checksum)

    def lower_is_better(self):
        return True

    def stddev_threshold(self):
        return None

    def get_suite_name(self) -> str:
        return self.suite.name()

    def description(self):
        return ""

    def notes(self) -> str:
        return None

    def unstable(self) -> str:
        return None

    def get_tags(self) -> list[str]:
        return []

    def range(self) -> tuple[float, float]:
        return None

    def get_metadata(self) -> dict[str, BenchmarkMetadata]:
        range = self.range()

        return {
            self.name(): BenchmarkMetadata(
                type="benchmark",
                description=self.description(),
                notes=self.notes(),
                unstable=self.unstable(),
                tags=self.get_tags(),
                range_min=range[0] if range else None,
                range_max=range[1] if range else None,
                display_name=self.display_name(),
                explicit_group=self.explicit_group(),
            )
        }


class Suite(ABC):
    @abstractmethod
    def benchmarks(self) -> list[Benchmark]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def setup(self) -> None:
        return

    def additional_metadata(self) -> dict[str, BenchmarkMetadata]:
        return {}
