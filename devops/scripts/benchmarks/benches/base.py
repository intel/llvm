# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import os
import shutil
import subprocess
from pathlib import Path
from utils.result import BenchmarkMetadata, BenchmarkTag, Result
from options import options
from utils.utils import download, run
from abc import ABC, abstractmethod
import glob

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

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass

    @abstractmethod
    def run(
        self, env_vars, unitrace_timestamp: str = None
    ) -> list[Result]:
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
        unitrace_timestamp: str = None,
        extra_unitrace_opt=[],
    ):
        env_vars = env_vars.copy()
        if options.ur is not None:
            env_vars.update(
                {"UR_ADAPTERS_FORCE_LOAD": Benchmark.get_adapter_full_path()}
            )

        env_vars.update(options.extra_env_vars)

        ld_libraries = options.extra_ld_libraries.copy()
        ld_libraries.extend(ld_library)

        if unitrace_timestamp is not None:
            unitrace_bin = os.path.join(options.workdir, "unitrace-build", "unitrace")
            if not os.path.exists(unitrace_bin):
                raise FileNotFoundError(f"Unitrace binary not found: {unitrace_bin}. ")
            if not os.path.exists(options.unitrace_res_dir):
                os.makedirs(options.unitrace_res_dir)
            bench_dir = f"{options.unitrace_res_dir}/{self.name()}"
            os.makedirs(bench_dir, exist_ok=True)

            unitrace_output = f"{bench_dir}/{self.name()}_{unitrace_timestamp}"
            command = (
                [
                    str(unitrace_bin),
                    "--call-logging",
                    "--host-timing",
                    "--chrome-sycl-logging",
                    "--chrome-call-logging",
                    "--chrome-kernel-logging",
                    "--output",
                    unitrace_output,
                ]
                + extra_unitrace_opt
                + command
            )
            if options.verbose:
                print(f"Unitrace cmd: {' '.join(command)}")

        result = run(
            command=command,
            env_vars=env_vars,
            add_sycl=add_sycl,
            cwd=options.benchmark_cwd,
            ld_library=ld_libraries,
        )

        if unitrace_timestamp is not None:
            handle_unitrace_output(bench_dir, unitrace_output, unitrace_timestamp)

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

    def setup(self):
        return

    def additional_metadata(self) -> dict[str, BenchmarkMetadata]:
        return {}


def handle_unitrace_output(bench_dir, unitrace_output, timestamp):
    FILECNT = 20  # Set your desired max file count

    # 1. Handle unitrace_output.{pid} logs: rename to unitrace_output (remove pid)
    for f in os.listdir(bench_dir):
        if f.startswith(os.path.basename(unitrace_output) + "."):
            parts = f.rsplit(".", 1)
            if (
                len(parts) == 2
                and parts[1].isdigit()
                and os.path.isfile(os.path.join(bench_dir, f))
            ):
                src = os.path.join(bench_dir, f)
                dst = os.path.join(bench_dir, os.path.basename(unitrace_output))
                shutil.move(src, dst)
                break

    # 2. Handle {name}.{pid}.json files: move and rename to {self.name()}.{timestamp}.json
    pid_json_files = []
    for f in os.listdir(options.benchmark_cwd):
        parts = f.split(".")
        l = len(parts)
        if len(parts) >= 3 and parts[l - 1] == "json" and parts[l - 2].isdigit():
            pid_json_files.append(f)

    if len(pid_json_files) == 1:
        # Extract benchmark name from bench_dir path
        bench_name = os.path.basename(bench_dir)
        dst = f"{bench_dir}/{bench_name}_{timestamp}.json"
        shutil.move(os.path.join(options.benchmark_cwd, pid_json_files[0]), dst)
    elif len(pid_json_files) > 1:
        print(
            f"Warning: Found {len(pid_json_files)} files matching the pattern. Expected 1."
        )

    # Count files in the dir and remove oldest if more than FILECNT
    def extract_timestamp_from_name(filename):
        # Example: onednn-sum-padding-2-graph_20250701_114551
        base = os.path.basename(filename)
        parts = base.rsplit("_", 1)
        if len(parts) == 2:
            ts = parts[1]
            # Remove extension if present (for .json files)
            ts = ts.split(".", 1)[0]
            return ts
        return ""

    files = glob.glob(f"{bench_dir}/*")
    files_with_ts = []
    for f in files:
        ts = extract_timestamp_from_name(f)
        files_with_ts.append((f, ts))
    # Sort by timestamp string (lexicographically, which works for YYYYMMDD_HHMMSS)
    files_with_ts.sort(key=lambda x: x[1])
    sorted_files = [f for f, ts in files_with_ts if ts]

    if len(sorted_files) > FILECNT:
        for f in sorted_files[: len(sorted_files) - FILECNT]:
            os.remove(f)
