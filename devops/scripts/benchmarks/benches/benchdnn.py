# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from pathlib import Path

from .base import Suite, Benchmark, TracingType
from options import options
from utils.result import Result
from utils.oneapi import get_oneapi
from utils.logger import log
from .benchdnn_list import get_bench_dnn_list
from git_project import GitProject


class OneDnnBench(Suite):
    def __init__(self):
        self.project = None

    def git_url(self):
        return "https://github.com/uxlfoundation/oneDNN.git"

    def git_tag(self):
        # 7 Nov, 2025
        return "v3.10"

    def name(self):
        return "BenchDNN"

    def benchmarks(self) -> list:
        benchmarks = []
        for entry in get_bench_dnn_list():
            rungraph = True
            if len(entry) == 3:
                bench_driver, bench_name, bench_args = entry
            elif len(entry) == 4:
                bench_driver, bench_name, bench_args, rungraph = entry
            else:
                raise ValueError(
                    f"Invalid benchmark entry: {entry}. Expected 3 elements."
                )

            # Create a benchmark instance for both eager and graph execution modes
            benchmarks.append(
                OneDnnBenchmark(
                    self, bench_driver, bench_name, bench_args, syclgraph=False
                )
            )
            if rungraph == True:
                benchmarks.append(
                    OneDnnBenchmark(
                        self, bench_driver, bench_name, bench_args, syclgraph=True
                    )
                )
        return benchmarks

    def setup(self) -> None:
        if options.sycl is None:
            return

        self.oneapi = get_oneapi()
        if self.project is None:
            self.project = GitProject(
                self.git_url(),
                self.git_tag(),
                Path(options.workdir),
                "onednn",
                force_rebuild=True,
            )

        extra_cmake_args = [
            f"-DCMAKE_PREFIX_PATH={options.sycl}",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_C_COMPILER=clang",
            "-DDNNL_BUILD_TESTS=ON",
            "-DDNNL_BUILD_EXAMPLES=OFF",
            "-DDNNL_CPU_RUNTIME=NONE",  # Disable SYCL CPU support
            "-DDNNL_GPU_RUNTIME=SYCL",  # Enable SYCL GPU support
        ]
        self.project.configure(
            extra_cmake_args,
            install_prefix=False,
            add_sycl=True,
        )
        self.project.build(
            target="benchdnn",
            add_sycl=True,
            ld_library=[str(self.project.build_dir / "src")]
            + self.oneapi.ld_libraries(),
            timeout=60 * 20,
        )

    def teardown(self):
        pass


class OneDnnBenchmark(Benchmark):
    def __init__(self, suite, bench_driver, bench_name, bench_args, syclgraph=True):
        self.suite = suite
        self.bench_name = f"{bench_driver}-{bench_name}"
        self.bench_args = f"--{bench_driver} --mode=P --engine=gpu --max-ms-per-prb=100"

        self.exp_group = self.bench_name
        if syclgraph:
            self.bench_args += " --execution-mode=graph"
            self.bench_name += "-graph"
        else:
            self.bench_args += " --execution-mode=direct"
            self.bench_name += "-eager"
        self.bench_args += f" {bench_args}"

    @property
    def benchmark_bin(self) -> Path:
        return self.suite.project.build_dir / "tests" / "benchdnn" / "benchdnn"

    def enabled(self):
        if options.sycl is None:
            return False
        if options.ur_adapter == "cuda" or options.ur_adapter == "hip":
            return False
        return True

    def name(self):
        return f"onednn-{self.bench_name}"

    def explicit_group(self) -> str:
        return self.exp_group

    def setup(self):
        if not self.benchmark_bin.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {self.benchmark_bin}")

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        # Determine extra trace options based on tracing type
        if run_trace == TracingType.UNITRACE:
            extra_trace_opt = ["--chrome-dnn-logging"]
        else:
            extra_trace_opt = None

        command = [
            str(self.benchmark_bin),
            *self.bench_args.split(),
        ]

        ld_library = self.suite.oneapi.ld_libraries() + [
            str(self.suite.project.build_dir / "src")
        ]

        env_vars = dict(env_vars) if env_vars else {}
        env_vars["ONEAPI_DEVICE_SELECTOR"] = "level_zero:*"

        output = self.run_bench(
            command,
            env_vars,
            add_sycl=True,
            ld_library=ld_library,
            use_stdout=True,
            run_trace=run_trace,
            extra_trace_opt=extra_trace_opt,
            force_trace=force_trace,
        )
        result_value = self._extract_time(output)

        log.debug(f"[{self.name()}] Output: {output}")

        return [
            Result(
                label=self.name(),
                value=result_value,
                unit="ms",
                command=command,
                env=env_vars,
                git_url=self.suite.git_url(),
                git_hash=self.suite.git_tag(),
            )
        ]

    # example output:
    # Output template: perf,%engine%,%0time%,%-ops%,%-MB%,%-pr
    # perf,gpu,0.000000,0.000000,0.000000,0
    # perf,gpu,0.000000,0.000000,0.000000,0
    def _extract_time(self, output):
        lines = output.splitlines()
        idx_time = None
        values = []
        for i, line in enumerate(lines):
            if line.startswith("Output template:"):
                template = line.replace("Output template: ", "").strip().split(",")
                try:
                    idx_time = template.index("%0time%")
                except ValueError:
                    return 0.0
                continue
            if idx_time is not None and line.startswith("perf,"):
                fields = line.strip().split(",")
                if len(fields) > idx_time:
                    try:
                        values.append(float(fields[idx_time]))
                    except Exception:
                        continue
        if values:
            return sum(values)
        return 0.0

    def teardown(self):
        pass
