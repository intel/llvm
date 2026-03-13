# Copyright (C) 2026 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
from pathlib import Path

from .base import Benchmark, Suite, TracingType
from utils.logger import log
from utils.result import Result
from options import options
from git_project import GitProject


class PtiXptiSuite(Suite):
    def __init__(self):
        self.project = None

    def name(self) -> str:
        return "PTI XPTI Overhead"

    def git_url(self) -> str:
        return "https://github.com/intel/pti-gpu.git"

    def git_hash(self) -> str:
        # Latest master branch - can be updated as needed
        return "master"

    def setup(self) -> None:
        if options.sycl is None:
            return

        if self.project is None:
            self.project = GitProject(
                self.git_url(),
                self.git_hash(),
                Path(options.workdir),
                "pti-gpu",
                use_installdir=False,
            )

        if not self.project.needs_rebuild():
            log.info(f"Rebuilding {self.project.name} skipped")
            return

        # Build only the sdk subdirectory
        build_dir = self.project.src_dir / "sdk" / "build"

        extra_args = [
            f"-DCMAKE_C_COMPILER={options.sycl}/bin/clang",
            f"-DCMAKE_CXX_COMPILER={options.sycl}/bin/clang++",
            "-DCMAKE_CXX_FLAGS=-Wall -Wextra -Wextra-semi -pedantic -Wformat -Wformat-security -Werror=format-security -fstack-protector-strong -D_FORTIFY_SOURCE=2",
            "-DCMAKE_C_FLAGS=-Wall -Wextra -Wextra-semi -pedantic -Wformat -Wformat-security -Werror=format-security -fstack-protector-strong -D_FORTIFY_SOURCE=2",
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-z,relro,-z,now,-z,noexecstack",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,relro,-z,now,-z,noexecstack",
            f"-DCMAKE_PREFIX_PATH={options.sycl}",
        ]

        # Configure the sdk subdirectory
        from utils.utils import run
        run(
            [
                "cmake",
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-S{self.project.src_dir}/sdk",
                f"-B{build_dir}",
            ] + extra_args,
            add_sycl=True,
        )

        # Build
        run(
            f"cmake --build {build_dir} -j {options.build_jobs}",
            add_sycl=True,
        )

    def benchmarks(self) -> list[Benchmark]:
        return [
            XptiOverheadBench(self),
        ]


class XptiOverheadBench(Benchmark):
    def __init__(self, suite: PtiXptiSuite):
        super().__init__(suite)
        self.suite = suite

    def name(self):
        return f"{self.suite.name()} perf-profiling-overhead"

    def enabled(self) -> bool:
        return options.sycl is not None

    def get_tags(self):
        return ["SYCL", "micro", "latency"]

    def description(self) -> str:
        return "Measures XPTI instrumentation overhead in SYCL runtime"

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        """
        Run the PTI XPTI overhead benchmark.

        Note: This method is called multiple times by the framework (controlled by
        --iterations flag, default 3). Each call runs the ctest once, which itself
        performs multiple internal iterations and reports a median value. The framework
        then computes the median of all these median values.

        For stable results, use --iterations=20 or higher.
        """
        build_dir = self.suite.project.src_dir / "sdk" / "build"
        pti_lib_dir = build_dir / "lib"

        # Prepare environment
        env_vars = dict(env_vars) if env_vars else {}

        # Add PTI library to LD_LIBRARY_PATH
        ld_library_path = env_vars.get("LD_LIBRARY_PATH", "")
        if ld_library_path:
            env_vars["LD_LIBRARY_PATH"] = f"{pti_lib_dir}:{ld_library_path}"
        else:
            env_vars["LD_LIBRARY_PATH"] = str(pti_lib_dir)

        # Run the test using ctest
        command = [
            "ctest",
            "-V",
            "-R",
            "perf-profiling-overhead",
        ]

        output = self.run_bench(
            command,
            env_vars,
            add_sycl=True,
            ld_library=[str(pti_lib_dir)],
            run_trace=run_trace,
            force_trace=force_trace,
        )

        # Parse the output
        results = self.parse_output(output, command, env_vars)
        return results

    def parse_output(self, output, command, env_vars):
        """
        Parse ctest output to extract overhead percentage.
        Expected format from the test output includes lines like:
        "618: Overhead (%): med: 60.91 (PRIMARY) avg: 60.88 min: 60.81 max: 67.82"
        We extract the median value (60.91 in this example).
        """
        results = []

        # Look for overhead percentage with median value
        # Pattern matches "Overhead (%): med: 60.91 (PRIMARY)"
        overhead_pattern = r"Overhead\s*\(%\)\s*:\s*med:\s*([0-9.]+)\s*\(PRIMARY\)"

        for line in output.splitlines():
            match = re.search(overhead_pattern, line, re.IGNORECASE)
            if match:
                overhead_value = float(match.group(1))
                results.append(
                    Result(
                        label=self.name(),
                        value=overhead_value,
                        command=command,
                        env=env_vars,
                        unit="%",
                        git_url=self.suite.git_url(),
                        git_hash=self.suite.git_hash(),
                    )
                )
                log.info(f"Parsed overhead median: {overhead_value}%")
                break  # Only take the first match

        if not results:
            # Fallback: look for simpler overhead pattern
            fallback_pattern = r"(?:Overhead|overhead).*:\s*([0-9.]+)\s*%"
            for line in output.splitlines():
                match = re.search(fallback_pattern, line, re.IGNORECASE)
                if match:
                    overhead_value = float(match.group(1))
                    log.warning(
                        f"Used fallback pattern to extract overhead: {overhead_value}%"
                    )
                    results.append(
                        Result(
                            label=self.name(),
                            value=overhead_value,
                            command=command,
                            env=env_vars,
                            unit="%",
                            git_url=self.suite.git_url(),
                            git_hash=self.suite.git_hash(),
                        )
                    )
                    break

        if not results:
            raise ValueError(
                "Could not parse overhead information from test output. "
                "Expected format: 'Overhead (%): med: XX.XX (PRIMARY)'"
            )

        return results
