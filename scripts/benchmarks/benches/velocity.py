# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from utils.utils import git_clone
from .base import Benchmark
from .result import Result
from utils.utils import run, create_build_path
from .options import options
import os

class VelocityBench:
    def __init__(self, directory):
        self.directory = directory
        self.repo_path = git_clone(self.directory, "velocity-bench-repo", "https://github.com/oneapi-src/Velocity-Bench/", "b22215c16f789100449c34bf4eaa3fb178983d69")

class VelocityBase(Benchmark):
    def __init__(self, name: str, bin_name: str, vb: VelocityBench):
        super().__init__(vb.directory)
        self.vb = vb
        self.bench_name = name
        self.bin_name = bin_name
        self.code_path = os.path.join(self.vb.repo_path, self.bench_name, 'SYCL')

    def download_deps(self):
        return

    def setup(self):
        self.download_deps()
        self.benchmark_bin = os.path.join(self.directory, self.bench_name, self.bin_name)

        build_path = create_build_path(self.directory, self.bench_name)

        configure_command = [
            "cmake",
            f"-B {build_path}",
            f"-S {self.code_path}",
            f"-DCMAKE_BUILD_TYPE=Release"
        ]
        run(configure_command, {'CC': 'clang', 'CXX':'clang++'}, add_sycl=True)
        run(f"cmake --build {build_path} -j", add_sycl=True)

    def bin_args(self) -> list[str]:
        return []

    def extra_env_vars(self) -> dict:
        return {}

    def parse_output(self, stdout: str) -> float:
        raise NotImplementedError()

    def run(self, env_vars) -> list[Result]:
        env_vars.update(self.extra_env_vars())

        command = [
            f"{self.benchmark_bin}",
        ]
        command += self.bin_args()

        result = self.run_bench(command, env_vars)

        return [ Result(label=self.name(), value=self.parse_output(result), command=command, env=env_vars, stdout=result, lower_is_better=self.lower_is_better()) ]

    def teardown(self):
        return
