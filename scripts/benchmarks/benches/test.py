# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random
from utils.utils import git_clone
from .base import Benchmark, Suite
from .result import Result
from utils.utils import run, create_build_path
from .options import options
import os

class TestSuite(Suite):
    def __init__(self):
        return

    def setup(self):
        return

    def benchmarks(self) -> list[Benchmark]:
        bench_configs = [
            ("Memory Bandwidth", 2000, 200),
            ("Latency", 100, 20),
            ("Throughput", 1500, 150),
            ("FLOPS", 3000, 300),
            ("Cache Miss Rate", 250, 25),
        ]

        result = []
        for base_name, base_value, base_diff in bench_configs:
            for variant in range(6):
                value_multiplier = 1.0 + (variant * 0.2)
                name = f"{base_name} {variant+1}"
                value = base_value * value_multiplier
                diff = base_diff * value_multiplier

                result.append(TestBench(name, value, diff))

        return result

class TestBench(Benchmark):
    def __init__(self, name, value, diff):
        self.bname = name
        self.value = value
        self.diff = diff
        super().__init__("")

    def name(self):
        return self.bname

    def lower_is_better(self):
        return True

    def setup(self):
        return

    def run(self, env_vars) -> list[Result]:
        random_value = self.value + random.uniform(-1 * (self.diff), self.diff)
        return [
            Result(label=self.name(), value=random_value, command="", env={"A": "B"}, stdout="no output", unit="ms")
        ]

    def teardown(self):
        return
