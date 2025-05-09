# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random
from utils.utils import git_clone
from .base import Benchmark, Suite
from utils.result import BenchmarkMetadata, Result
from utils.utils import run, create_build_path
from options import options
import os


class TestSuite(Suite):
    def __init__(self):
        return

    def setup(self):
        return

    def name(self) -> str:
        return "Test Suite"

    def benchmarks(self) -> list[Benchmark]:
        bench_configs = [
            ("Memory Bandwidth", 2000, 200, "Foo Group", None, None),
            ("Latency", 100, 20, "Bar Group", "A Latency test note!", None),
            ("Throughput", 1500, 150, "Foo Group", None, None),
            ("FLOPS", 3000, 300, "Foo Group", None, "Unstable FLOPS test!"),
            ("Cache Miss Rate", 250, 25, "Bar Group", "Test Note", "And another note!"),
        ]

        result = []
        for base_name, base_value, base_diff, group, notes, unstable in bench_configs:
            for variant in range(6):
                value_multiplier = 1.0 + (variant * 0.2)
                name = f"{base_name} {variant+1}"
                value = base_value * value_multiplier
                diff = base_diff * value_multiplier

                result.append(
                    TestBench(self, name, value, diff, group, notes, unstable)
                )

        return result

    def additional_metadata(self) -> dict[str, BenchmarkMetadata]:
        return {
            "Foo Group": BenchmarkMetadata(
                type="group",
                description="This is a test benchmark for Foo Group.",
                notes="This is a test note for Foo Group.\n" "Look, multiple lines!",
            ),
            "Bar Group": BenchmarkMetadata(
                type="group",
                description="This is a test benchmark for Bar Group.",
                unstable="This is an unstable note for Bar Group.",
            ),
        }


class TestBench(Benchmark):
    def __init__(self, suite, name, value, diff, group="", notes=None, unstable=None):
        super().__init__("", suite)
        self.bname = name
        self.value = value
        self.diff = diff
        self.group = group
        self.notes_text = notes
        self.unstable_text = unstable

    def name(self):
        return self.bname

    def lower_is_better(self):
        return True

    def setup(self):
        return

    def description(self) -> str:
        return f"This is a test benchmark for {self.bname}."

    def notes(self) -> str:
        return self.notes_text

    def unstable(self) -> str:
        return self.unstable_text

    def run(self, env_vars) -> list[Result]:
        random_value = self.value + random.uniform(-1 * (self.diff), self.diff)
        return [
            Result(
                label=self.name(),
                explicit_group=self.group,
                value=random_value,
                command=["test", "--arg1", "foo"],
                env={"A": "B"},
                stdout="no output",
                unit="ms",
            )
        ]

    def teardown(self):
        return
