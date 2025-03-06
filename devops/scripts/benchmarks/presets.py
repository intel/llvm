# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Type

class Preset:
    def description(self) -> str:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def suites(self) -> List[str]:
        raise NotImplementedError

class Full(Preset):
    def description(self) -> str:
        return "All available benchmarks."

    def suites(self) -> List[str]:
        return [
            "Compute Benchmarks",
            "llama.cpp bench",
            "SYCL-Bench",
            "Velocity Bench",
            "UMF",
        ]

class SYCL(Preset):
    def description(self) -> str:
        return "All available benchmarks related to SYCL."

    def suites(self) -> List[str]:
        return ["Compute Benchmarks", "llama.cpp bench", "SYCL-Bench", "Velocity Bench"]

class Minimal(Preset):
    def description(self) -> str:
        return "Short microbenchmarks."

    def suites(self) -> List[str]:
        return ["Compute Benchmarks"]

class Normal(Preset):
    def description(self) -> str:
        return "Comprehensive mix of microbenchmarks and real applications."

    def suites(self) -> List[str]:
        return ["Compute Benchmarks", "llama.cpp bench", "Velocity Bench"]

class Test(Preset):
    def description(self) -> str:
        return "Noop benchmarks for framework testing."

    def suites(self) -> List[str]:
        return ["Test Suite"]

presets = [Full(), SYCL(), Minimal(), Normal(), Test()]

def preset_get_by_name(name: str) -> Preset:
    for p in presets:
        if p.name().upper() == name.upper():
            return p
    raise ValueError(f"Preset '{name}' not found.")
