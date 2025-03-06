# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

class Preset():
    def description(self):
        pass
    def suites(self) -> list[str]:
        return []

class Full(Preset):
    def description(self):
        return "All available benchmarks."
    def suites(self) -> list[str]:
        return ['Compute Benchmarks', 'llama.cpp bench', 'SYCL-Bench', 'Velocity Bench', 'UMF']

class SYCL(Preset):
    def description(self):
        return "All available benchmarks related to SYCL."
    def suites(self) -> list[str]:
        return ['Compute Benchmarks', 'llama.cpp bench', 'SYCL-Bench', 'Velocity Bench']

class Minimal(Preset):
    def description(self):
        return "Short microbenchmarks."
    def suites(self) -> list[str]:
        return ['Compute Benchmarks']

class Normal(Preset):
    def description(self):
        return "Comprehensive mix of microbenchmarks and real applications."
    def suites(self) -> list[str]:
        return ['Compute Benchmarks']

class Test(Preset):
    def description(self):
        return "Noop benchmarks for framework testing."
    def suites(self) -> list[str]:
        return ['Test Suite']


class Presets(Enum):
    FULL = Full
    SYCL = SYCL # Nightly
    NORMAL = Normal # PR
    MINIMAL = Minimal # Quick smoke tests
    TEST = Test
