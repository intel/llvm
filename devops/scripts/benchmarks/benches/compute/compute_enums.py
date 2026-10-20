# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum


class RUNTIMES(Enum):
    SYCL_PREVIEW = "syclpreview"
    SYCL = "sycl"
    LEVEL_ZERO = "l0"
    UR = "ur"
    OFFLOAD = "ol"


COMPUTE_BENCHMARK_RUNTIMES = [
    RUNTIMES.SYCL_PREVIEW,
    RUNTIMES.SYCL,
    RUNTIMES.LEVEL_ZERO,
    RUNTIMES.UR,
]

DEFAULT_BENCHMARK_RUNTIMES = [
    RUNTIMES.SYCL,
    RUNTIMES.LEVEL_ZERO,
    RUNTIMES.UR,
]

SUBMIT_KERNEL_RUNTIMES = [
    RUNTIMES.SYCL_PREVIEW,
    RUNTIMES.SYCL,
    RUNTIMES.LEVEL_ZERO,
    RUNTIMES.UR,
    RUNTIMES.OFFLOAD,
]

TORCH_BENCHMARK_RUNTIMES = [
    RUNTIMES.SYCL_PREVIEW,
    RUNTIMES.SYCL,
    RUNTIMES.LEVEL_ZERO,
]

SYCL_RUNTIMES = [
    RUNTIMES.SYCL_PREVIEW,
    RUNTIMES.SYCL,
]

SYCL_AND_LEVEL_ZERO_RUNTIMES = [
    RUNTIMES.SYCL,
    RUNTIMES.LEVEL_ZERO,
]

NATIVE_GRAPH_RUNTIMES = [
    RUNTIMES.LEVEL_ZERO,
    RUNTIMES.UR,
]

CUDA_COMPATIBLE_RUNTIMES = [
    RUNTIMES.SYCL_PREVIEW,
    RUNTIMES.SYCL,
    RUNTIMES.UR,
    RUNTIMES.OFFLOAD,
]


class PROFILERS(Enum):
    TIMER = "timer"
    CPU_COUNTER = "cpuCounter"


class KERNEL_NAME(Enum):
    ADD = "Add"
    ADD_SEQUENCE = "AddSequence"
    EMPTY = "Empty"


def runtime_to_tag_name(runtime: RUNTIMES) -> str:
    return {
        RUNTIMES.SYCL_PREVIEW: "SYCL",
        RUNTIMES.SYCL: "SYCL",
        RUNTIMES.LEVEL_ZERO: "L0",
        RUNTIMES.UR: "UR",
        RUNTIMES.OFFLOAD: "OFFLOAD",
    }[runtime]
