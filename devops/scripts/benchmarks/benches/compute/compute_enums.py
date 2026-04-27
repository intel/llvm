# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum


class RUNTIMES(Enum):
    SYCL_PREVIEW = "syclpreview"
    SYCL = "sycl"
    LEVEL_ZERO = "l0"
    UR = "ur"


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
    }[runtime]
