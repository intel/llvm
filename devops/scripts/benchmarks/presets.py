# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

presets: dict[str, list[str]] = {
    "Full": [
        "Compute Benchmarks",
        "llama.cpp bench",
        "SYCL-Bench",
        "Velocity Bench",
        "UMF",
    ],
    "SYCL": [
        "Compute Benchmarks",
        "llama.cpp bench",
        "SYCL-Bench",
        "Velocity Bench",
    ],
    "Minimal": [
        "Compute Benchmarks",
    ],
    "Normal": [
        "Compute Benchmarks",
        "llama.cpp bench",
        "Velocity Bench",
    ],
    "Test": [
        "Test Suite",
    ],
}


def enabled_suites(preset: str) -> list[str]:
    try:
        return presets[preset]
    except KeyError:
        raise ValueError(f"Preset '{preset}' not found.")
