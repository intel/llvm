# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

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


# Utility scripts to validate a given preset, useful for e.g. CI:


def main():
    parser = argparse.ArgumentParser(description="Benchmark Preset Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser(
        "query", help="Query benchmarks ran by a preset (as defined in presets.py)"
    )
    query_parser.add_argument("preset_to_query", type=str, help="preset name to query")
    query_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable stdout messages: Useful if you want to check if a preset exists within a shell script.",
    )

    args = parser.parse_args()
    if args.command == "query":
        if args.preset_to_query in presets:
            if not args.quiet:
                print(f"Benchmark suites to be ran in {args.preset_to_query}:")
                for suite in presets[args.preset_to_query]:
                    print(suite)
            exit(0)
        else:
            if not args.quiet:
                print(f"Error: No preset named '{args.preset_to_query}'.")
            exit(1)


if __name__ == "__main__":
    main()
