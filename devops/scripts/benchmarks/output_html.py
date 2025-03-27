# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from options import options
from utils.result import BenchmarkMetadata, BenchmarkOutput
from benches.base import benchmark_tags, benchmark_tags_dict


def generate_html(
    benchmark_runs: list,
    compare_names: list[str],
    html_path: str,
    metadata: dict[str, BenchmarkMetadata],
):
    benchmark_runs.sort(key=lambda run: run.date, reverse=True)
    # Sorted in reverse, such that runs are ordered from newest to oldest

    # Create the comprehensive output object
    output = BenchmarkOutput(
        runs=benchmark_runs,
        metadata=metadata,
        tags=benchmark_tags_dict,
        default_compare_names=compare_names,
    )

    if options.output_html == "local":
        data_path = os.path.join(html_path, "data.js")
        with open(data_path, "w") as f:
            # For local format, we need to write JavaScript variable assignments
            f.write("benchmarkRuns = ")
            json.dump(json.loads(output.to_json())["runs"], f, indent=2)
            f.write(";\n\n")

            f.write("benchmarkMetadata = ")
            json.dump(json.loads(output.to_json())["metadata"], f, indent=2)
            f.write(";\n\n")

            f.write("benchmarkTags = ")
            json.dump(json.loads(output.to_json())["tags"], f, indent=2)
            f.write(";\n\n")

            f.write("defaultCompareNames = ")
            json.dump(output.default_compare_names, f, indent=2)
            f.write(";\n")

        print(f"See {os.getcwd()}/html/index.html for the results.")
    else:
        # For remote format, we write a single JSON file
        data_path = os.path.join(html_path, "data.json")
        with open(data_path, "w") as f:
            json.dump(json.loads(output.to_json()), f, indent=2)

        print(
            f"Upload {data_path} to a location set in config.js remoteDataUrl argument."
        )
