# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from options import options
from utils.result import BenchmarkMetadata


def generate_html(
    benchmark_runs: list,
    compare_names: list[str],
    html_path: str,
    metadata: dict[str, BenchmarkMetadata],
):
    benchmark_runs.sort(key=lambda run: run.date, reverse=True)
    serializable_metadata = {k: v.__dict__ for k, v in metadata.items()}

    serializable_runs = [json.loads(run.to_json()) for run in benchmark_runs]

    data = {
        "runs": serializable_runs,
        "metadata": serializable_metadata,
        "defaultCompareNames": compare_names,
    }

    if options.output_html == "local":
        data_path = os.path.join(html_path, "data.js")
        with open(data_path, "w") as f:
            # For local format, we need to write JavaScript variable assignments
            f.write("benchmarkRuns = ")
            json.dump(data["runs"], f, indent=2)
            f.write(";\n\n")

            f.write("benchmarkMetadata = ")
            json.dump(data["metadata"], f, indent=2)
            f.write(";\n\n")

            f.write("defaultCompareNames = ")
            json.dump(data["defaultCompareNames"], f, indent=2)
            f.write(";\n")

        print(f"See {os.getcwd()}/html/index.html for the results.")
    else:
        # For remote format, we write a single JSON file
        data_path = os.path.join(html_path, "data.json")
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"Upload {data_path} to a location set in config.js remoteDataUrl argument."
        )
