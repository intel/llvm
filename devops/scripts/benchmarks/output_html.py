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

    if options.output_html == "local":
        data_path = os.path.join(html_path, "data.js")
        # Write data to js file
        # We can't store this as a standalone json file because it needs to be inline in the html
        with open(data_path, "w") as f:
            f.write("benchmarkRuns = [\n")
            # it might be tempting to just to create a list and convert
            # that to a json, but that leads to json being serialized twice.
            for i, run in enumerate(benchmark_runs):
                if i > 0:
                    f.write(",\n")
                f.write(run.to_json())

            f.write("\n];\n\n")  # terminates benchmarkRuns

            f.write("benchmarkMetadata = ")
            json.dump(serializable_metadata, f)

            f.write(";\n\n")  # terminates benchmarkMetadata

            f.write("defaultCompareNames = ")
            json.dump(compare_names, f)
            f.write(";\n")  # terminates defaultCompareNames

        print(f"See {os.getcwd()}/html/index.html for the results.")
    else:
        data_path = os.path.join(html_path, "data.json")
        with open(data_path, "w") as f:
            json_data = {"runs": benchmark_runs, "metadata": serializable_metadata}
            json.dump(json_data, f, indent=2)

        print(
            f"Upload {data_path} to a location set in config.js remoteDataUrl argument."
        )
