# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from options import options


def generate_html(benchmark_runs: list, compare_names: list[str]):
    # create path to data.js in html folder
    html_path = os.path.join(os.path.dirname(__file__), "html")

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

            f.write("defaultCompareNames = ")
            json.dump(compare_names, f)
            f.write(";\n")  # terminates defaultCompareNames

        print(f"See {os.getcwd()}/html/index.html for the results.")
    else:
        data_path = os.path.join(html_path, "data.json")
        with open(data_path, "w") as f:
            f.write("[\n")
            for i, run in enumerate(benchmark_runs):
                if i > 0:
                    f.write(",\n")
                f.write(run.to_json())
            f.write("\n]\n")

        print(
            f"Upload {data_path} to a location set in config.js remoteDataUrl argument."
        )
