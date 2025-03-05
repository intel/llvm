# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os


def generate_html(benchmark_runs: list, compare_names: list[str]):

    # Get unique suite names
    suite_names = {result.suite for run in benchmark_runs for result in run.results}

    # create path to data.js in html folder
    data_path = os.path.join(os.path.dirname(__file__), "html", "data.js")

    # Write data to js file
    # We can't store this as a standalone json file because it needs to be inline in the html
    with open(data_path, "w") as f:
        f.write("const benchmarkRuns = [\n")
        # it might be tempting to just to create a list and convert
        # that to a json, but that leads to json being serialized twice.
        for i, run in enumerate(benchmark_runs):
            if i > 0:
                f.write(",\n")
            f.write(run.to_json())

        f.write("\n];\n\n")  # terminates benchmarkRuns

        # these are not const because they might be modified
        # in config.js
        f.write("defaultCompareNames = ")
        json.dump(compare_names, f)
        f.write(";\n\n")  # terminates defaultCompareNames
        f.write("suiteNames = ")
        json.dump(list(suite_names), f)
        f.write(";")  # terminates suiteNames
