# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json


def generate_html(benchmark_runs: list, compare_names: list[str]):

    # Get unique suite names
    suite_names = {result.suite for run in benchmark_runs for result in run.results}

    # Write data to js file
    # We can't store this as a standalone json file because it needs to be inline in the html
    with open("benchmark_data.js", "w") as f:
        f.write("const benchmarkRuns = [\n")
        for i, run in enumerate(benchmark_runs):
            if i > 0:
                f.write(",\n")
            f.write(run.to_json())

        f.write("\n];\n\nconst defaultCompareNames = ")
        json.dump(compare_names, f)
        f.write(";\n\nconst suiteNames = ")
        json.dump(list(suite_names), f)
        f.write(";")
