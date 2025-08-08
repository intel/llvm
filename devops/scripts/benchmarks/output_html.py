# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import glob

from options import options
from utils.result import BenchmarkMetadata, BenchmarkOutput
from utils.logger import log
from history import BenchmarkHistory
from benches.base import benchmark_tags_dict


def _write_output_to_file(
    output: BenchmarkOutput, html_path: str, archive: bool = False
) -> None:
    """
    Helper function to write the BenchmarkOutput to a file in JSON format.
    """
    # Define variable configuration based on whether we're archiving or not
    filename = "data_archive" if archive else "data"

    if options.output_html == "local":
        data_path = os.path.join(html_path, f"{filename}.js")

        with open(data_path, "w") as f:
            # For local format, we need to write JavaScript variable assignments
            f.write("benchmarkRuns = ")
            json.dump(json.loads(output.to_json())["runs"], f, indent=2)
            f.write(";\n\n")

            f.write("defaultCompareNames = ")
            json.dump(output.default_compare_names, f, indent=2)
            f.write(";\n\n")

            f.write("benchmarkMetadata = ")
            json.dump(json.loads(output.to_json())["metadata"], f, indent=2)
            f.write(";\n\n")

            f.write("benchmarkTags = ")
            json.dump(json.loads(output.to_json())["tags"], f, indent=2)
            f.write(";\n")

            if not archive:
                log.info(f"See {html_path}/index.html for the results.")
    else:
        # For remote format, we write a single JSON file
        data_path = os.path.join(html_path, f"{filename}.json")
        with open(data_path, "w") as f:
            json.dump(json.loads(output.to_json()), f, indent=2)
        log.info(
            f"Upload {data_path} to a location set in config.js remoteDataUrl argument."
        )


def generate_html(
    history: BenchmarkHistory,
    compare_names: list[str],
    html_path: str,
    metadata: dict[str, BenchmarkMetadata],
):
    """Generate HTML output for benchmark results."""
    current_runs, archived_runs = history.partition_runs_by_age()

    # Sorted in reverse, such that runs are ordered from newest to oldest
    current_runs.sort(key=lambda run: run.date, reverse=True)

    # Create the comprehensive output object
    output = BenchmarkOutput(
        runs=current_runs,
        metadata=metadata,
        tags=benchmark_tags_dict,
        default_compare_names=compare_names,
    )
    _write_output_to_file(output, html_path)

    # Generate a separate file for archived runs if any
    if archived_runs:
        archived_runs.sort(key=lambda run: run.date, reverse=True)
        archived_output = BenchmarkOutput(
            runs=archived_runs,
            metadata=metadata,
            tags=benchmark_tags_dict,
            default_compare_names=compare_names,
        )
        _write_output_to_file(archived_output, html_path, archive=True)
