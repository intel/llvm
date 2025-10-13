# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from datetime import datetime

from options import options
from utils.result import BenchmarkMetadata, BenchmarkOutput
from utils.logger import log
from history import BenchmarkHistory
from benches.base import benchmark_tags_dict


def _get_flamegraph_data(html_path: str) -> dict:
    """
    Reconstruct flamegraph data by scanning the results directory for SVG files.
    This ensures data is available even in flamegraph-exclusive mode.
    """
    log.debug("Reconstructing flamegraph data from SVG files...")
    results_dir = os.path.join(html_path, "results", "flamegraphs")
    runs_data = {}

    if not os.path.exists(results_dir):
        log.debug("No flamegraph results directory found.")
        return {"runs": {}}

    for bench_dir in os.listdir(results_dir):
        bench_path = os.path.join(results_dir, bench_dir)
        if os.path.isdir(bench_path):
            for svg_file in os.listdir(bench_path):
                if not svg_file.endswith(".svg"):
                    continue

                # Filename format: {timestamp}_{run_name}.svg
                # e.g., 20250826_103000_MyRun.svg
                parts = svg_file.replace(".svg", "").split("_")
                if len(parts) < 3:
                    continue

                timestamp = "_".join(parts[:2])
                run_name = "_".join(parts[2:])

                if run_name not in runs_data:
                    runs_data[run_name] = {"suites": {}, "timestamp": timestamp}

                # The directory name is now SUITE__BENCHMARK
                dir_parts = bench_dir.split("__")
                if len(dir_parts) == 2:
                    suite_name, bench_name = dir_parts
                    # Store suite name for the benchmark
                    runs_data[run_name]["suites"][bench_name] = suite_name
                else:
                    # Fallback for old format or benchmarks without suites
                    runs_data[run_name]["suites"][bench_dir] = "default"

    if not runs_data:
        log.debug("No flamegraph SVG files were found to reconstruct data.")

    return {"runs": runs_data, "last_updated": datetime.now().isoformat()}


def _write_output_to_file(
    output: BenchmarkOutput, html_path: str, archive: bool = False
) -> None:
    """
    Helper function to write the BenchmarkOutput to a file in JSON format.
    """
    # Define variable configuration based on whether we're archiving or not
    filename = "data_archive" if archive else "data"
    output_data = json.loads(output.to_json())  # type: ignore

    if options.flamegraph:
        flamegraph_data = _get_flamegraph_data(html_path)
        if flamegraph_data and flamegraph_data.get("runs"):
            output_data["flamegraphData"] = flamegraph_data
            log.debug(
                f"Added flamegraph data for {len(flamegraph_data['runs'])} runs to {filename}.*"
            )

    runs_list = output_data.get("runs", [])

    if options.output_html == "local":
        # Local JS: emit standalone globals (legacy-style) without wrapper object
        data_path = os.path.join(html_path, f"{filename}.js")
        with open(data_path, "w") as f:
            f.write("benchmarkRuns = ")
            json.dump(runs_list, f, indent=2)
            f.write(";\n")
            if "flamegraphData" in output_data:
                f.write("flamegraphData = ")
                json.dump(output_data["flamegraphData"], f, indent=2)
                f.write(";\n")
            else:
                f.write("flamegraphData = { runs: {} };\n")
            f.write("benchmarkMetadata = ")
            json.dump(output_data.get("metadata", {}), f, indent=2)
            f.write(";\n")
            f.write("benchmarkTags = ")
            json.dump(output_data.get("tags", {}), f, indent=2)
            f.write(";\n")
            f.write("defaultCompareNames = ")
            json.dump(output.default_compare_names, f)
            f.write(";\n")
        if not archive:
            log.info(f"See {html_path}/index.html for the results.")
    else:
        # Remote JSON: emit flat schema aligning with local globals
        remote_obj = {
            "benchmarkRuns": runs_list,
            "benchmarkMetadata": output_data.get("metadata", {}),
            "benchmarkTags": output_data.get("tags", {}),
            "flamegraphData": output_data.get("flamegraphData", {"runs": {}}),
            "defaultCompareNames": output.default_compare_names,
        }
        data_path = os.path.join(html_path, f"{filename}.json")
        with open(data_path, "w") as f:
            json.dump(remote_obj, f, indent=2)
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
    current_runs.sort(key=lambda run: run.date or datetime.min, reverse=True)

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
        archived_runs.sort(key=lambda run: run.date or datetime.min, reverse=True)
        archived_output = BenchmarkOutput(
            runs=archived_runs,
            metadata=metadata,
            tags=benchmark_tags_dict,
            default_compare_names=compare_names,
        )
        _write_output_to_file(archived_output, html_path, archive=True)
