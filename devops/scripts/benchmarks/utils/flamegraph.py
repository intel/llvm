# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
import signal
import time

from options import options
from utils.utils import run, git_clone
from utils.logger import log

from datetime import datetime, timezone


class FlameGraph:
    """FlameGraph wrapper for managing FlameGraph tool execution and results."""

    # FlameGraph SVG width to fit within web interface container
    # Optimized width to avoid horizontal scrollbars within 1100px container
    FLAMEGRAPH_WIDTH = 1000

    def __init__(self):
        self.timestamp = (
            datetime.now(tz=timezone.utc).strftime(options.TIMESTAMP_FORMAT)
            if options.timestamp_override is None
            else options.timestamp_override
        )

        log.info("Downloading FlameGraph...")
        repo_dir = git_clone(
            options.workdir,
            "flamegraph-repo",
            "https://github.com/brendangregg/FlameGraph.git",
            "master",
        )

        # FlameGraph doesn't need building, just verify scripts exist and are executable
        flamegraph_scripts = [
            "flamegraph.pl",
            "stackcollapse-perf.pl",
            "stackcollapse.pl",
        ]

        for script in flamegraph_scripts:
            script_path = os.path.join(repo_dir, script)
            # First check if script exists
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"FlameGraph script not found: {script_path}")
            # Then check if it's executable
            if not os.access(script_path, os.X_OK):
                raise RuntimeError(f"FlameGraph script not executable: {script_path}")
            log.debug(f"Verified {script} exists and is executable")

        # Store repo_dir for later use when generating flamegraphs
        self.repo_dir = repo_dir

        log.info("FlameGraph tools ready.")

        if options.results_directory_override == None:
            self.flamegraphs_dir = os.path.join(
                options.workdir, "results", "flamegraphs"
            )
        else:
            self.flamegraphs_dir = os.path.join(
                options.results_directory_override, "flamegraphs"
            )

    def _prune_flamegraph_dirs(self, res_dir: str, FILECNT: int = 10):
        """Keep only the last FILECNT files in the flamegraphs directory."""
        files = os.listdir(res_dir)
        files.sort()  # Lexicographical sort matches timestamp order
        if len(files) > 2 * FILECNT:
            for f in files[: len(files) - 2 * FILECNT]:
                full_path = os.path.join(res_dir, f)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                    log.debug(f"Removing old flamegraph file: {full_path}")

    def cleanup(self, bench_cwd: str, perf_data_file: str):
        """
        Remove incomplete output files in case of failure.
        """
        flamegraph_dir = os.path.dirname(perf_data_file)
        flamegraph_base = os.path.basename(perf_data_file)
        for f in os.listdir(flamegraph_dir):
            if f.startswith(flamegraph_base + "."):
                os.remove(os.path.join(flamegraph_dir, f))
                log.debug(f"Cleanup: Removed {f} from {flamegraph_dir}")

    def setup(
        self, bench_name: str, command: list[str], extra_perf_opt: list[str] = None
    ):
        """
        Prepare perf data file name and full command for the benchmark run.
        Returns a tuple of (perf_data_file, perf_command).
        """
        # Check if perf is available
        if not shutil.which("perf"):
            raise FileNotFoundError(
                "perf command not found. Please install linux-tools or perf package."
            )

        os.makedirs(self.flamegraphs_dir, exist_ok=True)
        bench_dir = os.path.join(f"{self.flamegraphs_dir}", f"{bench_name}")

        os.makedirs(bench_dir, exist_ok=True)

        perf_data_file = os.path.join(
            bench_dir, f"{self.timestamp}_{options.save_name}.perf.data"
        )

        if extra_perf_opt is None:
            extra_perf_opt = []

        # Default perf record options for flamegraph generation
        perf_command = (
            [
                "perf",
                "record",
                "-g",  # Enable call-graph recording
                "-F",
                "99",  # Sample frequency
                "--call-graph",
                "dwarf",  # Use DWARF unwinding for better stack traces
                "-o",
                perf_data_file,
            ]
            + extra_perf_opt
            + ["--"]
            + command
        )
        log.debug(f"Perf cmd: {' '.join(perf_command)}")

        return perf_data_file, perf_command

    def handle_output(self, bench_name: str, perf_data_file: str):
        """
        Generate SVG flamegraph from perf data file.
        Returns the path to the generated SVG file.
        """
        if not os.path.exists(perf_data_file) or os.path.getsize(perf_data_file) == 0:
            raise FileNotFoundError(
                f"Perf data file not found or empty: {perf_data_file}"
            )

        # Generate output SVG filename following same pattern as perf data
        svg_file = perf_data_file.replace(".perf.data", ".svg")
        folded_file = perf_data_file.replace(".perf.data", ".folded")

        try:
            # Step 1: Convert perf script to folded format
            log.debug(f"Converting perf data to folded format: {folded_file}")
            with open(folded_file, "w") as f_folded:
                # Run perf script to get the stack traces
                perf_script_proc = subprocess.Popen(
                    ["perf", "script", "-i", perf_data_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

                # Pipe through stackcollapse-perf.pl
                stackcollapse_perf_path = os.path.join(
                    self.repo_dir, "stackcollapse-perf.pl"
                )
                stackcollapse_proc = subprocess.Popen(
                    [stackcollapse_perf_path],
                    stdin=perf_script_proc.stdout,
                    stdout=f_folded,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

                perf_script_proc.stdout.close()
                stackcollapse_proc.wait()
                perf_script_proc.wait()

            # Step 2: Generate flamegraph SVG
            log.debug(f"Generating flamegraph SVG: {svg_file}")
            flamegraph_pl_path = os.path.join(self.repo_dir, "flamegraph.pl")
            with open(folded_file, "r") as f_folded, open(svg_file, "w") as f_svg:
                flamegraph_proc = subprocess.Popen(
                    [
                        flamegraph_pl_path,
                        "--title",
                        f"{options.save_name} - {bench_name}",
                        "--width",
                        str(
                            self.FLAMEGRAPH_WIDTH
                        ),  # Fit within container without scrollbars
                    ],
                    stdin=f_folded,
                    stdout=f_svg,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                flamegraph_proc.wait()

            # Clean up intermediate files
            if os.path.exists(folded_file):
                os.remove(folded_file)

            if not os.path.exists(svg_file) or os.path.getsize(svg_file) == 0:
                raise RuntimeError(f"Failed to generate flamegraph SVG: {svg_file}")

            log.debug(f"Generated flamegraph: {svg_file}")

            # Create symlink immediately after SVG generation
            self._create_immediate_symlink(svg_file)

            # Prune old flamegraph directories
            self._prune_flamegraph_dirs(os.path.dirname(perf_data_file))

            return svg_file

        except Exception as e:
            # Clean up on failure
            for temp_file in [folded_file, svg_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise RuntimeError(f"Failed to generate flamegraph for {bench_name}: {e}")

    def _create_immediate_symlink(self, svg_file: str):
        """
        Create a symbolic link for the SVG file immediately after generation.
        This ensures the web interface can access the file right away.
        """
        try:
            # Check if workdir is available
            if not options.workdir:
                log.error("No workdir available for immediate symlink creation")
                return

            # Use the default HTML path relative to the script location
            script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from utils/
            html_path = os.path.join(script_dir, "html")

            if not os.path.exists(html_path):
                log.error(f"HTML directory not found: {html_path}")
                return

            # Calculate the relative path of the SVG file from the flamegraphs directory
            if not svg_file.startswith(self.flamegraphs_dir):
                log.error(f"SVG file not in expected flamegraphs directory: {svg_file}")
                return

            rel_path = os.path.relpath(svg_file, self.flamegraphs_dir)
            target_dir = os.path.join(html_path, "results", "flamegraphs")
            target_file = os.path.join(target_dir, rel_path)

            # Create target directory structure
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            # Remove existing symlink if it exists
            if os.path.islink(target_file):
                os.unlink(target_file)
            elif os.path.exists(target_file):
                os.remove(target_file)

            # Create the symlink
            os.symlink(svg_file, target_file)
            log.debug(f"Created immediate symlink: {target_file} -> {svg_file}")

            # Update the flamegraph manifest for the web interface
            self._update_flamegraph_manifest(html_path, rel_path, options.save_name)

        except Exception as e:
            log.debug(f"Failed to create immediate symlink for {svg_file}: {e}")

    def _update_flamegraph_manifest(
        self, html_path: str, svg_rel_path: str, run_name: str
    ):
        """
        Update the data.js file with flamegraph information by dynamically adding the flamegraphData variable.
        This works with a clean data.js from the repo and adds the necessary structure during execution.
        """
        try:
            import re
            from datetime import datetime

            # Extract benchmark name from the relative path
            # Format: benchmark_name/timestamp_runname.svg
            path_parts = svg_rel_path.split("/")
            if len(path_parts) >= 1:
                benchmark_name = path_parts[0]

                data_js_file = os.path.join(html_path, "data.js")

                # Read the current data.js file
                if not os.path.exists(data_js_file):
                    log.error(
                        f"data.js not found at {data_js_file}, cannot update flamegraph manifest"
                    )
                    return

                with open(data_js_file, "r") as f:
                    content = f.read()

                # Check if flamegraphData already exists
                if "flamegraphData" not in content:
                    # Add flamegraphData object at the end of the file
                    flamegraph_data = f"""

flamegraphData = {{
  runs: {{}},
  last_updated: '{datetime.now().isoformat()}'
}};"""
                    content = content.rstrip() + flamegraph_data
                    log.debug("Added flamegraphData object to data.js")

                # Parse and update the flamegraphData runs structure
                flamegraph_start = content.find("flamegraphData = {")
                if flamegraph_start != -1:
                    # Find the runs object within flamegraphData
                    runs_start = content.find("runs: {", flamegraph_start)
                    if runs_start != -1:
                        # Find the matching closing brace for the runs object
                        brace_count = 0
                        runs_content_start = runs_start + 7  # After "runs: {"
                        runs_content_end = runs_content_start

                        for i, char in enumerate(
                            content[runs_content_start:], runs_content_start
                        ):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                if brace_count == 0:
                                    runs_content_end = i
                                    break
                                brace_count -= 1

                        existing_runs_str = content[
                            runs_content_start:runs_content_end
                        ].strip()
                        existing_runs = {}

                        # Parse existing runs if any
                        if existing_runs_str:
                            # Simple parsing of run entries like: "RunName": { benchmarks: [...], timestamp: "..." }
                            run_matches = re.findall(
                                r'"([^"]+)":\s*\{[^}]*benchmarks:\s*\[([^\]]*)\][^}]*timestamp:\s*"([^"]*)"[^}]*\}',
                                existing_runs_str,
                            )

                            for run_match in run_matches:
                                existing_run_name = run_match[0]
                                existing_benchmarks_str = run_match[1]
                                existing_timestamp = run_match[2]

                                # Parse benchmarks array
                                existing_benchmarks = []
                                if existing_benchmarks_str.strip():
                                    benchmark_matches = re.findall(
                                        r'"([^"]*)"', existing_benchmarks_str
                                    )
                                    existing_benchmarks = benchmark_matches

                                existing_runs[existing_run_name] = {
                                    "benchmarks": existing_benchmarks,
                                    "timestamp": existing_timestamp,
                                }

                        # Add or update this run's benchmark
                        if run_name not in existing_runs:
                            existing_runs[run_name] = {
                                "benchmarks": [],
                                "timestamp": self.timestamp,
                            }
                        else:
                            # Update timestamp to latest for this run (in case of multiple benchmarks in same run)
                            if self.timestamp > existing_runs[run_name]["timestamp"]:
                                existing_runs[run_name]["timestamp"] = self.timestamp

                        # Add benchmark if not already present
                        if benchmark_name not in existing_runs[run_name]["benchmarks"]:
                            existing_runs[run_name]["benchmarks"].append(benchmark_name)
                            existing_runs[run_name]["benchmarks"].sort()  # Keep sorted
                            log.debug(
                                f"Added {benchmark_name} to flamegraphData for run {run_name}"
                            )

                        # Create the new runs object string
                        runs_entries = []
                        for rn, data in existing_runs.items():
                            benchmarks_array = (
                                "["
                                + ", ".join(f'"{b}"' for b in data["benchmarks"])
                                + "]"
                            )
                            runs_entries.append(
                                f'    "{rn}": {{\n      benchmarks: {benchmarks_array},\n      timestamp: "{data["timestamp"]}"\n    }}'
                            )

                        runs_object = "{\n" + ",\n".join(runs_entries) + "\n  }"

                        # Replace the runs object by reconstructing the content
                        before_runs = content[: runs_start + 7]  # Up to "runs: {"
                        after_runs_brace = content[
                            runs_content_end:
                        ]  # From the closing } onwards
                        content = (
                            before_runs + runs_object[1:-1] + after_runs_brace
                        )  # Remove outer braces from runs_object

                        # Update the last_updated timestamp
                        timestamp_pattern = r'(last_updated:\s*)["\'][^"\']*["\']'
                        content = re.sub(
                            timestamp_pattern,
                            rf'\g<1>"{datetime.now().isoformat()}"',
                            content,
                        )

                    # Write the updated content back to data.js
                    with open(data_js_file, "w") as f:
                        f.write(content)

                    log.debug(
                        f"Updated data.js with flamegraph data for {benchmark_name} in run {run_name}"
                    )

        except Exception as e:
            log.debug(f"Failed to update data.js with flamegraph data: {e}")


# Singleton pattern to ensure only one instance of FlameGraph is created
def get_flamegraph() -> FlameGraph:
    if not hasattr(get_flamegraph, "_instance"):
        get_flamegraph._instance = FlameGraph()
    return get_flamegraph._instance
