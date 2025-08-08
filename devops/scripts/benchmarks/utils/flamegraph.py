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

    def handle_output(self, bench_name: str, perf_data_file: str, suite_name: str = ""):
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

            # Generate perf script output
            perf_result = run(["perf", "script", "-i", perf_data_file])

            # Pipe through stackcollapse-perf.pl
            stackcollapse_perf_path = os.path.join(
                self.repo_dir, "stackcollapse-perf.pl"
            )
            with open(folded_file, "w") as f_folded:
                stackcollapse_proc = subprocess.Popen(
                    [stackcollapse_perf_path],
                    stdin=subprocess.PIPE,
                    stdout=f_folded,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                stackcollapse_proc.communicate(input=perf_result.stdout.decode())

            # Step 2: Generate flamegraph SVG
            log.debug(f"Generating flamegraph SVG: {svg_file}")
            flamegraph_pl_path = os.path.join(self.repo_dir, "flamegraph.pl")

            # Generate SVG
            flamegraph_cmd = [
                flamegraph_pl_path,
                "--title",
                f"{options.save_name} - {bench_name}",
                "--width",
                str(self.FLAMEGRAPH_WIDTH),
                folded_file,
            ]

            result = run(flamegraph_cmd)
            with open(svg_file, "w") as f_svg:
                f_svg.write(result.stdout.decode())

            # Clean up intermediate files
            if os.path.exists(folded_file):
                os.remove(folded_file)

            if not os.path.exists(svg_file) or os.path.getsize(svg_file) == 0:
                raise RuntimeError(f"Failed to generate flamegraph SVG: {svg_file}")

            log.debug(f"Generated flamegraph: {svg_file}")

            # Create symlink immediately after SVG generation
            self._create_immediate_symlink(svg_file, suite_name)

            # Prune old flamegraph directories
            self._prune_flamegraph_dirs(os.path.dirname(perf_data_file))

            return svg_file

        except Exception as e:
            # Clean up on failure
            for temp_file in [folded_file, svg_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise RuntimeError(f"Failed to generate flamegraph for {bench_name}: {e}")

    def _create_immediate_symlink(self, svg_file: str, suite_name: str = ""):
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
            self._update_flamegraph_manifest(
                html_path, rel_path, options.save_name, suite_name
            )

        except Exception as e:
            log.debug(f"Failed to create immediate symlink for {svg_file}: {e}")

    def _update_flamegraph_manifest(
        self, html_path: str, svg_rel_path: str, run_name: str, suite_name: str = ""
    ):
        """
        Store flamegraph information for later batch update.
        All flamegraphs for a run will be written together at the end.
        """
        try:
            from datetime import datetime

            # Extract benchmark name from the relative path
            # Format: benchmark_name/timestamp_runname.svg
            path_parts = svg_rel_path.split("/")
            if len(path_parts) >= 1:
                benchmark_name = path_parts[0]

                # Store the flamegraph info for this run (will be written in batch later)
                if not hasattr(self, "_flamegraph_data"):
                    self._flamegraph_data = {}

                if run_name not in self._flamegraph_data:
                    self._flamegraph_data[run_name] = {
                        "benchmarks": {},
                        "suites": {},
                        "timestamp": self.timestamp,
                    }

                # Store latest flamegraph for this benchmark (overwrites if same benchmark runs again)
                self._flamegraph_data[run_name]["benchmarks"][
                    benchmark_name
                ] = svg_rel_path

                # Store suite information for this benchmark
                if suite_name:
                    self._flamegraph_data[run_name]["suites"][
                        benchmark_name
                    ] = suite_name

                self._flamegraph_data[run_name][
                    "timestamp"
                ] = self.timestamp  # Update to latest

                log.debug(
                    f"Stored flamegraph data for {run_name}: {benchmark_name} (suite: {suite_name})"
                )

        except Exception as e:
            log.debug(f"Failed to store flamegraph data: {e}")

    def finalize_run_flamegraphs(self, html_path: str, run_name: str):
        """
        Write all flamegraphs for a completed run to the flamegraphs.js file.
        This should be called at the end of a benchmark run.
        Accumulates multiple runs - new runs are added to existing data.
        """
        try:
            from datetime import datetime
            import json
            import re

            if (
                not hasattr(self, "_flamegraph_data")
                or run_name not in self._flamegraph_data
            ):
                log.debug(f"No flamegraph data to finalize for run {run_name}")
                return

            flamegraphs_js_file = os.path.join(html_path, "flamegraphs.js")

            # Read existing data if file exists
            existing_runs = {}
            if os.path.exists(flamegraphs_js_file):
                try:
                    with open(flamegraphs_js_file, "r") as f:
                        content = f.read()

                    # Parse existing runs data using regex
                    # Look for the runs object structure - fix regex to handle nested braces
                    runs_match = re.search(
                        r"runs:\s*\{(.*)\},\s*last_updated:", content, re.DOTALL
                    )
                    if runs_match:
                        runs_content = runs_match.group(1)
                        # Simple parsing for existing run entries
                        # Extract run names that already exist
                        run_entries = re.findall(r'"([^"]+)":\s*\{', runs_content)
                        for existing_run in run_entries:
                            if (
                                existing_run != run_name
                            ):  # Don't include current run (will be overwritten)
                                # Extract benchmark list, timestamp, and suites for this run
                                run_match = re.search(
                                    rf'"{re.escape(existing_run)}":\s*\{{\s*benchmarks:\s*(\[[^\]]*\]),\s*timestamp:\s*"([^"]*)"(?:,\s*suites:\s*(\{{[^}}]*\}}))?',
                                    runs_content,
                                )
                                if run_match:
                                    benchmarks_str, timestamp, suites_str = (
                                        run_match.groups()
                                    )
                                    try:
                                        benchmarks = json.loads(benchmarks_str)
                                        existing_run_data = {
                                            "benchmarks": benchmarks,
                                            "timestamp": timestamp,
                                        }
                                        # Add suites if available
                                        if suites_str:
                                            suites = json.loads(suites_str)
                                            existing_run_data["suites"] = suites

                                        existing_runs[existing_run] = existing_run_data
                                    except json.JSONDecodeError:
                                        log.debug(
                                            f"Could not parse data for run {existing_run}"
                                        )

                        log.debug(
                            f"Found {len(existing_runs)} existing runs in flamegraphs.js"
                        )

                except Exception as e:
                    log.debug(f"Could not read existing flamegraphs.js: {e}")

            # Prepare data for this run
            run_data = self._flamegraph_data[run_name]
            benchmark_list = list(run_data["benchmarks"].keys())
            suites_dict = run_data.get("suites", {})

            # Add current run to existing runs
            all_runs = existing_runs.copy()
            all_runs[run_name] = {
                "benchmarks": benchmark_list,
                "suites": suites_dict,
                "timestamp": run_data["timestamp"],
            }

            # Generate runs object content
            runs_content_parts = []
            for run_name_key, run_info in all_runs.items():
                # Include suites information if available
                suites_content = ""
                if "suites" in run_info and run_info["suites"]:
                    suites_content = (
                        f',\n      suites: {json.dumps(run_info["suites"])}'
                    )

                runs_content_parts.append(
                    f"""    "{run_name_key}": {{
      benchmarks: {json.dumps(run_info['benchmarks'])},
      timestamp: "{run_info['timestamp']}"{suites_content}
    }}"""
                )

            runs_content = ",\n".join(runs_content_parts)

            # Create the complete content with all accumulated runs
            flamegraph_content = f"""// Flamegraph data - latest flamegraph from each benchmark per run
// This file is automatically generated by the flamegraph system
flamegraphData = {{
  runs: {{
{runs_content}
  }},
  last_updated: "{datetime.now().isoformat()}"
}};
"""

            # Write the flamegraph data
            with open(flamegraphs_js_file, "w") as f:
                f.write(flamegraph_content)

            log.info(
                f"Finalized flamegraphs.js with {len(benchmark_list)} benchmarks for run {run_name} (total runs: {len(all_runs)})"
            )

            # Clean up stored data for this run
            if hasattr(self, "_flamegraph_data"):
                del self._flamegraph_data[run_name]

        except Exception as e:
            log.debug(f"Failed to finalize flamegraphs.js: {e}")


# Singleton pattern to ensure only one instance of FlameGraph is created
def get_flamegraph() -> FlameGraph:
    if not hasattr(get_flamegraph, "_instance"):
        get_flamegraph._instance = FlameGraph()
    return get_flamegraph._instance
