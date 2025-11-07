# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shutil
from pathlib import Path

from options import options
from utils.utils import (
    run,
    prune_old_files,
    remove_by_prefix,
    sanitize_filename,
)
from utils.logger import log
from git_project import GitProject
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
        self.project = GitProject(
            "https://github.com/brendangregg/FlameGraph.git",
            "41fee1f99f9276008b7cd112fca19dc3ea84ac32",
            Path(options.workdir),
            "flamegraph",
        )
        log.info("FlameGraph tools ready.")

        if options.results_directory_override:
            self.flamegraphs_dir = (
                Path(options.results_directory_override) / "results" / "flamegraphs"
            )
        else:
            self.flamegraphs_dir = Path(options.workdir) / "results" / "flamegraphs"

    def cleanup(self, perf_data_file: str):
        """
        Remove incomplete output files in case of failure.
        """
        perf_data_path = Path(perf_data_file)
        remove_by_prefix(str(perf_data_path.parent), perf_data_path.stem)

    def setup(
        self,
        bench_name: str,
        suite_name: str,
        command: list[str],
    ):
        """
        Prepare perf data file name and full command for the benchmark run.
        Returns a tuple of (perf_data_file, perf_command).
        """
        if not shutil.which("perf"):
            raise FileNotFoundError(
                "perf command not found. Please install linux-tools-$(uname -r) or perf package."
            )

        sanitized_suite_name = sanitize_filename(suite_name)
        sanitized_bench_name = sanitize_filename(bench_name)
        dir_name = f"{sanitized_suite_name}__{sanitized_bench_name}"
        bench_dir = self.flamegraphs_dir / dir_name
        bench_dir.mkdir(parents=True, exist_ok=True)

        perf_data_file = bench_dir / f"{self.timestamp}_{options.save_name}.perf.data"

        perf_command = [
            "perf",
            "record",
            "-g",
            "-F",
            "5000",
            "--call-graph",
            "dwarf",
            "-o",
            str(perf_data_file),
        ]
        perf_command.extend(["--"] + command)

        log.debug(f"Perf cmd: {' '.join(perf_command)}")
        return str(perf_data_file), perf_command

    def handle_output(self, bench_name: str, perf_data_file: str, suite_name: str = ""):
        """
        Generate SVG flamegraph from perf data file.
        Returns the path to the generated SVG file.
        """
        perf_data_path = Path(perf_data_file)
        if not perf_data_path.exists() or perf_data_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Perf data file not found or empty: {perf_data_file}"
            )

        # Create SVG filename by replacing .perf.data with .svg
        # e.g., 20250826_125235_OneDNN_V2.perf.data -> 20250826_125235_OneDNN_V2.svg
        svg_file = perf_data_path.with_name(
            perf_data_path.stem.replace(".perf", "") + ".svg"
        )
        folded_file = perf_data_path.with_suffix(".folded")
        try:
            self._convert_perf_to_folded(perf_data_path, folded_file)
            self._generate_svg(folded_file, svg_file, bench_name)

            log.info(f"FlameGraph SVG created: {svg_file.resolve()}")
            self._create_immediate_symlink(svg_file)

            # Clean up the original perf data file after successful SVG generation
            if perf_data_path.exists():
                perf_data_path.unlink()
                log.debug(f"Removed original perf data file: {perf_data_path}")

            prune_old_files(str(perf_data_path.parent))
            return str(svg_file)
        except Exception as e:
            # Clean up on failure
            for temp_file in [folded_file, svg_file]:
                if temp_file.exists():
                    temp_file.unlink()
            raise RuntimeError(f"Failed to generate flamegraph for {bench_name}: {e}")
        finally:
            # Always clean up intermediate folded file
            if folded_file.exists():
                folded_file.unlink()

    def _convert_perf_to_folded(self, perf_data_file: Path, folded_file: Path):
        """Step 1: Convert perf script to folded format using a pipeline."""
        log.debug(f"Converting perf data to folded format: {folded_file}")
        perf_script_cmd = ["perf", "script", "-i", str(perf_data_file)]
        stackcollapse_cmd = [str(self.project.src_dir / "stackcollapse-perf.pl")]

        try:
            # Run perf script and capture its output
            perf_result = run(perf_script_cmd)

            # Pipe the output of perf script into stackcollapse-perf.pl
            stackcollapse_result = run(stackcollapse_cmd, input=perf_result.stdout)

            # Write the final folded output to a file (stdout is bytes)
            folded_file.write_bytes(stackcollapse_result.stdout)

        except Exception as e:
            log.error(f"Failed during perf-to-folded conversion: {e}")
            raise

    def _generate_svg(self, folded_file: Path, svg_file: Path, bench_name: str):
        """Step 2: Generate flamegraph SVG from a folded file."""
        log.debug(f"Generating flamegraph SVG: {svg_file}")
        flamegraph_cmd = [
            str(self.project.src_dir / "flamegraph.pl"),
            "--title",
            f"{options.save_name} - {bench_name}",
            "--width",
            str(self.FLAMEGRAPH_WIDTH),
            str(folded_file),
        ]
        result = run(flamegraph_cmd)
        svg_file.write_text(result.stdout.decode())

        if not svg_file.exists() or svg_file.stat().st_size == 0:
            raise RuntimeError(f"Failed to generate flamegraph SVG: {svg_file}")

    def _create_immediate_symlink(self, svg_file: Path):
        """Create a symbolic link for the SVG file for immediate web access."""
        try:
            script_dir = Path(__file__).resolve().parent.parent
            html_path = script_dir / "html"
            if not html_path.is_dir():
                log.error(f"HTML directory not found: {html_path}")
                return

            rel_path = svg_file.relative_to(self.flamegraphs_dir)
            target_file = html_path / "results" / "flamegraphs" / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing file/symlink atomically
            if target_file.exists() or target_file.is_symlink():
                target_file.unlink()

            target_file.symlink_to(svg_file.resolve())
            log.debug(f"Created immediate symlink: {target_file} -> {svg_file}")
        except Exception as e:
            log.debug(f"Failed to create immediate symlink for {svg_file}: {e}")


# Singleton pattern to ensure only one instance of FlameGraph is created
def get_flamegraph() -> FlameGraph:
    if not hasattr(get_flamegraph, "_instance"):
        get_flamegraph._instance = FlameGraph()  # type: ignore
    return get_flamegraph._instance  # type: ignore
