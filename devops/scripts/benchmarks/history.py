# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import json
from pathlib import Path
import socket
from utils.result import Result, BenchmarkRun
from options import Compare, options
from datetime import datetime, timezone
from utils.utils import run
from utils.validate import Validate

from utils.detect_versions import DetectVersion


class BenchmarkHistory:
    runs = []

    def __init__(self, dir):
        self.dir = dir

    def load_result(self, file_path: Path) -> BenchmarkRun:
        if file_path.exists():
            with file_path.open("r") as file:
                data = json.load(file)
                return BenchmarkRun.from_json(data)
        else:
            return None

    def load(self, n: int):
        results_dir = Path(self.dir) / "results"
        if not results_dir.exists() or not results_dir.is_dir():
            print(
                f"Warning: {results_dir} is not a valid directory: no historic results loaded."
            )
            return

        # Get all JSON files in the results directory
        benchmark_files = list(results_dir.glob("*.json"))

        # Extract timestamp and sort files by it
        def extract_timestamp(file_path: Path) -> str:
            try:
                # Assumes results are stored as <name>_YYYYMMDD_HHMMSS.json
                ts = file_path.stem[-len("YYYYMMDD_HHMMSS") :]
                return ts if Validate.timestamp(ts) else ""
            except IndexError:
                return ""

        benchmark_files.sort(key=extract_timestamp, reverse=True)

        # Load the first n benchmark files
        benchmark_runs = []
        for file_path in benchmark_files[:n]:
            benchmark_run = self.load_result(file_path)
            if benchmark_run:
                benchmark_runs.append(benchmark_run)

        self.runs = benchmark_runs

    def create_run(self, name: str, results: list[Result]) -> BenchmarkRun:
        def git_info_from_path(path: Path) -> (str, str):
            """
            Derives git repo, commit information from git repo located in path.

            Returns:
                (str, str): git_hash, github_repo
            """
            try:
                result = run("git rev-parse --short HEAD", cwd=path)
                git_hash = result.stdout.decode().strip()

                # Get the GitHub repo URL from git remote
                remote_result = run("git remote get-url origin", cwd=path)
                remote_url = remote_result.stdout.decode().strip()

                # Convert SSH or HTTPS URL to owner/repo format
                if remote_url.startswith("git@github.com:"):
                    # SSH format: git@github.com:owner/repo.git
                    github_repo = remote_url.split("git@github.com:")[1].rstrip(".git")
                elif remote_url.startswith("https://github.com/"):
                    # HTTPS format: https://github.com/owner/repo.git
                    github_repo = remote_url.split("https://github.com/")[1].rstrip(
                        ".git"
                    )
                else:
                    github_repo = None

            except:
                git_hash = "unknown"
                github_repo = None

            return git_hash, github_repo

        if options.git_commit_override is None or options.github_repo_override is None:
            if options.detect_versions.sycl:
                print(f"Auto-detecting sycl version...")
                github_repo, git_hash = DetectVersion.instance().get_dpcpp_git_info()
            else:
                git_hash, github_repo = git_info_from_path(
                    os.path.dirname(os.path.abspath(__file__))
                )
        else:
            git_hash, github_repo = (
                options.git_commit_override,
                options.github_repo_override,
            )

        # Check if RUNNER_NAME environment variable has been declared.
        #
        # Github runners obfusicate hostnames, thus running socket.gethostname()
        # twice produces two different hostnames. Since github runners always
        # define a RUNNER_NAME variable, use RUNNER_NAME instead if it exists:
        hostname = os.getenv("RUNNER_NAME")
        if hostname is None:
            hostname = socket.gethostname()
        else:
            # Ensure RUNNER_NAME has not been tampered with:
            # TODO is this overkill?
            Validate.runner_name(
                hostname,
                throw=ValueError("Illegal characters found in specified RUNNER_NAME."),
            )

        compute_runtime = None
        if options.build_compute_runtime:
            compute_runtime = options.compute_runtime_tag
        elif options.detect_versions.compute_runtime:
            print(f"Auto-detecting compute_runtime version...")
            detect_res = DetectVersion.instance()
            compute_runtime = detect_res.get_compute_runtime_ver()
            if detect_res.get_compute_runtime_ver_cached() is None:
                print(
                    "Warning: Could not find compute_runtime version via github tags API."
                )
        else:
            compute_runtime = "unknown"

        return BenchmarkRun(
            name=name,
            git_hash=git_hash,
            github_repo=github_repo,
            date=datetime.now(tz=timezone.utc),
            results=results,
            hostname=hostname,
            compute_runtime=compute_runtime,
        )

    def save(self, save_name, results: list[Result], to_file=True):
        benchmark_data = self.create_run(save_name, results)
        self.runs.append(benchmark_data)

        if not to_file:
            return

        serialized = benchmark_data.to_json()
        results_dir = Path(os.path.join(self.dir, "results"))
        os.makedirs(results_dir, exist_ok=True)

        # Use formatted timestamp for the filename
        timestamp = (
            datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            if options.timestamp_override is None
            else options.timestamp_override
        )
        file_path = Path(os.path.join(results_dir, f"{save_name}_{timestamp}.json"))
        with file_path.open("w") as file:
            json.dump(serialized, file, indent=4)
        print(f"Benchmark results saved to {file_path}")

    def find_first(self, name: str) -> BenchmarkRun:
        for r in self.runs:
            if r.name == name:
                return r
        return None

    def compute_average(self, data: list[BenchmarkRun]):
        first_run = data[0]
        average_results = []

        for i in range(len(first_run.results)):
            all_values = [run.results[i].value for run in data]

            # Calculate the average value for the current result index
            average_value = sum(all_values) / len(all_values)

            average_result = first_run.results[i]
            average_result.value = average_value

            average_results.append(average_result)

        average_benchmark_run = BenchmarkRun(
            results=average_results,
            name=first_run.name,
            git_hash="average",
            date=first_run.date,  # should this be different?
            hostname=first_run.hostname,
        )

        return average_benchmark_run

    def get_compare(self, name: str) -> BenchmarkRun:
        if options.compare == Compare.LATEST:
            return self.find_first(name)

        data = []
        for r in self.runs:
            if r.name == name:
                data.append(r)
                if len(data) == options.compare_max:
                    break

        if len(data) == 0:
            return None

        if options.compare == Compare.MEDIAN:
            return data[len(data) // 2]

        if options.compare == Compare.AVERAGE:
            return self.compute_average(data)

        raise Exception("invalid compare type")
