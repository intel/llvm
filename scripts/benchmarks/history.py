# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import json
from pathlib import Path
from benches.result import Result, BenchmarkRun
from benches.options import Compare, options
from datetime import datetime, timezone
from utils.utils import run;

class BenchmarkHistory:
    benchmark_run_index_max = 0
    runs = []

    def __init__(self, dir):
        self.dir = dir

    def load_result(self, file_path: Path) -> BenchmarkRun:
        if file_path.exists():
            with file_path.open('r') as file:
                data = json.load(file)
                return BenchmarkRun.from_json(data)
        else:
            return None

    def load(self, n: int):
        results_dir = Path(self.dir) / 'results'
        if not results_dir.exists() or not results_dir.is_dir():
            return []

        # Get all JSON files in the results directory
        benchmark_files = list(results_dir.glob('*.json'))

        # Extract index numbers and sort files by index number
        def extract_index(file_path: Path) -> int:
            try:
                return int(file_path.stem.split('_')[0])
            except (IndexError, ValueError):
                return -1

        benchmark_files = [file for file in benchmark_files if extract_index(file) != -1]
        benchmark_files.sort(key=extract_index)

        # Load the first n benchmark files
        benchmark_runs = []
        for file_path in benchmark_files[n::-1]:
            benchmark_run = self.load_result(file_path)
            if benchmark_run:
                benchmark_runs.append(benchmark_run)

        if benchmark_files:
            self.benchmark_run_index_max = extract_index(benchmark_files[-1])

        self.runs = benchmark_runs

    def create_run(self, name: str, results: list[Result]) -> BenchmarkRun:
        try:
            result = run("git rev-parse --short HEAD")
            git_hash = result.stdout.decode().strip()
        except:
            git_hash = 'unknown'

        return BenchmarkRun(name = name, git_hash = git_hash, date = datetime.now(tz=timezone.utc), results = results)

    def save(self, save_name, results: list[Result], to_file = True):
        benchmark_data = self.create_run(save_name, results)
        self.runs.append(benchmark_data)

        if not to_file:
            return

        serialized = benchmark_data.to_json()
        results_dir = Path(os.path.join(self.dir, 'results'))
        os.makedirs(results_dir, exist_ok=True)

        self.benchmark_run_index_max += 1
        file_path = Path(os.path.join(results_dir, f"{self.benchmark_run_index_max}_{save_name}.json"))
        with file_path.open('w') as file:
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
            results = average_results,
            name = first_run.name,
            git_hash = "average",
            date = first_run.date # should this be different?
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
