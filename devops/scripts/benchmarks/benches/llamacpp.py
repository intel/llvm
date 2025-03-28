# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv
import io
from pathlib import Path
from utils.utils import download, git_clone
from .base import Benchmark, Suite
from utils.result import Result
from utils.utils import run, create_build_path
from options import options
from utils.oneapi import get_oneapi
import os


class LlamaCppBench(Suite):
    def __init__(self, directory):
        if options.sycl is None:
            return

        self.directory = directory

    def name(self) -> str:
        return "llama.cpp bench"

    def git_url(self) -> str:
        return "https://github.com/ggerganov/llama.cpp"

    def git_hash(self) -> str:
        return "1ee9eea094fe5846c7d8d770aa7caa749d246b23"

    def setup(self):
        if options.sycl is None:
            return

        repo_path = git_clone(
            self.directory,
            "llamacpp-repo",
            self.git_url(),
            self.git_hash(),
        )

        self.models_dir = os.path.join(self.directory, "models")
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.model = download(
            self.models_dir,
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            "Phi-3-mini-4k-instruct-q4.gguf",
            checksum="fc4f45c9729874a33a527465b2ec78189a18e5726b7121182623feeae38632ace4f280617b01d4a04875acf49d263ee4",
        )

        self.oneapi = get_oneapi()

        self.build_path = create_build_path(self.directory, "llamacpp-build")

        configure_command = [
            "cmake",
            f"-B {self.build_path}",
            f"-S {repo_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DGGML_SYCL=ON",
            f"-DCMAKE_C_COMPILER=clang",
            f"-DCMAKE_CXX_COMPILER=clang++",
            f"-DDNNL_DIR={self.oneapi.dnn_cmake()}",
            f"-DTBB_DIR={self.oneapi.tbb_cmake()}",
            f'-DCMAKE_CXX_FLAGS=-I"{self.oneapi.mkl_include()}"',
            f"-DCMAKE_SHARED_LINKER_FLAGS=-L{self.oneapi.compiler_lib()} -L{self.oneapi.mkl_lib()}",
        ]

        run(configure_command, add_sycl=True)

        run(
            f"cmake --build {self.build_path} -j {options.build_jobs}",
            add_sycl=True,
            ld_library=self.oneapi.ld_libraries(),
        )

    def benchmarks(self) -> list[Benchmark]:
        if options.sycl is None:
            return []

        if options.ur_adapter == "cuda" or options.ur_adapter == "hip":
            return []

        return [LlamaBench(self)]


class LlamaBench(Benchmark):
    def __init__(self, bench):
        super().__init__(bench.directory, bench)
        self.bench = bench

    def setup(self):
        self.benchmark_bin = os.path.join(self.bench.build_path, "bin", "llama-bench")

    def name(self):
        return f"llama.cpp"

    def description(self) -> str:
        return (
            "Performance testing tool for llama.cpp that measures LLM inference speed in tokens per second. "
            "Runs both prompt processing (initial context processing) and text generation benchmarks with "
            "different batch sizes. Higher values indicate better performance. Uses the Phi-3-mini-4k-instruct "
            "quantized model and leverages SYCL with oneDNN for acceleration."
        )

    def get_tags(self):
        return ["SYCL", "application", "inference", "throughput"]

    def lower_is_better(self):
        return False

    def run(self, env_vars) -> list[Result]:
        command = [
            f"{self.benchmark_bin}",
            "--output",
            "csv",
            "-n",
            "128",
            "-p",
            "512",
            "-b",
            "128,256,512",
            "--numa",
            "isolate",
            "-t",
            "56",  # TODO: use only as many threads as numa node 0 has cpus
            "--model",
            f"{self.bench.model}",
        ]

        result = self.run_bench(
            command, env_vars, ld_library=self.bench.oneapi.ld_libraries()
        )
        parsed = self.parse_output(result)
        results = []
        for r in parsed:
            (extra_label, mean) = r
            label = f"{self.name()} {extra_label}"
            results.append(
                Result(
                    label=label,
                    value=mean,
                    command=command,
                    env=env_vars,
                    stdout=result,
                    unit="token/s",
                    git_url=self.bench.git_url(),
                    git_hash=self.bench.git_hash(),
                )
            )
        return results

    def parse_output(self, output):
        csv_file = io.StringIO(output)
        reader = csv.DictReader(csv_file)

        results = []
        for row in reader:
            try:
                n_batch = row["n_batch"]
                avg_ts = float(row["avg_ts"])
                n_prompt = int(row["n_prompt"])
                label = "Prompt Processing" if n_prompt != 0 else "Text Generation"
                label += f" Batched {n_batch}"
                results.append((label, avg_ts))
            except KeyError as e:
                raise ValueError(f"Error parsing output: {e}")

        return results

    def teardown(self):
        return
