# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv
import io
import os
from pathlib import Path

from utils.utils import download
from .base import Benchmark, Suite, TracingType
from utils.result import Result
from options import options
from utils.oneapi import get_oneapi
from git_project import GitProject
from utils.logger import log


class LlamaCppBench(Suite):
    def __init__(self):
        self.project = None

    def name(self) -> str:
        return "llama.cpp bench"

    def git_url(self) -> str:
        return "https://github.com/ggerganov/llama.cpp"

    def git_hash(self) -> str:
        # 12 Nov, 2025
        return "78010a0d52ad03cd469448df89101579b225582c"

    def setup(self) -> None:
        if options.sycl is None:
            return

        if self.project is None:
            self.project = GitProject(
                self.git_url(),
                self.git_hash(),
                Path(options.workdir),
                "llamacpp",
            )

        models_dir = Path(options.workdir, "llamacpp-models")
        models_dir.mkdir(parents=True, exist_ok=True)

        self.model = download(
            models_dir,
            "https://huggingface.co/ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/resolve/main/deepseek-r1-distill-qwen-1.5b-q4_0.gguf",
            "deepseek-r1-distill-qwen-1.5b-q4_0.gguf",
            checksum="791f6091059b653a24924b9f2b9c3141c8f892ae13fff15725f77a2bf7f9b1b6b71c85718f1e9c0f26c2549aba44d191",
        )

        self.oneapi = get_oneapi()

        if not self.project.needs_rebuild():
            log.info(f"Rebuilding {self.project.name} skipped")
            return

        extra_args = [
            f"-DGGML_SYCL=ON",
            f"-DCMAKE_C_COMPILER=clang",
            f"-DCMAKE_CXX_COMPILER=clang++",
            f"-DDNNL_GPU_VENDOR=INTEL",
            f"-DTBB_DIR={self.oneapi.tbb_cmake()}",
            f"-DDNNL_DIR={self.oneapi.dnn_cmake()}",
            f"-DSYCL_COMPILER=ON",
            f"-DMKL_DIR={self.oneapi.mkl_cmake()}",
        ]
        self.project.configure(extra_args, add_sycl=True)
        self.project.build(add_sycl=True, ld_library=self.oneapi.ld_libraries())

    def benchmarks(self) -> list[Benchmark]:
        return [LlamaBench(self)]


class LlamaBench(Benchmark):
    def __init__(self, suite: LlamaCppBench):
        super().__init__(suite)
        self.suite = suite

    @property
    def benchmark_bin(self) -> Path:
        return self.suite.project.build_dir / "bin" / "llama-bench"

    def enabled(self):
        if options.sycl is None:
            return False
        if options.ur_adapter == "cuda" or options.ur_adapter == "hip":
            return False
        return True

    def model(self):
        return "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"

    def name(self):
        return f"llama.cpp {self.model()}"

    def description(self) -> str:
        return (
            "Performance testing tool for llama.cpp that measures LLM inference speed in tokens per second. "
            "Runs both prompt processing (initial context processing) and text generation benchmarks with "
            f"different batch sizes. Higher values indicate better performance. Uses the {self.model()} "
            "quantized model and leverages SYCL with oneDNN for acceleration."
        )

    def get_tags(self):
        return ["SYCL", "application", "inference", "throughput"]

    def lower_is_better(self):
        return False

    def run(
        self,
        env_vars,
        run_trace: TracingType = TracingType.NONE,
        force_trace: bool = False,
    ) -> list[Result]:
        command = [
            str(self.benchmark_bin),
            "--output",
            "csv",
            "-n",
            "128",
            "-p",
            "512",
            "-pg",
            "0,0",
            "-sm",
            "none",
            "-ngl",
            "99",
            "--numa",
            "isolate",
            "-t",
            "8",
            "--mmap",
            "0",
            "--model",
            f"{self.suite.model}",
        ]

        result = self.run_bench(
            command,
            env_vars,
            ld_library=self.suite.oneapi.ld_libraries(),
            run_trace=run_trace,
            force_trace=force_trace,
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
                    unit="token/s",
                    git_url=self.suite.git_url(),
                    git_hash=self.suite.git_hash(),
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
