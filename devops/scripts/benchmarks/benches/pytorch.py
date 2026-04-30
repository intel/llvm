# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PyTorch benchmarks suite: run out-of-tree PyTorch benchmarks and parse results."""

import json
import sys
import tempfile
from pathlib import Path

from .base import Benchmark, Suite
from options import options
from utils.logger import log
from utils.result import Result


def is_pytorch_bench_available() -> bool:
    return options.pytorch_root is not None and Path(options.pytorch_root).is_dir()


class PytorchBenchSuite(Suite):
    def name(self) -> str:
        return "PyTorch Benchmarks"

    def setup(self) -> None:
        if not is_pytorch_bench_available():
            return

    def benchmarks(self) -> list[Benchmark]:
        return [
            RealWorldAppBench(self),
        ]


class RealWorldAppBench(Benchmark):
    """Runs ``benchmarks/dynamo/microbenchmarks/xpu/real_world_app.py`` from ``options.pytorch_root``."""

    _SCRIPT_NAME = "benchmarks/dynamo/microbenchmarks/xpu/real_world_app.py"

    def __init__(self, suite: PytorchBenchSuite):
        super().__init__(suite)
        self._script_path: Path | None = None
        if is_pytorch_bench_available():
            root = Path(options.pytorch_root)
            self._script_path = root / self._SCRIPT_NAME

    def name(self) -> str:
        return "real-world-app"

    def display_name(self) -> str:
        return "PyTorch Real-World App Microbenchmark"

    def enabled(self) -> bool:
        if not is_pytorch_bench_available():
            return False
        return self._script_path.is_file() if self._script_path else False

    def description(self) -> str:
        return "Measures real-world XPU application performance via PyTorch dynamo microbenchmark."

    def get_tags(self) -> list[str]:
        return ["pytorch", "micro", "inference", "latency"]

    def run(
        self,
        env_vars,
        flamegraph_enabled: bool = False,
        force_trace: bool = False,
    ) -> list[Result]:
        env_vars = dict(env_vars) if env_vars else {}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        command = [
            sys.executable,
            str(self._script_path),
            "--device",
            "xpu",
            "--iter",
            "10000",
            "--model",
            "vit",
            "--batch",
            "50",
            "--output-json",
            tmp_path,
        ]

        self.run_bench(
            command,
            env_vars,
            add_sycl=False,
            flamegraph_enabled=flamegraph_enabled,
            force_trace=force_trace,
        )
        return self.parse_output(tmp_path, command, env_vars)

    def parse_output(
        self, output_path: str, command: list[str], env_vars: dict
    ) -> list[Result]:
        with open(output_path) as f:
            raw = f.read()
        data = json.loads(raw)

        results: list[Result] = []
        for key, value in data.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            # Infer unit from key suffix (e.g. "latency_ms" -> "ms").
            unit = key.rsplit("_", 1)[-1] if "_" in key else ""
            results.append(
                Result(
                    label=key,
                    value=numeric,
                    command=command,
                    env=env_vars,
                    unit=unit,
                )
            )
        if not results:
            log.info(f"Raw benchmark output:\n{raw}")
            raise ValueError(
                "Could not extract any numeric metrics from benchmark JSON output"
            )
        return results
