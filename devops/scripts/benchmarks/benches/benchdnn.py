# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from .base import Suite, Benchmark
from options import options
from utils.utils import git_clone, run, create_build_path
from utils.result import Result
from utils.oneapi import get_oneapi


class OneDnnBench(Suite):
    def git_url(self):
        return "https://github.com/oneapi-src/oneDNN.git"

    def git_tag(self):
        return "v3.8""engine_kind": "gpu"/home/mateuszpn/oneDNN/tests/benchdnn/inputs/graph/complex_fusion/mha/MHA-stable_diffusion-inf-fp32-bs1.json

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        build_path = create_build_path(self.directory, "onednn-build")
        self.build_dir = Path(build_path)
        self.src_dir = self.directory / "onednn-repo"

    def name(self):
        return "BenchDNN"

    def benchmarks(self) -> list:
        return [
            # OneDnnBenchmark(
            #     self,
            #     "binary",
            #     "shapes",
            #     "--max-ms-per-prb=1000 --batch=shapes_perf_1st_conv",
            # ),
            # OneDnnBenchmark(
            #     self,
            #     "conv",
            #     "in8dst",
            #     "--max-ms-per-prb=100 --dt=f16:f16:s8,f16:f16:u8,bf16:bf16:s8,bf16:bf16:u8,f32:f32:s8,f32:f32:u8 --batch=shapes_3d_gpu",
            # ),
            OneDnnBenchmark(
                self,
                "rnn",
                "shapes-ds2",
                "--max-ms-per-prb=500 --alg=VANILLA_RNN,LBR_GRU --direction=left2right --batch=shapes_deepspeech_2",
            ),
            OneDnnBenchmark(
                self,
                "ip",
                "GPT-J",
                "--max-ms-per-prb=500 --batch=harness_ip_gpt-j_32-32_inf_lb_bfloat16",
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "abs",
                "--max-ms-per-prb=100 --dt=f32,bf16,f16 --op-kind=0:Abs,0:Exp,0:GELU,0:HardSwish,0:Log,0:Mish,0:Sigmoid,0:Sqrt,0:Square,0:Tanh --case=op/f32/abs.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "abs",
                "--max-ms-per-prb=100 --dt=f32,bf16,f16 --op-kind=0:Abs,0:Exp,0:GELU,0:HardSwish,0:Log,0:Mish,0:Sigmoid,0:Sqrt,0:Square,0:Tanh --case=op/f32/abs.json",
                eager=True,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "MHA-stable_diffusion",
                "--max-ms-per-prb=1000 --dt=f32,bf16,f16 --case=complex_fusion/mha/MHA-stable_diffusion-inf-fp32-bs1.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "MHA-stable_diffusion",
                "--max-ms-per-prb=1000 --dt=f32,bf16,f16 --case=complex_fusion/mha/MHA-stable_diffusion-inf-fp32-bs1.json",
                eager=True,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "MHA-stable_diffusion-rewritten",
                "--max-ms-per-prb=1000 --dt=f32,bf16,f16 --case=complex_fusion/mha/MHA-stable_diffusion-inf-fp32-bs1.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "MHA-stable_diffusion-rewritten",
                "--max-ms-per-prb=1000 --dt=f32,bf16,f16 --in-shapes=0:56x8x1024x80+1:56x8x77x80+2:56x8x77x80 --case=complex_fusion/mha/MHA-stable_diffusion-inf-fp32-bs1.json",
                eager=True,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "gated-mlp-int4",
                "--max-ms-per-prb=2500 --case=complex_fusion/mlp/gated-mlp-int4.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "gated-mlp-int4",
                "--max-ms-per-prb=2500 --case=complex_fusion/mlp/gated-mlp-int4.json",
                eager=True,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "GQA-fp16-v2",
                "--max-ms-per-prb=500 --dt=f32,bf16,f16 --case=complex_fusion/mha/GQA-fp16-v2.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "GQA-fp16-v2",
                "--max-ms-per-prb=500 --dt=f32,f16,bf16 --dt=f32,bf16,f16 --case=complex_fusion/mha/GQA-fp16-v2.json",
                eager=True,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "sdpa-plain",
                "--max-ms-per-prb=500 --dt=f32,bf16,f16 --op-kind=1:Multiply,1:Divide --case=complex_fusion/mha/sdpa-plain-simplified-f16.json",
                eager=False,
            ),
            OneDnnBenchmark(
                self,
                "graph",
                "sdpa-plain",
                "--max-ms-per-prb=500 --dt=f32,bf16,f16 --op-kind=1:Multiply,1:Divide --case=complex_fusion/mha/sdpa-plain-simplified-f16.json",
                eager=True,
            ),
        ]

    def setup(self):
        self.src_dir = git_clone(
            self.directory,
            "onednn-repo",
            self.git_url(),
            self.git_tag(),
        )

        self.oneapi = get_oneapi()
        cmake_args = [
            "cmake",
            f"-S {self.src_dir}",
            f"-B {self.build_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDNNL_BUILD_TESTS=ON",
            "-DDNNL_BUILD_EXAMPLES=OFF",
            "-DDNNL_CPU_RUNTIME=SYCL",  # Enable SYCL support
            "-DDNNL_GPU_RUNTIME=SYCL",  # Enable SYCL GPU support
        ]

        if not options.sycl == None:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={options.sycl}")

        run(
            cmake_args,
            add_sycl=True,
        )

        run(
            f"cmake --build {self.build_dir} --target benchdnn -j {options.build_jobs}",
            add_sycl=True,
            ld_library=[str(self.build_dir) + "/src"] + self.oneapi.ld_libraries(),
        )

    def teardown(self):
        pass


class OneDnnBenchmark(Benchmark):
    def __init__(self, suite, bench_driver, bench_name, bench_args, eager=False):
        self.suite = suite
        self.bench_name = f"{bench_driver}-{bench_name}"
        self.bench_args = f"--{bench_driver} --mode=p --engine=gpu --memory-kind=usm_device {bench_args}"
        self.exp_group = self.bench_name
        if bench_driver == "graph":
            if eager:
                self.bench_args += " --execution-mode=direct"
                self.bench_name += "-eager"
            else:
                self.bench_args += " --execution-mode=graph"
                self.bench_name += "-graph"
        self.bench_bin = suite.build_dir / "tests" / "benchdnn" / "benchdnn"

    def name(self):
        return f"onednn-{self.bench_name}"

    def explicit_group(self) -> str:
        return self.exp_group

    def setup(self):
        if not self.bench_bin.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {self.bench_bin}")

    def run(self, env_vars):
        command = [
            str(self.bench_bin),
            *self.bench_args.split(),
        ]

        ld_library = self.suite.oneapi.ld_libraries() + [
            str(self.suite.build_dir / "src")
        ]

        output = self.run_bench(
            command,
            env_vars,
            add_sycl=True,
            ld_library=ld_library,
            use_stdout=True,
        )
        result_value = self._extract_time(output)

        if options.verbose:
            print(f"[{self.name()}] Output: {output}")

        return [
            Result(
                label=self.name(),
                value=result_value,
                unit="ms",
                command=command,
                env=env_vars,
                stdout=output,
                git_url=self.suite.git_url(),
                git_hash=self.suite.git_tag(),
            )
        ]

    def _extract_time(self, output):
        lines = output.splitlines()
        idx_0time = None
        values = []
        for i, line in enumerate(lines):
            if line.startswith("Output template:"):
                template = line.replace("Output template: ", "").strip().split(",")
                try:
                    idx_0time = template.index("%0time%")
                except ValueError:
                    return 0.0
                continue
            if idx_0time is not None and line.startswith("perf,"):
                fields = line.strip().split(",")
                if len(fields) > idx_0time:
                    try:
                        values.append(float(fields[idx_0time]))
                    except Exception:
                        continue
        if values:
            return sum(values)
        return 0.0

    def teardown(self):
        pass
