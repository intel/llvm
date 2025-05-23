# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
from pathlib import Path
from .base import Suite, Benchmark
from options import options
from utils.utils import git_clone, download, run, create_build_path
from utils.result import Result
from utils.oneapi import get_oneapi
import re


class GromacsBench(Suite):

    def git_url(self):
        return "https://gitlab.com/gromacs/gromacs.git"

    def git_tag(self):
        return "v2025.2"

    def grappa_url(self):
        return "https://zenodo.org/record/11234002/files/grappa-1.5k-6.1M_rc0.9.tar.gz"

    def grappa_file(self):
        return Path(os.path.basename(self.grappa_url()))

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        model_path = str(self.directory / self.grappa_file()).replace(".tar.gz", "")
        self.grappa_dir = Path(model_path)
        build_path = create_build_path(self.directory, "gromacs-build")
        self.gromacs_build_path = Path(build_path)
        self.gromacs_src = self.directory / "gromacs-repo"

    def name(self):
        return "Gromacs Bench"

    def benchmarks(self) -> list[Benchmark]:
        return [
            GromacsBenchmark(self, "0006", "pme", "graphs"),
            GromacsBenchmark(self, "0006", "pme", "eager"),
            GromacsBenchmark(self, "0006", "rf", "graphs"),
            GromacsBenchmark(self, "0006", "rf", "eager"),
            # some people may need it
            # GromacsBenchmark(self, "0192", "pme", "eager"),
            # GromacsBenchmark(self, "0192", "rf", "eager"),
        ]

    def setup(self):
        self.gromacs_src = git_clone(
            self.directory,
            "gromacs-repo",
            self.git_url(),
            self.git_tag(),
        )

        # TODO: Detect the GPU architecture and set the appropriate flags

        # Build GROMACS

        self.oneapi = get_oneapi()

        run(
            [
                "cmake",
                f"-S {str(self.directory)}/gromacs-repo",
                f"-B {self.gromacs_build_path}",
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CXX_COMPILER=clang++",
                f"-DCMAKE_C_COMPILER=clang",
                f"-DGMX_GPU=SYCL",
                f"-DGMX_SYCL_ENABLE_GRAPHS=ON",
                f"-DGMX_SYCL_ENABLE_EXPERIMENTAL_SUBMIT_API=ON",
                f"-DGMX_FFT_LIBRARY=MKL",
                f"-DGMX_GPU_FFT_LIBRARY=MKL",
                f"-DMKLROOT={self.oneapi.mkl_dir()}",
                f"-DGMX_GPU_NB_CLUSTER_SIZE=8",
                f"-DGMX_GPU_NB_NUM_CLUSTER_PER_CELL_X=1",
                f"-DGMX_OPENMP=OFF",
            ],
            add_sycl=True,
        )
        run(
            f"cmake --build {self.gromacs_build_path} -j {options.build_jobs}",
            add_sycl=True,
            ld_library=self.oneapi.ld_libraries(),
        )
        download(
            self.directory,
            self.grappa_url(),
            self.directory / self.grappa_file(),
            checksum="cc02be35ba85c8b044e47d097661dffa8bea57cdb3db8b5da5d01cdbc94fe6c8902652cfe05fb9da7f2af0698be283a2",
            untar=True,
        )

    def teardown(self):
        pass


class GromacsBenchmark(Benchmark):
    def __init__(self, suite, model, type, option):
        self.suite = suite
        self.model = model  # The model name (e.g., "0001.5")
        self.type = type  # The type of benchmark ("pme" or "rf")
        self.option = option  # "graphs" or "eager"

        self.gromacs_src = suite.gromacs_src
        self.grappa_dir = suite.grappa_dir
        self.gmx_path = suite.gromacs_build_path / "bin" / "gmx"

        if self.type == "pme":
            self.extra_args = [
                "-pme",
                "gpu",
                "-pmefft",
                "gpu",
                "-notunepme",
            ]
        else:
            self.extra_args = []

    def name(self):
        return f"gromacs-{self.model}-{self.type}-{self.option}"

    def setup(self):
        if self.type != "rf" and self.type != "pme":
            raise ValueError(f"Unknown benchmark type: {self.type}")

        if self.option != "graphs" and self.option != "eager":
            raise ValueError(f"Unknown option: {self.option}")

        if not self.gmx_path.exists():
            raise FileNotFoundError(f"gmx executable not found at {self.gmx_path}")

        model_dir = self.grappa_dir / self.model

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        cmd_list = [
            str(self.gmx_path),
            "grompp",
            "-f",
            f"{str(self.grappa_dir)}/{self.type}.mdp",
            "-c",
            str(model_dir / "conf.gro"),
            "-p",
            str(model_dir / "topol.top"),
            "-o",
            f"{str(model_dir)}/{self.type}.tpr",
        ]

        # Generate configuration files
        self.conf_result = run(
            cmd_list,
            add_sycl=True,
            ld_library=self.suite.oneapi.ld_libraries(),
        )

    def run(self, env_vars):
        model_dir = self.grappa_dir / self.model

        env_vars.update({"SYCL_CACHE_PERSISTENT": "1"})

        if self.option == "graphs":
            env_vars.update({"GMX_CUDA_GRAPH": "1"})

        # Run benchmark
        command = [
            str(self.gmx_path),
            "mdrun",
            "-s",
            f"{str(model_dir)}/{self.type}.tpr",
            "-nb",
            "gpu",
            "-update",
            "gpu",
            "-bonded",
            "gpu",
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
            "-nobackup",
            "-noconfout",
            "-nstlist",
            "100",
            "-pin",
            "on",
            "-resethway",
        ] + self.extra_args

        mdrun_output = self.run_bench(
            command,
            env_vars,
            add_sycl=True,
            use_stdout=False,
            ld_library=self.suite.oneapi.ld_libraries(),
        )

        if not self._validate_correctness(options.benchmark_cwd + "/md.log"):
            raise ValueError(
                f"Validation failed: Conserved energy drift exceeds threshold in {model_dir / 'md.log'}"
            )

        time = self._extract_execution_time(mdrun_output)

        if options.verbose:
            print(f"[{self.name()}] Time: {time:.3f} seconds")

        return [
            Result(
                label=f"{self.name()}",
                value=time,
                unit="s",
                command=command,
                env=env_vars,
                stdout=mdrun_output,
                git_url=self.suite.git_url(),
                git_hash=self.suite.git_tag(),
            )
        ]

    def _extract_execution_time(self, log_content):
        # Look for the line containing "Time:"
        # and extract the first numeric value after it
        time_lines = [line for line in log_content.splitlines() if "Time:" in line]

        if len(time_lines) != 1:
            raise ValueError(
                f"Expected exactly 1 line containing 'Time:' in the log content, "
                f"but found {len(time_lines)}."
            )

        for part in time_lines[0].split():
            if part.replace(".", "", 1).isdigit():
                return float(part)

        raise ValueError(f"No numeric value found in the 'Time:' line.")

    def _validate_correctness(self, log_file):
        threshold = 1e-2  # Define an acceptable energy drift threshold

        log_file = Path(log_file)
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        sci_pattern = r"([-+]?\d*\.\d+(?:e[-+]?\d+)?)"
        with open(log_file, "r") as file:
            for line in file:
                if "Conserved energy drift:" in line:
                    match = re.search(sci_pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            drift_value = float(match.group(1))
                            return abs(drift_value) <= threshold
                        except ValueError:
                            print(
                                f"Parsed drift value: {drift_value} exceeds threshold"
                            )
                            return False
                    else:
                        raise ValueError(
                            f"No valid numerical value found in line: {line}"
                        )

        raise ValueError(f"Conserved Energy Drift not found in log file: {log_file}")

    def teardown(self):
        pass
