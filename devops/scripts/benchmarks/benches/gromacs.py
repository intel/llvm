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


class GromacsBench(Suite):

    def git_url(self):
        return "https://gitlab.com/gromacs/gromacs.git"

    def git_tag(self):
        return "v2025.1"

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
        models_rf = [
            "0001.5",
            "0003",
            "0006",
            "0012",
            "0024",
            "0048",
            "0096",
            "0192",
            "0384",
        ]
        benches_rf = [GromacsBenchmark(self, model, "rf") for model in models_rf]
        models_pme = [
            "0001.5",
            "0003",
            "0006",
            "0012",
            "0024",
            "0048",
            "0096",
            "0192",
            "0384",
        ]
        benches_pme = [GromacsBenchmark(self, model, "pme") for model in models_pme]
        # Add more models as needed

        return benches_rf + benches_pme

    def setup(self):
        self.gromacs_src = git_clone(
            self.directory,
            "gromacs-repo",
            self.git_url(),
            self.git_tag(),
        )

        # Build GROMACS
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
                f"-DGMX_FFT_LIBRARY=MKL",
                f"-DGMX_BUILD_OWN_FFTW=ON",
                f"-DGMX_GPU_FFT_LIBRARY=MKL",
                f"-DGMX_GPU_NB_CLUSTER_SIZE=8",
                f"-DGMX_OPENMP=OFF",
            ],
            add_sycl=True,
        )
        run(
            f"cmake --build {self.gromacs_build_path} -j {options.build_jobs}",
            add_sycl=True,
        )
        self.download_and_extract_grappa()

    def download_and_extract_grappa(self):
        grappa_tar_file = self.directory / self.grappa_file()

        if not grappa_tar_file.exists():
            model = download(
                self.directory,
                self.grappa_url(),
                grappa_tar_file,
                checksum="cc02be35ba85c8b044e47d097661dffa8bea57cdb3db8b5da5d01cdbc94fe6c8902652cfe05fb9da7f2af0698be283a2",
                untar=True,
            )
            if options.verbose:
                print(f"Grappa tar file downloaded and extracted to {model}")
        else:
            if options.verbose:
                print(f"Grappa tar file already exists at {grappa_tar_file}")

    def teardown(self):
        pass


class GromacsBenchmark(Benchmark):
    def __init__(self, suite, model, type):
        self.suite = suite
        self.model = model  # The model name (e.g., "0001.5")
        self.type = type
        self.gromacs_src = suite.gromacs_src
        self.grappa_dir = suite.grappa_dir
        self.gmx_path = suite.gromacs_build_path / "bin" / "gmx"

    def name(self):
        return f"gromacs-{self.model}"

    def setup(self):
        model_dir = self.grappa_dir / self.model
        if self.type == "rf":
            cmd_list = [
                str(self.gmx_path),
                "grompp",
                "-f",
                str(self.grappa_dir / "rf.mdp"),
                "-c",
                str(model_dir / "conf.gro"),
                "-p",
                str(model_dir / "topol.top"),
                "-o",
                str(model_dir / "rf.tpr"),
            ]
        elif self.type == "pme":
            cmd_list = [
                str(self.gmx_path),
                "grompp",
                "-f",
                str(self.grappa_dir / "pme.mdp"),
                "-c",
                str(model_dir / "conf.gro"),
                "-p",
                str(model_dir / "topol.top"),
                "-o",
                str(model_dir / "pme.tpr"),
            ]
        else:
            raise ValueError(f"Unknown benchmark type: {self.type}")

        # Generate configuration files
        self.conf_result = run(
            cmd_list,
            add_sycl=True,
        )

    def run(self, env_vars):
        if not self.gmx_path.exists():
            raise FileNotFoundError(f"gmx executable not found at {self.gmx_path}")

        model_dir = self.grappa_dir / self.model

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        env_vars.update(
            {
                "SYCL_CACHE_PERSISTENT": "1",
                "GMX_CUDA_GRAPH": "1",
            }
        )

        # Run benchmark
        if self.type == "rf":
            command = [
                str(self.gmx_path),
                "mdrun",
                "-s",
                str(model_dir / "rf.tpr"),
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
            ]
        else:  # type == "pme"
            command = [
                str(self.gmx_path),
                "mdrun",
                "-s",
                str(model_dir / "pme.tpr"),
                "-pme",
                "gpu",
                "-pmefft",
                "gpu",
                "-notunepme",
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
            ]

        mdrun_result = run(
            command,
            add_sycl=True,
        )
        mdrun_result_output = mdrun_result.stderr.decode()
        time = self._extract_execution_time(mdrun_result_output, type)

        if options.verbose:
            print(f"[{self.name()}-RF] Time: {time:.3f} seconds")

        # Return results as a list of Result objects
        return [
            Result(
                label=f"{self.name()}-{self.type}",
                value=time,
                unit="s",
                passed=(mdrun_result.returncode == 0),
                command=" ".join(map(str, command)),
                env=env_vars,
                stdout=self.conf_result.stderr.decode() + mdrun_result.stderr.decode(),
                git_url=self.suite.git_url(),
                git_hash=self.suite.git_tag(),
            )
        ]

    def _extract_execution_time(self, log_content, benchmark_type):
        # Look for the line containing "Time:"
        # and extract the first numeric value after it
        time_lines = [line for line in log_content.splitlines() if "Time:" in line]

        if len(time_lines) != 1:
            raise ValueError(
                f"Expected exactly 1 line containing 'Time:' in the log content for {benchmark_type}, "
                f"but found {len(time_lines)}."
            )

        for part in time_lines[0].split():
            if part.replace(".", "", 1).isdigit():
                return float(part)

        raise ValueError(
            f"No numeric value found in the 'Time:' line for {benchmark_type}."
        )

    def teardown(self):
        pass
