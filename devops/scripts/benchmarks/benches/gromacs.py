# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from .base import Suite, Benchmark
from options import options
from utils.utils import git_clone
from utils.result import Result


class GromacsBench(Suite):
    GROMACS_REPO = "https://gitlab.com/gromacs/gromacs.git"
    GROMACS_TAG = "v2025.1"
    GRAPPA_BENCHMARKS_URL = (
        "https://zenodo.org/record/11234002/files/grappa-1.5k-6.1M_rc0.9.tar.gz"
    )

    def __init__(self, directory):
        # Initialize GromacsBench-specific attributes
        self.directory = Path(directory).resolve()
        self.gromacs_dir = self.directory / "gromacs"
        self.grappa_dir = self.directory / "grappa-1.5k-6.1M_rc0.9"
        self.build_dir = self.gromacs_dir / "build"

    def name(self):
        return "Gromacs Bench"

    def benchmarks(self) -> list[Benchmark]:
        systems = [
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
        return [
            GromacsSystemBenchmark(self, system, self.gromacs_dir, self.grappa_dir)
            for system in systems
        ]

    def setup(self):
        print(f"Working directory: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)

        if not self.gromacs_dir.exists():
            print(
                f"Cloning GROMACS repository (tag: {self.GROMACS_TAG}) into {self.gromacs_dir}..."
            )
            repo_path = git_clone(
                self.directory,
                "gromacs-repo",
                self.GROMACS_REPO,
                self.GROMACS_TAG,
            )
            print(f"GROMACS repository cloned to {repo_path}")
        else:
            print(f"GROMACS repository already exists at {self.gromacs_dir}")

        # Build GROMACS
        self.build_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building GROMACS in {self.build_dir}...")
        subprocess.run(
            [
                "cmake",
                "../",
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CXX_COMPILER={options.sycl}/bin/clang++",
                f"-DCMAKE_C_COMPILER={options.sycl}/bin/clang",
                f"-DGMX_GPU=SYCL",
                f"-DGMX_SYCL_ENABLE_GRAPHS=ON",
                f"-DGMX_FFT_LIBRARY=MKL",
                f"-DGMX_BUILD_OWN_FFTW=ON",
                f"-DGMX_GPU_FFT_LIBRARY=MKL",
                f"-DGMX_GPU_NB_CLUSTER_SIZE=8",
                f"-DGMX_OPENMP=OFF",
            ],
            check=True,
            cwd=self.build_dir,  # Ensure the command runs in the build directory
        )
        subprocess.run(["make", "-j"], check=True, cwd=self.build_dir)

        if not self.grappa_dir.exists():
            self.download_and_extract_grappa()
        else:
            print(f"GRAPPA benchmarks already exist at {self.grappa_dir}")

    def download_and_extract_grappa(self):
        """Download and extract the GRAPPA benchmarks."""
        grappa_tar_path = self.directory / os.path.basename(self.GRAPPA_BENCHMARKS_URL)

        # Download the GRAPPA tar.gz file
        if not grappa_tar_path.exists():
            print(f"Downloading GRAPPA benchmarks from {self.GRAPPA_BENCHMARKS_URL}...")
            urllib.request.urlretrieve(self.GRAPPA_BENCHMARKS_URL, grappa_tar_path)

        # Extract the GRAPPA tar.gz file
        print(f"Extracting GRAPPA benchmarks to {self.directory}...")
        with tarfile.open(grappa_tar_path, "r:gz") as tar:
            tar.extractall(path=self.directory)

    def teardown(self):
        print(f"Tearing down GROMACS suite in {self.directory}...")
        pass


class GromacsSystemBenchmark(Benchmark):
    def __init__(self, suite, system, gromacs_dir, grappa_dir):
        self.suite = suite
        self.system = system  # The system name (e.g., "0001.5")
        self.gromacs_dir = gromacs_dir
        self.grappa_dir = grappa_dir
        self.gmx_path = gromacs_dir / "build" / "bin" / "gmx"

    def name(self):
        return f"gromacs-{self.system}"

    def setup(self):
        system_dir = self.grappa_dir / self.system
        if not system_dir.exists():
            raise FileNotFoundError(f"System directory not found: {system_dir}")
        print(f"Setting up benchmark for system: {self.system}")

    def run(self, env_vars):
        if not self.gmx_path.exists():
            raise FileNotFoundError(f"gmx executable not found at {self.gmx_path}")

        env_vars.update(
            {
                "LD_LIBRARY_PATH": f"{options.sycl}/lib"
                + os.pathsep
                + os.environ.get("LD_LIBRARY_PATH", ""),
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "SYCL_CACHE_PERSISTENT": "1",
                "GMX_CUDA_GRAPH": "1",
                "SYCL_UR_USE_LEVEL_ZERO_V2": "1",
            }
        )

        system_dir = self.grappa_dir / self.system

        if not system_dir.exists():
            raise FileNotFoundError(f"System directory not found: {system_dir}")

        rf_log_file = self.grappa_dir / f"{self.name()}-rf.log"
        pme_log_file = self.grappa_dir / f"{self.name()}-pme.log"

        try:
            # Generate configurations for RF
            if options.verbose:
                print(f"Running grompp for RF benchmark: {self.name()}")
            subprocess.run(
                [
                    str(self.gmx_path),
                    "grompp",
                    "-f",
                    str(self.grappa_dir / "rf.mdp"),
                    "-c",
                    str(system_dir / "conf.gro"),
                    "-p",
                    str(system_dir / "topol.top"),
                    "-o",
                    str(system_dir / "rf.tpr"),
                ],
                check=True,
                stdout=open(rf_log_file, "w"),
                stderr=subprocess.STDOUT,
                env=env_vars,
            )

            # Run RF benchmark
            if options.verbose:
                print(f"Running mdrun for RF benchmark: {self.name()}")
            rf_command = [
                str(self.gmx_path),
                "mdrun",
                "-s",
                str(system_dir / "rf.tpr"),
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
            rf_result = subprocess.run(
                rf_command,
                check=True,
                stdout=open(rf_log_file, "a"),
                stderr=subprocess.STDOUT,
                env=env_vars,
            )
            rf_time = self._extract_execution_time(rf_log_file, "RF")
            if options.verbose:
                print(f"[{self.name()}-RF] Time: {rf_time:.3f} seconds")

            # Generate configurations for PME
            if options.verbose:
                print(f"Running grompp for PME benchmark: {self.name()}")
            subprocess.run(
                [
                    str(self.gmx_path),
                    "grompp",
                    "-f",
                    str(self.grappa_dir / "pme.mdp"),
                    "-c",
                    str(system_dir / "conf.gro"),
                    "-p",
                    str(system_dir / "topol.top"),
                    "-o",
                    str(system_dir / "pme.tpr"),
                ],
                check=True,
                stdout=open(pme_log_file, "w"),
                stderr=subprocess.STDOUT,
                env=env_vars,
            )

            # Run PME benchmark
            if options.verbose:
                print(f"Running mdrun for PME benchmark: {self.name()}")
            pme_command = [
                str(self.gmx_path),
                "mdrun",
                "-s",
                str(system_dir / "pme.tpr"),
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
            pme_result = subprocess.run(
                pme_command,
                check=True,
                stdout=open(pme_log_file, "a"),
                stderr=subprocess.STDOUT,
                env=env_vars,
            )
            pme_time = self._extract_execution_time(pme_log_file, "PME")
            if options.verbose:
                print(f"[{self.name()}-PME] Time: {pme_time:.3f} seconds")

        except subprocess.CalledProcessError as e:
            print(f"Error during execution of {self.name()}: {e}")
            raise

        # Return results as a list of Result objects
        return [
            Result(
                label=f"{self.name()}-RF",
                value=rf_time,
                unit="seconds",
                passed=rf_result.returncode == 0,
                command=" ".join(map(str, rf_command)),
                env={k: str(v) for k, v in env_vars.items()},
                stdout=str(rf_log_file),
            ),
            Result(
                label=f"{self.name()}-PME",
                value=pme_time,
                unit="seconds",
                passed=pme_result.returncode == 0,
                command=" ".join(map(str, pme_command)),
                env={k: str(v) for k, v in env_vars.items()},
                stdout=str(pme_log_file),
            ),
        ]

    def _extract_execution_time(self, log_file, benchmark_type):
        with open(log_file, "r") as log:
            time_lines = [line for line in log if "Time:" in line]

            if len(time_lines) != 1:
                raise ValueError(
                    f"Expected exactly 1 line containing 'Time:' in the log file for {benchmark_type}, "
                    f"but found {len(time_lines)}. Log file: {log_file}"
                )

            return self._extract_first_number(time_lines[0])

    def _extract_first_number(self, line):
        parts = line.split()
        for part in parts:
            if part.replace(".", "", 1).isdigit():
                return float(part)
        return None

    def _parse_result(self, result, benchmark_type, execution_time):
        passed = result.returncode == 0
        return {
            "type": f"{self.name()}-{benchmark_type}",
            "passed": passed,
            "execution_time": execution_time,  # Include the extracted execution time
            "output": result.stdout,
            "error": result.stderr if not passed else None,
        }

    def teardown(self):
        pass
