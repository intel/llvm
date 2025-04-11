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
        models = [
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
        return [GromacsSystemBenchmark(self, model) for model in models]

    def setup(self):
        if not (self.gromacs_src).exists():
            self.gromacs_src = git_clone(
                self.directory,
                "gromacs-repo",
                self.git_url(),
                self.git_tag(),
            )
        else:
            if options.verbose:
                print(f"GROMACS repository already exists at {self.gromacs_src}")

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
            print(f"Grappa tar file downloaded and extracted to {model}")
        else:
            print(f"Grappa tar file already exists at {grappa_tar_file}")

    def teardown(self):
        pass


class GromacsSystemBenchmark(Benchmark):
    def __init__(self, suite, model):
        self.suite = suite
        self.model = model  # The model name (e.g., "0001.5")
        self.gromacs_src = suite.gromacs_src
        self.grappa_dir = suite.grappa_dir
        self.gmx_path = suite.gromacs_build_path / "bin" / "gmx"

    def name(self):
        return f"gromacs-{self.model}"

    def setup(self):
        pass

    def run(self, env_vars):
        if not self.gmx_path.exists():
            raise FileNotFoundError(f"gmx executable not found at {self.gmx_path}")

        system_dir = self.grappa_dir / self.model

        if not system_dir.exists():
            raise FileNotFoundError(f"System directory not found: {system_dir}")

        env_vars.update(
            {
                "LD_LIBRARY_PATH": str(self.grappa_dir)
                + os.pathsep
                + os.environ.get("LD_LIBRARY_PATH", ""),
                "SYCL_CACHE_PERSISTENT": "1",
                "GMX_CUDA_GRAPH": "1",
            }
        )

        try:
            # Generate configurations for RF
            rf_grompp_result = run(
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
                add_sycl=True,
            )

            # Run RF benchmark
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
            rf_mdrun_result = run(
                rf_command,
                add_sycl=True,
            )
            rf_mdrun_result_output = rf_mdrun_result.stderr.decode()
            rf_time = self._extract_execution_time(rf_mdrun_result_output, "RF")

            print(f"[{self.name()}-RF] Time: {rf_time:.3f} seconds")

            # Generate configurations for PME
            pme_grompp_result = run(
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
                add_sycl=True,
            )

            # Run PME benchmark
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
            pme_mdrun_result = run(
                pme_command,
                add_sycl=True,
            )

            pme_mdrun_result_output = pme_mdrun_result.stderr.decode()

            pme_time = self._extract_execution_time(pme_mdrun_result_output, "PME")
            print(f"[{self.name()}-PME] Time: {pme_time:.3f} seconds")

        except subprocess.CalledProcessError as e:
            print(f"Error during execution of {self.name()}: {e}")
            raise

        # Return results as a list of Result objects
        return [
            self._result(
                "RF",
                rf_time,
                rf_grompp_result,
                rf_mdrun_result,
                rf_command,
                env_vars,
            ),
            self._result(
                "PME",
                pme_time,
                pme_grompp_result,
                pme_mdrun_result,
                pme_command,
                env_vars,
            ),
        ]

    def _result(self, label, time, gr_result, md_result, command, env_vars):
        return Result(
            label=f"{self.name()}-{label}",
            value=time,
            unit="s",
            passed=(gr_result.returncode == 0 and md_result.returncode == 0),
            command=" ".join(map(str, command)),
            env=env_vars,
            stdout=gr_result.stderr.decode() + md_result.stderr.decode(),
            git_url=self.suite.git_url(),
            git_hash=self.suite.git_tag(),
        )

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

    # def _extract_first_number(self, line):
    #     parts = line.split()
    #     for part in parts:
    #         if part.replace(".", "", 1).isdigit():
    #             return float(part)
    #     return None

    # def _parse_result(self, result, benchmark_type, execution_time):
    #     passed = result.returncode == 0
    #     return {
    #         "type": f"{self.name()}-{benchmark_type}",
    #         "passed": passed,
    #         "execution_time": execution_time,
    #         "output": result.stdout,
    #         "error": result.stderr if not passed else None,
    #     }

    def teardown(self):
        pass
