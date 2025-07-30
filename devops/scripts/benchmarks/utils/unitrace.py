# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import shutil

from options import options
from utils.utils import run, git_clone
from utils.logger import log

from datetime import datetime, timezone


class Unitrace:
    """Unitrace wrapper for managing Unitrace tool execution and results."""

    def __init__(self):
        self.timestamp = (
            datetime.now(tz=timezone.utc).strftime(options.TIMESTAMP_FORMAT)
            if options.timestamp_override is None
            else options.timestamp_override
        )

        log.info("Downloading and building Unitrace...")
        repo_dir = git_clone(
            options.workdir,
            "pti-gpu-repo",
            "https://github.com/intel/pti-gpu.git",
            "master",
        )
        build_dir = os.path.join(options.workdir, "unitrace-build")
        unitrace_src = os.path.join(repo_dir, "tools", "unitrace")
        os.makedirs(build_dir, exist_ok=True)

        unitrace_exe = os.path.join(build_dir, "unitrace")
        if not os.path.isfile(unitrace_exe):
            run(
                [
                    "cmake",
                    f"-S {unitrace_src}",
                    f"-B {build_dir}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_COMPILER=clang++",
                    "-DCMAKE_C_COMPILER=clang",
                    "-DBUILD_WITH_L0=1",
                    "-DBUILD_WITH_OPENCL=0",
                    "-DBUILD_WITH_ITT=1",
                    "-DBUILD_WITH_XPTI=1",
                    "-DBUILD_WITH_MPI=0",
                ],
                add_sycl=True,
            )
            run(["cmake", "--build", build_dir, "-j", str(options.build_jobs)])
            log.info("Unitrace built successfully.")
        else:
            log.info("Unitrace build skipped (already built).")

        if options.results_directory_override == None:
            self.traces_dir = os.path.join(options.workdir, "results", "traces")
        else:
            self.traces_dir = os.path.join(options.results_directory_override, "traces")

    def _prune_unitrace_dirs(self, res_dir: str, FILECNT: int = 10):
        """Keep only the last FILECNT files in the traces directory."""
        files = os.listdir(res_dir)
        files.sort()  # Lexicographical sort matches timestamp order
        if len(files) > 2 * FILECNT:
            for f in files[: len(files) - 2 * FILECNT]:
                full_path = os.path.join(res_dir, f)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                    log.debug(f"Removing old unitrace file: {full_path}")

    def cleanup(self, bench_cwd: str, unitrace_output: str):
        """
        Remove incomplete output files in case of failure.
        """
        unitrace_dir = os.path.dirname(unitrace_output)
        unitrace_base = os.path.basename(unitrace_output)
        for f in os.listdir(unitrace_dir):
            if f.startswith(unitrace_base + "."):
                os.remove(os.path.join(unitrace_dir, f))
                log.debug(f"Cleanup: Removed {f} from {unitrace_dir}")
        if os.path.exists(bench_cwd):
            for f in os.listdir(bench_cwd):
                if f.endswith(".json"):
                    os.remove(os.path.join(bench_cwd, f))
                    log.debug(f"Cleanup: Removed {f} from {bench_cwd}")

    def setup(
        self, bench_name: str, command: list[str], extra_unitrace_opt: list[str] = None
    ):
        """
        Prepare Unitrace output file name and full command for the benchmark run.
        Returns a tuple of (unitrace_output, unitrace_command).
        """
        unitrace_bin = os.path.join(options.workdir, "unitrace-build", "unitrace")
        if not os.path.exists(unitrace_bin):
            raise FileNotFoundError(f"Unitrace binary not found: {unitrace_bin}. ")
        os.makedirs(self.traces_dir, exist_ok=True)
        bench_dir = os.path.join(f"{self.traces_dir}", f"{bench_name}")

        os.makedirs(bench_dir, exist_ok=True)

        unitrace_output = os.path.join(
            bench_dir, f"{self.timestamp}_{options.save_name}.out"
        )

        if extra_unitrace_opt is None:
            extra_unitrace_opt = []

        unitrace_command = (
            [
                str(unitrace_bin),
                "--call-logging",
                "--host-timing",
                "--device-timing",
                "--chrome-sycl-logging",
                "--chrome-call-logging",
                "--chrome-kernel-logging",
                "--output",
                unitrace_output,
            ]
            + extra_unitrace_opt
            + command
        )
        log.debug(f"Unitrace cmd: {' '.join(unitrace_command)}")

        return unitrace_output, unitrace_command

    def handle_output(self, unitrace_output: str):
        """
        Handle .json trace files in cwd: move and rename to {self.name()}_{timestamp}.{pid}.json,
        to make them have the same name as unitrace log file and moved to the same directory.
        """

        pid_json_files = []
        pid = ""
        for f in os.listdir(options.benchmark_cwd):
            parts = f.split(".")
            l = len(parts)
            # make sure if filename is in the format {name}.{pid}.json
            if len(parts) >= 3 and parts[l - 1] == "json" and parts[l - 2].isdigit():
                pid_json_files.append(f)
                pid = parts[l - 2]
            else:
                log.debug(
                    f"Skipping renaming of {f} as the name does not match the expected format."
                )

        # If nothing went wrong, cwd contains only one .pid.json file, but we cannot be sure
        if len(pid_json_files) == 0:
            raise FileNotFoundError(
                f"No .pid.json files found in {options.benchmark_cwd}."
            )
        elif len(pid_json_files) > 1:
            # If there are multiple .pid.json files due to previous failures, keep only the most recent one
            pid_json_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(options.benchmark_cwd, f))
            )
            for f in pid_json_files[:-1]:
                os.remove(os.path.join(options.benchmark_cwd, f))

        # unitrace_output variable is in the format {name}.out, but unitrace makes
        # the actual file name as {name}.{pid}.out and we want .json file name to follow the same pattern
        json_name = unitrace_output[: -len(".out")] + f".{pid}.json"

        # even if the pid_json_files contains more entries, only the last one is valid
        shutil.move(os.path.join(options.benchmark_cwd, pid_json_files[-1]), json_name)
        log.debug(f"Moved {pid_json_files[-1]} to {json_name}")

        # Prune old unitrace directories
        self._prune_unitrace_dirs(os.path.dirname(unitrace_output))


# Singleton pattern to ensure only one instance of Unitrace is created
def get_unitrace() -> Unitrace:
    if not hasattr(get_unitrace, "_instance"):
        get_unitrace._instance = Unitrace()
    return get_unitrace._instance
