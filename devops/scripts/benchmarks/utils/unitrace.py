# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import shutil
import re

from options import options
from utils.utils import run, git_clone
from utils.oneapi import get_oneapi


def extract_save_name_and_timestamp(dirname):
    """
    Extracts (save_name, timestamp) from a directory name of the form {save_name}_{timestamp},
    where timestamp is always 15 characters: YYYYMMDD_HHMMSS.
    save_name may contain underscores.
    """
    m = re.match(r"(.+)_(\d{8}_\d{6})$", dirname)
    if m:
        return m.group(1), m.group(2)
    return None, None


def prune_unitrace_dirs(base_dir, FILECNT=10):
    """
    Keeps only FILECNT newest directories for each save_name group in base_dir.
    """
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    groups = {}
    for d in dirs:
        save_name, ts = extract_save_name_and_timestamp(d)
        if save_name and ts:
            groups.setdefault(save_name, []).append((d, ts))
    for save_name, dirlist in groups.items():
        # Sort by timestamp string (lexicographically, works for YYYYMMDD_HHMMSS)
        dirlist.sort(key=lambda x: x[1])
        if len(dirlist) > FILECNT:
            for d, ts in dirlist[: len(dirlist) - FILECNT]:
                full_path = os.path.join(base_dir, d)
                print(f"Removing old unitrace dir: {full_path}")
                shutil.rmtree(full_path)


def unitrace_cleanup(bench_cwd, unitrace_output):
    # Remove .pid files from the benchmark directory and .json files from cwd
    unitrace_dir = os.path.dirname(unitrace_output)
    unitrace_base = os.path.basename(unitrace_output)
    print(f"Cleanup unitrace output {unitrace_base} from {unitrace_dir}")
    for f in os.listdir(unitrace_dir):
        if f.startswith(unitrace_base + "."):
            os.remove(os.path.join(unitrace_dir, f))
            print(f"Cleanup: Removed {f} from {unitrace_dir}")
    if os.path.exists(bench_cwd):
        for f in os.listdir(bench_cwd):
            if f.endswith(".json"):
                os.remove(os.path.join(bench_cwd, f))
                print(f"Cleanup: Removed {f} from {bench_cwd}")


def unitrace_prepare(name, unitrace_timestamp, command, extra_unitrace_opt=[]):
    unitrace_bin = os.path.join(options.workdir, "unitrace-build", "unitrace")
    if not os.path.exists(unitrace_bin):
        raise FileNotFoundError(f"Unitrace binary not found: {unitrace_bin}. ")
    os.makedirs(options.unitrace_res_dir, exist_ok=True)
    if not options.save_name:
        raise ValueError(
            "Unitrace requires a save name to be specified via --save option."
        )
    bench_dir = f"{options.unitrace_res_dir}/{options.save_name}_{unitrace_timestamp}"
    os.makedirs(bench_dir, exist_ok=True)

    unitrace_output = f"{bench_dir}/{name}_{unitrace_timestamp}"
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
    if options.verbose:
        print(f"Unitrace cmd: {' '.join(unitrace_command)}")

    return bench_dir, unitrace_output, unitrace_command


def handle_unitrace_output(bench_dir, unitrace_output, timestamp):
    # Handle unitrace_output.{pid} logs: rename to unitrace_output (remove pid)
    for f in os.listdir(bench_dir):
        if f.startswith(os.path.basename(unitrace_output) + "."):
            parts = f.rsplit(".", 1)
            if (
                len(parts) == 2
                and parts[1].isdigit()
                and os.path.isfile(os.path.join(bench_dir, f))
            ):
                src = os.path.join(bench_dir, f)
                dst = os.path.join(bench_dir, os.path.basename(unitrace_output))
                shutil.move(src, dst)
                if options.verbose:
                    print(f"Renamed {src} to {dst}")
                break

    # Handle {name}.{pid}.json files in cwd: move and rename to {self.name()}_{timestamp}.json
    pid_json_files = []
    for f in os.listdir(options.benchmark_cwd):
        parts = f.split(".")
        l = len(parts)
        if len(parts) >= 3 and parts[l - 1] == "json" and parts[l - 2].isdigit():
            pid_json_files.append(f)

    if len(pid_json_files) == 1:
        dst = f"{unitrace_output}.json"
    else:
        print(
            f"Warning: Found {len(pid_json_files)} files matching the pattern. Expected 1."
        )
        # Find the newest file by modification time
        newest_file = max(
            pid_json_files,
            key=lambda f: os.path.getmtime(os.path.join(options.benchmark_cwd, f)),
        )
        dst = f"{unitrace_output}.json"
        for f in pid_json_files:
            if f != newest_file:
                os.remove(os.path.join(options.benchmark_cwd, f))
                if options.verbose:
                    print(f"Removed extra file {f}")

    shutil.move(os.path.join(options.benchmark_cwd, pid_json_files[0]), dst)
    if options.verbose:
        print(f"Moved {pid_json_files[0]} to {dst}")

    # Prune old unitrace directories
    prune_unitrace_dirs(options.unitrace_res_dir, FILECNT=5)


def download_and_build_unitrace(workdir):
    repo_dir = git_clone(
        workdir,
        "pti-gpu-repo",
        "https://github.com/intel/pti-gpu.git",
        "master",
    )
    build_dir = os.path.join(workdir, "unitrace-build")
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
            ld_library=get_oneapi().ld_libraries() + [f"{options.sycl}/lib"],
            add_sycl=True,
        )
        run(
            ["cmake", "--build", build_dir, "-j"],
            ld_library=get_oneapi().ld_libraries() + [f"{options.sycl}/lib"],
            add_sycl=True,
        )
    print("Unitrace built successfully.")
