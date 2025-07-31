# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import platform
import subprocess
import os
from datetime import datetime
from utils.result import Platform
from options import options


def find_project_binaries():
    """Find the project's built binaries and libraries"""
    # If --sycl path is provided, use it directly
    if options.sycl:
        bin_dir = os.path.join(options.sycl, "bin")
        return bin_dir if os.path.exists(bin_dir) else None

    # Fallback: Start from the script's directory and go up to find the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # From utils/platform.py -> utils -> benchmarks -> scripts -> devops -> project_root
    project_root = current_dir
    for _ in range(4):  # Go up 4 levels to reach bench-scripts
        project_root = os.path.dirname(project_root)

    bin_dir = os.path.join(project_root, "build", "install", "bin")

    return bin_dir if os.path.exists(bin_dir) else None


def get_project_clang_version(bin_dir):
    """Get clang version from the project's built binary"""
    if not bin_dir:
        return "(unknown)"

    clang_path = os.path.join(bin_dir, "clang")
    try:
        result = subprocess.run(
            [clang_path, "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.split("\n")[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "(unknown)"


def get_level_zero_version_detailed():
    """Get Level Zero library version information"""

    workdir_ze_install = os.path.join(options.workdir, "level-zero-install", "lib")

    # Look for built Level Zero libraries
    if os.path.exists(workdir_ze_install):
        ze_loader_pattern = os.path.join(workdir_ze_install, "libze_loader.so.*")
        import glob
        import re

        ze_files = glob.glob(ze_loader_pattern)
        if ze_files:
            # Extract version from library filename
            for ze_file in ze_files:
                filename_version = re.search(r"\.so\.(\d+\.\d+\.\d+)", ze_file)
                if filename_version:
                    return filename_version.group(1)

    return "(unknown)"


def get_compute_runtime_version_detailed():
    """Get compute runtime version information"""
    # First check for optionally built Compute Runtime in workdir
    workdir_cr_src = os.path.join(options.workdir, "compute-runtime-src")

    # Try to get version from git tag in source directory
    if os.path.exists(workdir_cr_src):
        try:
            result = subprocess.run(
                ["git", "describe", "--tags"],
                cwd=workdir_cr_src,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                tag = result.stdout.strip()
                # Return the full git tag like "23.45.67890.1"
                return tag
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    return "(unknown)"


def get_gpu_info():
    """Get GPU information including device list and driver version"""
    gpu_list = []
    gpu_count = 0
    gpu_driver_version = "(unknown)"

    # Get GPU info from lspci
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_lines = [
                line
                for line in result.stdout.split("\n")
                if "VGA" in line or "Display" in line
            ]

            for line in gpu_lines:
                # Extract GPU name (after the first ": ")
                if ": " in line:
                    gpu_name = line.split(": ", 1)[1]
                    gpu_list.append(gpu_name)

            gpu_count = len(gpu_list)
    except Exception:
        gpu_list = ["Detection failed"]
        gpu_count = 0

    # Try to get GPU driver version
    try:
        # For Intel GPUs, try to get driver version - check both xe (newer Arc) and i915 (legacy)
        for driver in ["xe", "i915"]:
            result = subprocess.run(["modinfo", driver], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if driver == "xe" and line.startswith("vermagic:"):
                        # For xe driver, extract kernel version from vermagic field
                        parts = line.split()
                        if len(parts) >= 2:
                            kernel_version = parts[1]
                            gpu_driver_version = f"{driver} (kernel {kernel_version})"
                            break
                    elif driver == "i915" and line.startswith("version:"):
                        # For i915 driver, use traditional version field
                        driver_version = line.split(":", 1)[1].strip()
                        gpu_driver_version = f"{driver} {driver_version}"
                        break
                if gpu_driver_version != "(unknown)":
                    break
    except Exception:
        pass

    return gpu_list, gpu_count, gpu_driver_version


def get_platform_info() -> Platform:
    """Collect comprehensive platform information for Linux systems"""

    # Find project binaries and libraries
    bin_dir = find_project_binaries()

    # OS information
    os_info = f"{platform.system()} {platform.release()} {platform.version()}"

    # Python information
    python_info = f"{platform.python_implementation()} {platform.python_version()}"

    # CPU information
    cpu_info = "Unknown"
    cpu_count = os.cpu_count() or 0

    try:
        # Get CPU info from /proc/cpuinfo (Linux only)
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        for line in cpuinfo.split("\n"):
            if "model name" in line:
                cpu_info = line.split(":")[1].strip()
                break
    except Exception as e:
        cpu_info = f"Detection failed: {str(e)}"

    # Get GPU information
    gpu_list, gpu_count, gpu_driver_version = get_gpu_info()

    # Compiler versions - GCC from system, clang project-built
    gcc_version = "gcc (unknown)"
    clang_version = "clang (unknown)"

    try:
        # GCC version
        result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            gcc_version = result.stdout.split("\n")[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Only use project clang - assume it exists
    clang_version = get_project_clang_version(bin_dir)

    # Check compute runtime option first
    show_compute_runtime = (
        getattr(options.detect_versions, "compute_runtime", False)
        or options.build_compute_runtime
    )

    # Initialize runtime versions
    level_zero_version = ""
    compute_runtime_version = ""

    if show_compute_runtime:
        # Get Level Zero library version
        level_zero_raw = get_level_zero_version_detailed()

        # Get Compute Runtime version
        compute_runtime_version = get_compute_runtime_version_detailed()

        # Format Level Zero version with adapter info
        adapter_used = options.ur_adapter
        if adapter_used == "level_zero_v2":
            level_zero_version = f"L0 v2 adapter | level-zero {level_zero_raw}"
        elif adapter_used == "level_zero":
            level_zero_version = f"L0 v1 adapter | level-zero {level_zero_raw}"
        else:
            level_zero_version = f"level-zero {level_zero_raw}"

    # Always inform about L0 adapter if specified, even without compute runtime detection
    adapter_used = options.ur_adapter
    if not level_zero_version and adapter_used in ["level_zero", "level_zero_v2"]:
        adapter_name = (
            "L0 v2 adapter" if adapter_used == "level_zero_v2" else "L0 v1 adapter"
        )
        level_zero_version = f"{adapter_name} | level-zero (version unknown)"

    return Platform(
        timestamp=datetime.now().isoformat(),
        os=os_info,
        python=python_info,
        cpu_count=cpu_count,
        cpu_info=cpu_info,
        gpu_count=gpu_count,
        gpu_info=gpu_list,
        gpu_driver_version=gpu_driver_version,
        gcc_version=gcc_version,
        clang_version=clang_version,
        level_zero_version=level_zero_version,
        compute_runtime_version=compute_runtime_version,
    )
