#!/usr/bin/env python3

# Copyright (C) 2026 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check NVML library and NVIDIA driver version compatibility."""

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, Tuple

COMMAND_TIMEOUT = 30
NVML_SEARCH_PATHS = ['/usr/lib', '/usr/lib64', '/usr/local/lib']


def get_driver_version() -> Optional[str]:
    """Get NVIDIA driver version from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
            timeout=COMMAND_TIMEOUT
        )
        version = result.stdout.strip()
        if not version:
            print("::error::Failed to get NVIDIA driver version")
            print("ERROR: nvidia-smi returned empty version", file=sys.stderr)
            return None
        return version
    except subprocess.CalledProcessError as e:
        print("::error::Failed to run nvidia-smi")
        print(f"ERROR: Command failed: {e}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("::error::nvidia-smi timeout")
        print(f"ERROR: Command timed out after {COMMAND_TIMEOUT}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("::error::nvidia-smi not found")
        print("ERROR: Not installed or not in PATH", file=sys.stderr)
        return None


def find_nvml_library() -> Optional[str]:
    """Find NVML library and extract its version."""
    for search_path in NVML_SEARCH_PATHS:
        try:
            path = Path(search_path)
            if not path.exists():
                continue
            
            for lib_file in path.rglob('libnvidia-ml.so.1'):
                real_path = lib_file.resolve()
                version_match = re.search(
                    r'\.so\.(\d+\.\d+(?:\.\d+)?)',
                    str(real_path)
                )
                if version_match:
                    return version_match.group(1)
        except (OSError, PermissionError):
            continue
    
    print("::error::NVML library not found in container")
    print(f"ERROR: libnvidia-ml.so.1 not found in {NVML_SEARCH_PATHS}", file=sys.stderr)
    return None


def parse_version(version: str) -> Optional[Tuple[int, ...]]:
    """Parse version string into tuple of integers."""
    try:
        return tuple(int(x) for x in version.split('.'))
    except (ValueError, AttributeError):
        return None


def compare_versions(driver_version: str, nvml_version: str) -> bool:
    """Check if NVML library version is compatible with driver version."""
    driver_parts = parse_version(driver_version)
    nvml_parts = parse_version(nvml_version)
    
    if not driver_parts or not nvml_parts:
        print("::error::Failed to parse version numbers")
        print(
            f"ERROR: Driver: {driver_version}, Library: {nvml_version}",
            file=sys.stderr
        )
        return False
    
    driver_major = driver_parts[0]
    nvml_major = nvml_parts[0]
    
    if driver_major != nvml_major:
        print(
            f"::error::NVML version mismatch - "
            f"Driver: {driver_version}, Library: {nvml_version}"
        )
        print(
            f"ERROR: Major versions differ ({driver_major} vs {nvml_major})",
            file=sys.stderr
        )
        return False
    
    if nvml_parts > driver_parts:
        print(
            f"::error::NVML version mismatch - "
            f"Driver: {driver_version}, Library: {nvml_version}"
        )
        print(
            f"ERROR: Library version ({nvml_version}) "
            f"is newer than driver ({driver_version})",
            file=sys.stderr
        )
        return False
    
    return True


def main() -> int:
    """Main entry point."""
    driver_version = get_driver_version()
    if not driver_version:
        return 1
    
    nvml_version = find_nvml_library()
    if not nvml_version:
        return 1
    
    if not compare_versions(driver_version, nvml_version):
        return 1
    
    print(
        f"NVML version check passed: "
        f"Driver {driver_version}, Library {nvml_version}"
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
