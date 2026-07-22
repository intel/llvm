#!/usr/bin/env python3
"""CI utilities for UR test execution."""

import sys
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

MAX_LINES_TO_SCAN = 1000
MAX_JOBS = 16


@dataclass
class TestConfig:
    """Test execution configuration."""

    target: str
    log_file: str
    xml_name: str
    xml_search_path: str
    lit_filter_out: Optional[str] = None


def find_xml_file(search_path: str, xml_name: str) -> str:
    """Find XML file - checks search_path and common fallback locations."""
    if ".." in search_path or not search_path:
        return ""

    search_root = Path(search_path)
    
    # Handle wildcard for adapter-specific tests (searches in adapter subfolders)
    if xml_name == "*.xml":
        print(f"Note: Searching for XML files in adapter subfolders", file=sys.stderr)
        if search_root.exists() and search_root.is_dir():
            # Search in adapter subfolders: level_zero, cuda, hip, etc.
            for xml_path in search_root.rglob("*.xml"):
                if xml_path.is_file():
                    print(f"Note: Found XML at: {xml_path.absolute()}", file=sys.stderr)
                    return str(xml_path.absolute())
        print(f"Warning: No XML files found in {search_root}", file=sys.stderr)
        return ""

    # Standard search for named XML files
    search_locations = [
        search_root / xml_name,  # Primary: configured path
        search_root.parent / xml_name,  # Fallback: parent dir
        search_root.parent.parent / xml_name,  # Fallback: build root
    ]

    print(
        f"Note: Searching for {xml_name} in {len(search_locations)} locations:",
        file=sys.stderr,
    )
    for i, xml_path in enumerate(search_locations, 1):
        print(f"  {i}. {xml_path}", file=sys.stderr)
        try:
            if xml_path.exists() and xml_path.is_file():
                print(f"Note: Found XML at: {xml_path.absolute()}", file=sys.stderr)
                return str(xml_path.absolute())
        except (OSError, ValueError):
            pass

    # Fallback: recursive search in build directory (slower but thorough)
    build_root = search_root.parent.parent
    if build_root.exists() and build_root.is_dir():
        print(f"Note: Attempting recursive search in {build_root}", file=sys.stderr)
        for xml_path in build_root.rglob(xml_name):
            if xml_path.is_file():
                print(f"Note: Found XML at: {xml_path.absolute()}", file=sys.stderr)
                return str(xml_path.absolute())

    print(f"Warning: XML file {xml_name} not found in any location", file=sys.stderr)
    return ""


def validate_build_dir(build_dir: str, workspace: Optional[str] = None) -> bool:
    """Validate build directory is safe and within workspace."""
    if not build_dir or ".." in build_dir or build_dir.startswith("/"):
        return False

    dangerous_chars = {";", "&", "#", "$", "|", "`", "\\"}
    if any(c in build_dir for c in dangerous_chars):
        return False

    if workspace:
        try:
            build_path = Path(build_dir).resolve(strict=False)
            workspace_path = Path(workspace).resolve(strict=False)
            build_path.relative_to(workspace_path)
            return True
        except (ValueError, OSError):
            return False
    return True


def check_log_has_tests(log_file: str) -> bool:
    """Check if log file contains test results."""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(MAX_LINES_TO_SCAN):
                line = f.readline()
                if not line:
                    break
                if "Testing:" in line:
                    return True
        return False
    except OSError:
        return False


def get_test_config(test_type: str, build_dir: str) -> TestConfig:
    """Get test configuration based on test type."""
    if test_type == "adapter-specific":
        return TestConfig(
            target="check-unified-runtime-adapter",
            log_file="adapter_tests.log",
            xml_name="*.xml",  # Wildcard - adapters may have different XML names
            xml_search_path=f"{build_dir}/test/adapters",
            lit_filter_out="(adapters/level_zero/memcheck.test|adapters/level_zero/v2/deferred_kernel_memcheck.test)",
        )
    elif test_type == "conformance":
        return TestConfig(
            target="check-unified-runtime-conformance",
            log_file="conformance_tests.log",
            xml_name="conformance_results.xml",
            xml_search_path=f"{build_dir}/test/conformance",
        )
    else:
        raise ValueError(f"Invalid test_type: {test_type}")


def calculate_jobs() -> int:
    """Calculate number of parallel jobs (nproc/3 capped at MAX_JOBS)."""
    try:
        nproc = os.cpu_count() or 4
        return min(nproc // 3, MAX_JOBS)
    except Exception:
        return 4


def run_ur_tests(test_type: str, build_dir: str, workspace: str) -> int:
    """Run UR tests with full orchestration. Returns exit code."""
    if not validate_build_dir(build_dir, workspace):
        print("::error::Invalid build_dir", file=sys.stderr)
        return 1

    try:
        config = get_test_config(test_type, build_dir)
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["LIT_OPTS"] = (
        "--show-unsupported --show-pass --show-xfail --no-progress-bar "
        "--succinct --timeout 120 -j 50 --time-tests --show-flakypass "
        f"--show-skipped --xunit-xml-output {config.xml_name}"
    )
    if config.lit_filter_out:
        env["LIT_FILTER_OUT"] = config.lit_filter_out
    env["ZE_ENABLE_LOADER_DEBUG_TRACE"] = "1"

    jobs = calculate_jobs()
    cmake_cmd = ["cmake", "--build", build_dir, "-j", str(jobs), "--", config.target]

    # Output configuration for GitHub Actions (always, before tests run)
    print(f"log_file={config.log_file}", flush=True)
    print(f"xml_name={config.xml_name}", flush=True)
    print(f"xml_search_path={config.xml_search_path}", flush=True)
    sys.stdout.flush()  # Ensure outputs are written before subprocess

    print(f"Running: {' '.join(cmake_cmd)}", file=sys.stderr)
    print(f"Log: {config.log_file}, Jobs: {jobs}", file=sys.stderr)

    try:
        with open(config.log_file, "w", encoding="utf-8") as log:
            # Use cmake with validated arguments - no user input, safe list form
            result = subprocess.run(  # nosec B603 B607
                cmake_cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd="."
            )
    except Exception as e:
        print(f"::error::Test execution failed: {e}", file=sys.stderr)
        return 1

    if not Path(config.log_file).exists() or Path(config.log_file).stat().st_size == 0:
        print("::error::No log generated", file=sys.stderr)
        return 1

    if test_type == "adapter-specific" and not check_log_has_tests(config.log_file):
        print("No adapter-specific tests found", file=sys.stderr)
        print("skip_artifacts=1", flush=True)
        return 0

    return result.returncode


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Error: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "find-xml":
        if len(sys.argv) < 4:
            print(f"Error: find-xml <search_path> <xml_name>", file=sys.stderr)
            sys.exit(1)
        result = find_xml_file(sys.argv[2], sys.argv[3])
        print(result, flush=True)

    elif command == "validate-build-dir":
        if len(sys.argv) < 3:
            print(f"Error: validate-build-dir <build_dir> [workspace]", file=sys.stderr)
            sys.exit(1)
        workspace = sys.argv[3] if len(sys.argv) > 3 else None
        is_valid = validate_build_dir(sys.argv[2], workspace)
        sys.exit(0 if is_valid else 1)

    elif command == "check-log-has-tests":
        if len(sys.argv) < 3:
            print(f"Error: check-log-has-tests <log_file>", file=sys.stderr)
            sys.exit(1)
        has_tests = check_log_has_tests(sys.argv[2])
        sys.exit(0 if has_tests else 1)

    elif command == "run-tests":
        if len(sys.argv) < 5:
            print(
                f"Error: run-tests <test_type> <build_dir> <workspace>",
                file=sys.stderr,
            )
            sys.exit(1)
        exit_code = run_ur_tests(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(exit_code)

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
