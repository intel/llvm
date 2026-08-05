#!/usr/bin/env python3
"""CI utilities for UR test execution."""

import sys
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

MAX_LINES_TO_SCAN = 1000  # Scan first 1k lines when checking if log contains tests
# Rationale: LIT outputs "-- Testing: N tests, M workers --" at the START of run,
# typically within first 50 lines. 1000 lines provides safety margin for cmake
# build output that may precede test execution.
# Risk: If cmake/build produces >1000 lines of output before LIT starts, we may
# miss "Testing:" marker and incorrectly skip artifacts for adapter-specific tests.
# Historical data shows typical logs have <100 lines before "Testing:" marker.

MAX_JOBS = 16  # Cap on parallel cmake build jobs (nproc/3 capped at this value)
# Rationale: Controls `cmake --build -j N` parallelism, NOT test execution
# (LIT uses separate `-j 50` in LIT_OPTS). Prevents resource exhaustion:
#   - Memory: Each cmake job can use ~500MB-1GB during compilation/linking
#   - I/O: Too many parallel builds can saturate disk on shared CI runners
#   - On 96-core machines: nproc/3 = 32, capped to 16 to stay under ~16GB peak
# Tuning: If builds are I/O bound, consider lowering. If CPU bound, can increase
# but monitor peak memory usage (OOM kills observed above 20 on some runners).


@dataclass
class TestConfig:
    """Test execution configuration."""

    target: str
    log_file: str
    xml_search_path: str
    lit_filter_out: Optional[str] = None


def validate_build_dir(build_dir: str, workspace: Optional[str] = None) -> bool:
    """Validate build directory is safe and within workspace."""
    if not build_dir or ".." in build_dir or build_dir.startswith("/"):
        return False

    # Block shell metacharacters, quotes, and control characters
    # to prevent injection in f-strings, env vars, and logs
    dangerous_chars = {";", "&", "#", "$", "|", "`", "\\", "'", '"', "\n", "\r"}
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
            xml_search_path=f"{build_dir}/test/adapters",
            lit_filter_out=(
                "(adapters/level_zero/memcheck.test|"
                "adapters/level_zero/v2/deferred_kernel_memcheck.test)"
            ),
        )
    elif test_type == "conformance":
        return TestConfig(
            target="check-unified-runtime-conformance",
            log_file="conformance_tests.log",
            xml_search_path=f"{build_dir}/test/conformance",
        )
    else:
        raise ValueError(f"Invalid test_type: {test_type}")


def calculate_jobs() -> int:
    """Calculate number of parallel jobs (nproc/3 capped at MAX_JOBS)."""
    try:
        nproc = os.cpu_count() or 4
        return min(nproc // 3, MAX_JOBS)
    except (OSError, AttributeError):
        # Fallback if cpu_count fails or returns unexpected value
        return 4


def run_ur_tests(test_type: str, build_dir: str, workspace: str) -> int:
    """Run UR tests with full orchestration. Returns exit code."""
    if not validate_build_dir(build_dir, workspace):
        print("::error::Invalid build_dir", file=sys.stderr)
        return 1

    try:
        config: TestConfig = get_test_config(test_type, build_dir)
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1

    # Convert to Path and ensure all operations are relative to workspace
    workspace_path: Path = Path(workspace).resolve()

    env: Dict[str, str] = os.environ.copy()

    # Generate unique XML name to avoid literal *.xml filename
    xml_output_name: str = f"{test_type.replace('-', '_')}_results.xml"
    xml_output_path: Path = (
        workspace_path / config.xml_search_path / xml_output_name
    ).absolute()

    # Ensure XML output directory exists
    xml_output_path.parent.mkdir(parents=True, exist_ok=True)

    env["LIT_OPTS"] = (
        "--show-unsupported --show-pass --show-xfail --no-progress-bar "
        "-v --timeout 120 -j 50 --time-tests --show-flakypass "
        f"--show-skipped --xunit-xml-output {xml_output_path}"
    )
    if config.lit_filter_out:
        env["LIT_FILTER_OUT"] = config.lit_filter_out
    env["ZE_ENABLE_LOADER_DEBUG_TRACE"] = "1"

    jobs: int = calculate_jobs()
    cmake_cmd: List[str] = [
        "cmake",
        "--build",
        build_dir,
        "-j",
        str(jobs),
        "--",
        config.target,
    ]

    # Construct absolute log file path
    log_file_path: Path = workspace_path / config.log_file

    # Output configuration for GitHub Actions (always, before tests run)
    print(f"log_file={log_file_path}", flush=True)
    print(f"xml_search_path={workspace_path / config.xml_search_path}", flush=True)
    sys.stdout.flush()  # Ensure outputs are written before subprocess

    print(f"Running: {' '.join(cmake_cmd)}", file=sys.stderr)
    print(f"Log: {log_file_path}, Jobs: {jobs}", file=sys.stderr)
    print(f"Expected XML: {xml_output_path}", file=sys.stderr)

    try:
        with open(log_file_path, "w", encoding="utf-8") as log:
            # Use cmake with validated arguments - no user input, safe list form
            result: subprocess.CompletedProcess = subprocess.run(  # nosec B603 B607
                cmake_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=workspace_path,
            )
    except (OSError, PermissionError) as e:
        # OSError: cmake not found, invalid cwd, file I/O errors
        # PermissionError: no write permission for log file
        print(f"::error::Test execution failed: {e}", file=sys.stderr)
        return 1

    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        print("::error::No log generated", file=sys.stderr)
        return 1

    if test_type == "adapter-specific" and not check_log_has_tests(str(log_file_path)):
        print("No adapter-specific tests found", file=sys.stderr)
        print("skip_artifacts=1", flush=True)
        return 0

    if xml_output_path.exists():
        print(f"xml_file={xml_output_path.absolute()}", flush=True)
    else:
        print(
            f"Warning: Expected XML file not found at {xml_output_path}",
            file=sys.stderr,
        )

    return result.returncode


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Error: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate-build-dir":
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
