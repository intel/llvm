"""Test execution."""

import os
import sys
import subprocess  # nosec B404 - Used safely with list args, no shell=True
from pathlib import Path
from typing import List, Optional

from .constants import (
    DEFAULT_LIT_TIMEOUT,
    DEFAULT_LIT_JOBS,
    TEST_TYPE_ADAPTER_SPECIFIC,
    TEST_TYPE_CONFORMANCE,
    LIT_FILTER_OUT_ADAPTER_SPECIFIC,
    MAX_LINES_TO_SCAN,
)
from .models.config import TestConfig, TestExecutionContext
from .outputs.github_actions import GitHubActionsOutput
from .parsers.log_parser import _read_with_utf8_fallback


def get_test_config(test_type: str) -> TestConfig:
    """Get test configuration for test type."""
    if test_type == TEST_TYPE_ADAPTER_SPECIFIC:
        return TestConfig(
            target="check-unified-runtime-adapter",
            log_file="adapter_tests.log",
            lit_filter_out=LIT_FILTER_OUT_ADAPTER_SPECIFIC,
        )
    elif test_type == TEST_TYPE_CONFORMANCE:
        return TestConfig(
            target="check-unified-runtime-conformance",
            log_file="conformance_tests.log",
        )
    else:
        raise ValueError(f"Invalid test_type: {test_type}")


def calculate_jobs() -> int:
    """Calculate parallel jobs (nproc/3, min 1)."""
    try:
        nproc = os.cpu_count() or 4
        return max(1, nproc // 3)
    except (OSError, AttributeError):
        return 4


def check_log_has_tests(log_file: str) -> bool:
    """Check if log contains test results."""

    def _scan_for_testing(f):
        for _ in range(MAX_LINES_TO_SCAN):
            line = f.readline()
            if not line:
                break
            if "Testing:" in line:
                return True
        return False

    try:
        return _read_with_utf8_fallback(log_file, _scan_for_testing)
    except OSError:
        return False


class TestRunner:
    """Execute UR tests."""

    def __init__(self, context: TestExecutionContext):
        self.context = context
        self.github_output = GitHubActionsOutput()
        self.jobs = calculate_jobs()

    def run(self) -> int:
        """Run tests and return exit code."""
        self._setup_environment()

        result = self._execute_tests()
        if result is None:
            return 1

        if not self._validate_output():
            return 1

        self._publish_outputs(result)
        return result.returncode

    def _setup_environment(self) -> None:
        lit_opts = (
            f"--show-unsupported --show-pass --show-xfail --no-progress-bar "
            f"-v --timeout {DEFAULT_LIT_TIMEOUT} -j {DEFAULT_LIT_JOBS} "
            f"--time-tests --show-flakypass --show-skipped "
            f"--xunit-xml-output {self.context.xml_output_path}"
        )
        self.context.env["LIT_OPTS"] = lit_opts

        if self.context.config.lit_filter_out:
            self.context.env["LIT_FILTER_OUT"] = self.context.config.lit_filter_out

    def _build_cmake_command(self) -> List[str]:
        return [
            "cmake",
            "--build",
            str(self.context.build_dir),
            "-j",
            str(self.jobs),
            "--",
            self.context.config.target,
        ]

    def _execute_tests(self) -> Optional[subprocess.CompletedProcess]:
        cmd = self._build_cmake_command()

        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        print(f"Log: {self.context.log_file_path}, Jobs: {self.jobs}", file=sys.stderr)
        print(f"Expected XML: {self.context.xml_output_path}", file=sys.stderr)

        try:
            with open(self.context.log_file_path, "w", encoding="utf-8") as log:
                return subprocess.run(  # nosec B603 B607
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self.context.env,
                    cwd=self.context.workspace,
                )
        except (OSError, PermissionError) as e:
            self.github_output.print_error(f"Test execution failed: {e}")
            return None

    def _validate_output(self) -> bool:
        log_path = self.context.log_file_path

        if not log_path.exists() or log_path.stat().st_size == 0:
            self.github_output.print_error("No log generated")
            return False

        return True

    def _publish_outputs(self, result: subprocess.CompletedProcess) -> None:
        self.github_output.set_output("log-file", str(self.context.log_file_path))

        if (
            self.context.test_type == TEST_TYPE_ADAPTER_SPECIFIC
            and not check_log_has_tests(str(self.context.log_file_path))
        ):
            print("No adapter-specific tests found", file=sys.stderr)
            self.github_output.set_output("skip-artifacts", "1")
            return

        if self.context.xml_output_path.exists():
            self.github_output.set_output("xml-file", str(self.context.xml_output_path))
        else:
            self.github_output.print_warning(
                f"Expected XML file not found at {self.context.xml_output_path}"
            )
