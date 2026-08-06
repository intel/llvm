"""Orchestrate UR test execution."""
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from ..constants import (
    DEFAULT_LIT_TIMEOUT,
    DEFAULT_LIT_JOBS,
    TEST_TYPE_ADAPTER_SPECIFIC,
    MAX_LINES_TO_SCAN,
    MAX_JOBS,
)
from ..models.config import TestConfig, TestExecutionContext
from ..formatters.github_actions import GitHubActionsOutput


def get_test_config(test_type: str, build_dir: str) -> TestConfig:
    """Get test configuration based on test type.
    
    Args:
        test_type: Type of tests to run ('adapter-specific', 'conformance').
        build_dir: Build directory path (unused but kept for compatibility).
    
    Returns:
        TestConfig for the specified test type.
    
    Raises:
        ValueError: If test_type is invalid.
    """
    if test_type == "adapter-specific":
        return TestConfig(
            target="check-unified-runtime-adapter",
            log_file="adapter_tests.log",
            lit_filter_out=(
                "(adapters/level_zero/memcheck.test|"
                "adapters/level_zero/v2/deferred_kernel_memcheck.test)"
            ),
        )
    elif test_type == "conformance":
        return TestConfig(
            target="check-unified-runtime-conformance",
            log_file="conformance_tests.log",
        )
    else:
        raise ValueError(f"Invalid test_type: {test_type}")


def calculate_jobs() -> int:
    """Calculate number of parallel jobs (nproc/3 capped at MAX_JOBS).
    
    Returns:
        Number of parallel jobs to use for cmake builds.
    """
    try:
        nproc = os.cpu_count() or 4
        return min(nproc // 3, MAX_JOBS)
    except (OSError, AttributeError):
        # Fallback if cpu_count fails or returns unexpected value
        return 4


def check_log_has_tests(log_file: str) -> bool:
    """Check if log file contains test results.
    
    Scans the first MAX_LINES_TO_SCAN lines looking for "Testing:" marker.
    
    Args:
        log_file: Path to log file.
    
    Returns:
        True if log contains test results, False otherwise.
    """
    try:
        # Try strict decoding first
        with open(log_file, "r", encoding="utf-8", errors="strict") as f:
            for _ in range(MAX_LINES_TO_SCAN):
                line = f.readline()
                if not line:
                    break
                if "Testing:" in line:
                    return True
        return False
    except UnicodeDecodeError:
        # Fallback to replacement and log warning
        print(
            f"Warning: Log file {log_file} contains non-UTF-8 characters, "
            f"replacing with U+FFFD",
            file=sys.stderr,
        )
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
    except OSError:
        return False


class TestRunner:
    """Execute UR tests with full orchestration.
    
    Responsibilities clearly separated into focused methods.
    """

    def __init__(self, context: TestExecutionContext):
        """Initialize runner with validated context.
        
        Args:
            context: Test execution context with all configuration.
        """
        self.context = context
        self.gha = GitHubActionsOutput()

    def run(self) -> int:
        """Run tests and return exit code.
        
        High-level orchestration only - delegates to helper methods.
        
        Returns:
            Exit code (0 on success, 1 on error, >0 on test failures).
        """
        self._setup_environment()

        result = self._execute_tests()
        if result is None:
            return 1

        if not self._validate_output():
            return 1

        self._publish_outputs(result)
        return result.returncode

    def _setup_environment(self) -> None:
        """Configure environment variables for LIT execution."""
        lit_opts = (
            f"--show-unsupported --show-pass --show-xfail --no-progress-bar "
            f"-v --timeout {DEFAULT_LIT_TIMEOUT} -j {DEFAULT_LIT_JOBS} "
            f"--time-tests --show-flakypass --show-skipped "
            f"--xunit-xml-output {self.context.xml_output_path}"
        )
        self.context.env["LIT_OPTS"] = lit_opts

        if self.context.config.lit_filter_out:
            self.context.env["LIT_FILTER_OUT"] = self.context.config.lit_filter_out

        self.context.env["ZE_ENABLE_LOADER_DEBUG_TRACE"] = "1"

    def _build_cmake_command(self) -> List[str]:
        """Build cmake command with validated parameters.
        
        Returns:
            cmake command as list of arguments.
        """
        jobs = calculate_jobs()
        return [
            "cmake",
            "--build",
            str(self.context.build_dir),
            "-j",
            str(jobs),
            "--",
            self.context.config.target,
        ]

    def _execute_tests(self) -> Optional[subprocess.CompletedProcess]:
        """Execute cmake test command.
        
        Returns:
            CompletedProcess on success, None on error.
        """
        cmd = self._build_cmake_command()

        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        print(
            f"Log: {self.context.log_file_path}, "
            f"Jobs: {calculate_jobs()}",
            file=sys.stderr
        )
        print(
            f"Expected XML: {self.context.xml_output_path}",
            file=sys.stderr
        )

        try:
            with open(
                self.context.log_file_path, "w", encoding="utf-8"
            ) as log:
                return subprocess.run(  # nosec B603 B607
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self.context.env,
                    cwd=self.context.workspace,
                )
        except (OSError, PermissionError) as e:
            self.gha.print_error(f"Test execution failed: {e}")
            return None

    def _validate_output(self) -> bool:
        """Validate test output files were generated.
        
        Returns:
            True if output is valid, False otherwise.
        """
        log_path = self.context.log_file_path

        if not log_path.exists() or log_path.stat().st_size == 0:
            self.gha.print_error("No log generated")
            return False

        return True

    def _publish_outputs(self, result: subprocess.CompletedProcess) -> None:
        """Publish outputs for GitHub Actions.
        
        Args:
            result: Completed subprocess result.
        """
        self.gha.set_output("log-file", str(self.context.log_file_path))

        if (self.context.test_type == TEST_TYPE_ADAPTER_SPECIFIC and
                not check_log_has_tests(str(self.context.log_file_path))):
            print("No adapter-specific tests found", file=sys.stderr)
            self.gha.set_output("skip-artifacts", "1")
            return

        if self.context.xml_output_path.exists():
            self.gha.set_output("xml-file", str(self.context.xml_output_path))
        else:
            self.gha.print_warning(
                f"Expected XML file not found at {self.context.xml_output_path}"
            )
