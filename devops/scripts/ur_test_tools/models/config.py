"""Configuration dataclasses for UR test tools."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List


@dataclass
class TestConfig:
    """Test execution configuration.

    Attributes:
        target: CMake target to build (e.g., 'check-unified-runtime-adapter').
        log_file: Name of the log file to write test output.
        lit_filter_out: Optional LIT_FILTER_OUT pattern to exclude tests.
    """

    target: str
    log_file: str
    lit_filter_out: Optional[str] = None

    def __post_init__(self):
        """Validate configuration on creation."""
        if not self.target or not self.log_file:
            raise ValueError("target and log_file are required")


@dataclass(frozen=True)
class TestExecutionContext:
    """Context for test execution (immutable).

    Attributes:
        test_type: Type of tests to run ('adapter-specific', 'unit').
        build_dir: Path to the build directory.
        workspace: Path to the workspace root.
        xml_output_path: Path where XML test results will be written.
        log_file_path: Path where test log will be written.
        config: Test configuration.
        env: Environment variables for test execution.
    """

    test_type: str
    build_dir: Path
    workspace: Path
    xml_output_path: Path
    log_file_path: Path
    config: TestConfig
    env: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        try:
            # Ensure log file is within workspace
            self.log_file_path.resolve().relative_to(self.workspace.resolve())

            # Ensure XML output is within workspace
            self.xml_output_path.resolve().relative_to(self.workspace.resolve())

            # Ensure build dir is within workspace
            self.build_dir.resolve().relative_to(self.workspace.resolve())
        except ValueError as e:
            raise ValueError(f"Path outside workspace: {e}") from e


@dataclass
class SummaryConfigFromLines:
    """Configuration for summary generation from parsed log lines.

    Attributes:
        log_lines: List of log file lines.
        xml_file: Optional path to XML file as string.
    """

    log_lines: List[str]
    xml_file: Optional[str] = None
