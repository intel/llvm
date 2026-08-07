"""Configuration dataclasses for UR test tools."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List


@dataclass
class TestConfig:
    """Test execution configuration."""

    target: str
    log_file: str
    lit_filter_out: Optional[str] = None

    def __post_init__(self):
        """Validate configuration on creation."""
        if not all([self.target, self.log_file]):
            raise ValueError("target and log_file are required")


@dataclass(frozen=True)
class TestExecutionContext:
    """Context for test execution (immutable)."""

    test_type: str
    build_dir: Path
    workspace: Path
    xml_output_path: Path
    log_file_path: Path
    config: TestConfig
    env: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        try:
            workspace_resolved = self.workspace.resolve()
            for path in [self.log_file_path, self.xml_output_path, self.build_dir]:
                path.resolve().relative_to(workspace_resolved)
        except ValueError as e:
            raise ValueError(f"Path outside workspace: {e}") from e


@dataclass
class SummaryConfigFromLines:
    """Configuration for summary generation from parsed log lines."""

    log_lines: List[str]
    xml_file: Optional[str] = None
