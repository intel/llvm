"""CLI entry points for UR test tools."""

import sys
import os
from pathlib import Path

from .models.config import SummaryConfigFromLines, TestConfig, TestExecutionContext
from .validation.path_validator import PathValidator
from .parsers.log_parser import (
    LITLogParser,
    read_log_file,
)
from .outputs.console import ConsoleOutput
from .summary_generator import SummaryReporter
from .test_runner import (
    TestRunner,
    get_test_config,
)
from .outputs.github_actions import GitHubActionsOutput


def main() -> int:
    """Unified CLI entry point."""
    if len(sys.argv) < 2:
        print("Error: Missing command", file=sys.stderr)
        return 1

    command = sys.argv[1]

    if command == "run":
        return main_ci_utils("run-tests")

    elif command in ("summary", "extract-errors", "filter-log"):
        internal_cmd = "show-summary" if command == "summary" else command
        return main_test_summary(internal_cmd)

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        return 1


def main_test_summary(command: str) -> int:
    """Entry point for ur_test_summary CLI."""
    try:
        if len(sys.argv) < 3:
            print(
                f"Error: {sys.argv[0]} <command> <log_file> [xml_file]",
                file=sys.stderr,
            )
            return 1

        log_file = sys.argv[2]
        PathValidator.validate_log_path(log_file)
        lines = read_log_file(log_file)
        parser = LITLogParser(lines)

        if command == "extract-errors":
            for line in parser.extract_error_details():
                print(line, end="")

        elif command == "filter-log":
            for line in ConsoleOutput.filter_log_for_display(lines):
                print(line, end="")

        elif command == "show-summary":
            xml_file = PathValidator.validate_optional_path(
                sys.argv[3] if len(sys.argv) > 3 else "", "XML", allow_absolute=True
            )
            config = SummaryConfigFromLines(
                log_lines=lines, xml_file=xml_file or None
            )
            SummaryReporter(config).generate()

        else:
            print(f"Error: Unknown command '{command}'", file=sys.stderr)
            return 1

        return 0

    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main_ci_utils(command: str) -> int:
    """Entry point for ur_ci_utils CLI."""
    if command == "run-tests":
        return _run_tests_command()

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        return 1


def _run_tests_command() -> int:
    """Execute run-tests command."""
    if len(sys.argv) < 5:
        print(
            f"Error: run-tests <test_type> <build_dir> <workspace>",
            file=sys.stderr,
        )
        return 1

    test_type = sys.argv[2]
    build_dir = sys.argv[3]
    workspace = sys.argv[4]

    # Validate inputs
    gha = GitHubActionsOutput()
    if not PathValidator.validate_build_dir(build_dir, workspace):
        gha.print_error("Invalid build_dir")
        return 1

    try:
        config = get_test_config(test_type)
    except ValueError as e:
        gha.print_error(str(e))
        return 1

    # Convert to paths and create context
    workspace_path = Path(workspace).resolve()
    build_dir_path = workspace_path / build_dir

    xml_output_name = f"{test_type.replace('-', '_')}_results.xml"
    xml_output_path = (build_dir_path / xml_output_name).absolute()
    xml_output_path.parent.mkdir(parents=True, exist_ok=True)

    log_file_path = workspace_path / config.log_file

    env = os.environ.copy()

    context = TestExecutionContext(
        test_type=test_type,
        build_dir=build_dir_path,
        workspace=workspace_path,
        xml_output_path=xml_output_path,
        log_file_path=log_file_path,
        config=config,
        env=env,
    )

    # Validate context
    try:
        context.validate()
    except ValueError as e:
        gha.print_error(str(e))
        return 1

    # Run tests
    runner = TestRunner(context)
    return runner.run()
