"""CLI entry points for UR test tools."""
import sys
import os
from pathlib import Path

from .models.config import SummaryConfigFromLines, TestConfig, TestExecutionContext
from .validation.path_validator import PathValidator
from .parsers.log_parser import (
    read_log_file,
    extract_error_details,
)
from .outputs.console import filter_log_for_display
from .summary_generator import print_test_summary
from .test_runner import (
    TestRunner,
    get_test_config,
    check_log_has_tests,
)
from .outputs.github_actions import GitHubActionsOutput


def main() -> int:
    """Unified CLI entry point.

    Routes commands to appropriate handlers based on first argument.

    Returns:
        0 on success, 1 on error, >0 on test failure.
    """
    if len(sys.argv) < 2:
        print("Usage: ur-test <command> [args...]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Test execution commands:", file=sys.stderr)
        print(
            "  run <type> <build_dir> <workspace>    Run UR tests",
            file=sys.stderr
        )
        print(
            "  validate <build_dir> <workspace>      "
            "Validate build directory",
            file=sys.stderr
        )
        print(
            "  check-log <log_file>                  "
            "Check if log has tests",
            file=sys.stderr
        )
        print("", file=sys.stderr)
        print("Summary commands:", file=sys.stderr)
        print(
            "  summary <log_file> [xml_file]         Show test summary",
            file=sys.stderr
        )
        print(
            "  extract-errors <log_file>             Extract error details",
            file=sys.stderr
        )
        print(
            "  filter-log <log_file>                 "
            "Filter log for display",
            file=sys.stderr
        )
        return 1

    command = sys.argv[1]

    # Route to ci_utils commands
    if command in ("run", "validate", "check-log"):
        # Map friendly names to internal command names
        command_map = {
            "run": "run-tests",
            "validate": "validate-build-dir",
            "check-log": "check-log-has-tests",
        }
        sys.argv[1] = command_map[command]
        return main_ci_utils()

    # Route to test_summary commands
    elif command in ("summary", "extract-errors", "filter-log"):
        # Map friendly names to internal command names
        command_map = {
            "summary": "show-summary",
        }
        sys.argv[1] = command_map.get(command, command)
        return main_test_summary()

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("Run 'ur-test' without arguments for usage help.", file=sys.stderr)
        return 1


def main_test_summary() -> int:
    """Entry point for ur_test_summary CLI.

    Returns:
        0 on success, 1 on error.
    """
    try:
        if len(sys.argv) < 3:
            print(
                f"Error: {sys.argv[0]} <command> <log_file> [xml_file]",
                file=sys.stderr,
            )
            return 1

        command = sys.argv[1]

        log_file = sys.argv[2]
        PathValidator.validate_log_path(log_file)
        lines = read_log_file(log_file)

        if command == "extract-errors":
            for line in extract_error_details(lines):
                print(line, end="")

        elif command == "filter-log":
            for line in filter_log_for_display(lines):
                print(line, end="")

        elif command == "show-summary":
            xml_file = PathValidator.validate_optional_path(
                sys.argv[3] if len(sys.argv) > 3 else "",
                "XML",
                allow_absolute=True
            )
            config = SummaryConfigFromLines(
                log_lines=lines,
                xml_file=xml_file if xml_file else None
            )
            print_test_summary(config)

        else:
            print(f"Error: Unknown command '{command}'", file=sys.stderr)
            return 1

        return 0

    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main_ci_utils() -> int:
    """Entry point for ur_ci_utils CLI.

    Returns:
        0 on success, 1 on error, >0 on test failure.
    """
    if len(sys.argv) < 2:
        print(f"Error: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        return 1

    command = sys.argv[1]

    if command == "validate-build-dir":
        return _validate_build_dir_command()

    elif command == "check-log-has-tests":
        return _check_log_has_tests_command()

    elif command == "run-tests":
        return _run_tests_command()

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        return 1


def _validate_build_dir_command() -> int:
    """Execute validate-build-dir command."""
    if len(sys.argv) < 3:
        print(
            f"Error: validate-build-dir <build_dir> [workspace]",
            file=sys.stderr
        )
        return 1

    workspace = sys.argv[3] if len(sys.argv) > 3 else None
    is_valid = PathValidator.validate_build_dir(sys.argv[2], workspace)
    return 0 if is_valid else 1


def _check_log_has_tests_command() -> int:
    """Execute check-log-has-tests command."""
    if len(sys.argv) < 3:
        print(f"Error: check-log-has-tests <log_file>", file=sys.stderr)
        return 1

    has_tests = check_log_has_tests(sys.argv[2])
    return 0 if has_tests else 1


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
        config = get_test_config(test_type, build_dir)
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
        env=env
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
