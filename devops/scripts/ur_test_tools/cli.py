"""CLI entry points for UR test tools."""
import sys
import os
from pathlib import Path

from .models.config import SummaryConfigLegacy, TestConfig, TestExecutionContext
from .validation.path_validator import PathValidator
from .parsers.log_parser import (
    read_log_file,
    extract_error_details,
)
from .formatters.display import filter_log_for_display
from .orchestration.summary_generator import show_statistics_and_lists
from .orchestration.test_runner import (
    TestRunner,
    get_test_config,
    check_log_has_tests,
)
from .formatters.github_actions import GitHubActionsOutput


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
            config = SummaryConfigLegacy(
                log_lines=lines,
                xml_file=xml_file if xml_file else None
            )
            show_statistics_and_lists(config)

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
