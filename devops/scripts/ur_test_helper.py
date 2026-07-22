#!/usr/bin/env python3
"""Helper utilities for UR test execution in CI."""

import sys
from pathlib import Path
from typing import Optional

MAX_LINES_TO_SCAN = 1000

def find_xml_file(search_path: str, xml_name: str) -> str:
    """Find XML file at top-level of search_path.
    
    LIT always outputs XML to its working directory (search_path),
    so there's no need for recursive search.
    """
    if ".." in search_path or not search_path:
        return ""
    
    try:
        xml_path = Path(search_path) / xml_name
        if xml_path.exists() and xml_path.is_file():
            return str(xml_path.absolute())
    except (OSError, ValueError):
        pass
    
    return ""


def validate_build_dir(build_dir: str, workspace: Optional[str] = None) -> bool:
    """Validate build directory is safe and within workspace."""
    if not build_dir or ".." in build_dir:
        return False
    
    if build_dir.startswith("/"):
        return False
    
    # Check for shell metacharacters
    dangerous_chars = {";", "&", "#", "$", "|", "`", "\\"}
    if any(c in build_dir for c in dangerous_chars):
        return False
    
    if workspace:
        try:
            build_path = Path(build_dir).resolve(strict=False)
            workspace_path = Path(workspace).resolve(strict=False)
            try:
                build_path.relative_to(workspace_path)
                return True
            except ValueError:
                return False
        except (ValueError, OSError):
            return False
    return True


def check_log_has_tests(log_file: str) -> bool:
    """Check if log file contains actual test results."""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            # Read first MAX_LINES_TO_SCAN lines to find "Testing:" marker
            for _ in range(MAX_LINES_TO_SCAN):
                line = f.readline()
                if not line:
                    break
                if "Testing:" in line:
                    return True
        return False
    except OSError:
        return False


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
        print(result)
    
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
    
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
