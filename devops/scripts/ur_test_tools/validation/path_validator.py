"""Path validation for security."""
from pathlib import Path
from typing import Optional


class PathValidator:
    """Validate paths for security and correctness."""

    @staticmethod
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

    @staticmethod
    def validate_log_path(path: str) -> None:
        """Validate log file path (detects path traversal)."""
        try:
            # Resolve path to detect encoded forms of path traversal (e.g., %2e%2e)
            resolved = Path(path).resolve(strict=False)

            # Check for path traversal in original string (simple check)
            if ".." in path:
                raise ValueError(
                    f"Invalid log file path (path traversal not allowed): {path}"
                )

            # Verify file exists
            if not resolved.exists():
                raise ValueError(f"Log file not found: {path}")
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                raise
            raise OSError(f"Invalid log file path: {path} ({e})") from e

    @staticmethod
    def validate_optional_path(
        path: str, path_type: str, allow_absolute: bool = False
    ) -> str:
        """Validate optional file path."""
        if not path:
            return ""

        try:
            # Resolve path to detect encoded forms of path traversal
            Path(path).resolve(strict=False)

            # Check for path traversal in original string
            if ".." in path:
                raise ValueError(
                    f"Invalid {path_type} file path (path traversal): {path}"
                )

            # Check absolute path restriction
            if not allow_absolute and path.startswith("/"):
                raise ValueError(
                    f"Invalid {path_type} file path "
                    f"(absolute paths not allowed): {path}"
                )
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError):
                raise
            raise OSError(f"Invalid {path_type} file path: {path} ({e})") from e

        return path

    @staticmethod
    def ensure_within_workspace(path: Path, workspace: Path) -> Path:
        """Ensure path is within workspace."""
        resolved = path.resolve()
        workspace_resolved = workspace.resolve()

        try:
            resolved.relative_to(workspace_resolved)
            return resolved
        except ValueError as e:
            raise ValueError(
                f"Path outside workspace: {path} not in {workspace}"
            ) from e
