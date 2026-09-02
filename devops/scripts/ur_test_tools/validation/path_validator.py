"""Path validation for security."""

from pathlib import Path


class PathValidator:
    """Validate paths for security and correctness."""

    @staticmethod
    def validate_build_dir(build_dir: str, workspace: str) -> bool:
        """Check build_dir is relative and resolves within workspace."""
        if not build_dir:
            return False

        path = Path(build_dir)
        if path.is_absolute():
            return False

        workspace_path = Path(workspace).resolve()
        resolved = (workspace_path / path).resolve()
        try:
            resolved.relative_to(workspace_path)
        except ValueError:
            return False
        return True

    @staticmethod
    def validate_log_path(path: str) -> None:
        """Validate log file path has no traversal and exists."""
        if ".." in Path(path).parts:
            raise ValueError(f"Invalid log file path (path traversal): {path}")
        if not Path(path).exists():
            raise ValueError(f"Log file not found: {path}")

    @staticmethod
    def validate_optional_path(
        path: str, path_type: str, allow_absolute: bool = False
    ) -> str:
        """Validate optional file path has no traversal and matches absolute-path policy."""
        if not path:
            return ""

        if ".." in Path(path).parts:
            raise ValueError(f"Invalid {path_type} file path (path traversal): {path}")

        if not allow_absolute and Path(path).is_absolute():
            raise ValueError(
                f"Invalid {path_type} file path (absolute paths not allowed): {path}"
            )

        return path
